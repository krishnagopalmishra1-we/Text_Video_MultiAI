"""
Wan2.1-T2V-14B wrapper — primary model on A100 80GB.
~40GB VRAM in bfloat16. Best quality/speed tradeoff.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


class Wan2Runner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.device = torch.device("cuda")
        dtype_cfg = str(self.cfg.get("dtype", "bf16")).lower()
        self.dtype = torch.float16 if dtype_cfg in {"fp16", "float16"} else torch.bfloat16
        self.variant = "fp16" if self.dtype == torch.float16 else "bf16"
        self._prompt_cache: dict[str, object] = {}
        self._warmed_up = False

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import WanPipeline
        except ImportError:
            raise ImportError("Install: pip install diffusers transformers accelerate")

        logger.info("Loading Wan2.1-14B (bfloat16)...")
        model_id = self.cfg.get("hf_id", _DEFAULT_MODEL_ID)
        try:
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant=self.variant,
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Wan2 load with variant=bf16 failed, retrying without variant: {e}")
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
            ).to(self.device)

        # Tiling hits a slow_conv3d CUDA path on this stack; slicing is sufficient on A100 80GB.
        if self.cfg.get("vae_tiling", False):
            self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        logger.info("VAE slicing enabled.%s", " Tiling enabled." if self.cfg.get("vae_tiling", False) else "")

        if self.cfg.get("flash_attention"):
            try:
                self.pipe.transformer.enable_flash_attn()
                logger.info("Flash Attention enabled on transformer.")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e} -- skipping.")

        if self.cfg.get("teacache"):
            try:
                from diffusers.hooks import apply_pyramid_attention_broadcast
                from diffusers.hooks.pyramid_attention_broadcast import PyramidAttentionBroadcastConfig
                pab_config = PyramidAttentionBroadcastConfig(
                    spatial_attention_block_skip_range=2,
                    temporal_attention_block_skip_range=2,
                    cross_attention_block_skip_range=2,
                    spatial_attention_timestep_skip_range=(100, 800),
                    temporal_attention_timestep_skip_range=(100, 800),
                    cross_attention_timestep_skip_range=(100, 800),
                )
                apply_pyramid_attention_broadcast(self.pipe.transformer, pab_config)
                logger.info("PAB (Pyramid Attention Broadcast) applied.")
            except Exception as e:
                logger.warning(f"PAB unavailable, trying TaylorSeer: {e}")
                try:
                    from diffusers.hooks import apply_taylorseer_cache
                    from diffusers.hooks.taylorseer_cache import TaylorSeerCacheConfig
                    ts_config = TaylorSeerCacheConfig()
                    apply_taylorseer_cache(self.pipe.transformer, ts_config)
                    logger.info("TaylorSeer cache applied.")
                except Exception as e2:
                    logger.warning(f"TaylorSeer also unavailable: {e2}")

        if self.cfg.get("compile"):
            from torch import _dynamo

            _dynamo.config.suppress_errors = True
            compile_mode = "reduce-overhead" if self.cfg.get("teacache") else "max-autotune"
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode=compile_mode,
                backend="inductor",
            )
            logger.info(f"torch.compile applied (mode={compile_mode}).")

        # Move text encoder to CPU -- saves ~2GB VRAM, only needed at prompt-encode time
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        logger.info("Text encoder moved to CPU to free VRAM.")

    def warmup(self) -> None:
        """Run a tiny inference to trigger torch.compile cache. Call once per worker startup."""
        if self._warmed_up:
            return
        self.load()
        logger.info("Warmup: triggering torch.compile with dummy inference...")
        self.pipe.text_encoder.to(self.device)
        with torch.autocast("cuda", dtype=self.dtype):
            self.pipe(
                prompt="warmup",
                negative_prompt="",
                num_frames=5,
                width=128,
                height=128,
                guidance_scale=1.0,
                num_inference_steps=2,
            )
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        self._warmed_up = True
        logger.info("Warmup complete -- compile cache is hot.")

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        self._prompt_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_prompt(self, prompt: str, negative_prompt: str):
        """Cache text embeddings so the text encoder isn't re-run for repeated prompts."""
        cache_key = f"{prompt}|{negative_prompt}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        self.pipe.text_encoder.to(self.device)
        embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        self._prompt_cache[cache_key] = embeds
        return embeds

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text, cartoon",
        duration: float = 5.0,
        output_path: str | Path | None = None,
        fps: int | None = None,
        width: int = 1280,
        height: int = 720,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 16)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 81))
        if num_frames % 2 == 0:
            num_frames += 1

        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )

        # Pre-encode prompt (cached across scenes with same prompt)
        prompt_embeds = self._encode_prompt(prompt, negative_prompt)

        with torch.autocast("cuda", dtype=self.dtype):
            output = self.pipe(
                prompt_embeds=prompt_embeds[0],
                negative_prompt_embeds=prompt_embeds[1],
                num_frames=num_frames,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=gen,
            )

        frames = output.frames[0]
        out_path = Path(output_path) if output_path else Path(f"/tmp/wan2_{seed or 0}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=_fps)
        logger.info(f"Wan2.1 -> {out_path}")
        return out_path
