"""
Wan2.1-T2V wrapper — primary model on A100 80GB.
~40GB VRAM in bfloat16. Best quality/speed tradeoff.

LoRA support:
  Configure in model_config.yaml under models.local.wan2_14b.lora_weights:
    - path: "username/my-lora"        # HF repo or local path
      weight_name: "lora.safetensors" # optional
      scale: 0.85                     # 0.0–1.0
      name: "cinematic"               # adapter label (optional)
  LoRAs are fused into weights at model load — no per-inference overhead.
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

        logger.info("Loading Wan2.1 (%s)...", self.cfg.get("hf_id", _DEFAULT_MODEL_ID))
        model_id = self.cfg.get("hf_id", _DEFAULT_MODEL_ID)
        try:
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant=self.variant,
            ).to(self.device)
        except Exception as e:
            logger.warning(f"Wan2 load with variant={self.variant} failed, retrying without variant: {e}")
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
            ).to(self.device)

        # ── 1. VAE optimization ──────────────────────────────────────
        if self.cfg.get("vae_tiling", False):
            self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        logger.info("VAE slicing enabled.%s", " Tiling enabled." if self.cfg.get("vae_tiling", False) else "")

        # ── 2. Attention backend ─────────────────────────────────────
        # SageAttention > Flash Attention 2 (1.5-1.7× faster, drop-in)
        _sage_ok = False
        if self.cfg.get("sage_attention", False):
            try:
                from sageattention import sageattn
                torch.nn.functional.scaled_dot_product_attention = sageattn
                _sage_ok = True
                logger.info("SageAttention monkey-patched (replaces FA2).")
            except ImportError:
                logger.warning("sageattention not installed — falling back to flash_attention.")

        if not _sage_ok and self.cfg.get("flash_attention"):
            try:
                self.pipe.transformer.enable_flash_attn()
                logger.info("Flash Attention enabled on transformer.")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e} -- skipping.")

        # ── 3. Attention caching (PAB) ───────────────────────────────
        if self.cfg.get("teacache") or self.cfg.get("pab"):
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
                logger.warning(f"PAB unavailable: {e}")

        # ── 4. TeaCache (step-level caching) ─────────────────────────
        if self.cfg.get("tea_cache"):
            try:
                from diffusers.hooks import apply_tea_cache
                from diffusers.hooks.tea_cache import TeaCacheConfig
                tea_cfg = TeaCacheConfig(
                    threshold=self.cfg.get("teacache_thresh", 0.15),
                )
                apply_tea_cache(self.pipe.transformer, tea_cfg)
                logger.info("TeaCache applied.")
            except Exception as e:
                logger.warning(f"TeaCache unavailable: {e}")

        # ── 5. First-Block Cache (FBCache) ───────────────────────────
        if self.cfg.get("fb_cache"):
            try:
                from diffusers.hooks import apply_first_block_cache
                from diffusers.hooks.first_block_cache import FirstBlockCacheConfig
                fb_cfg = FirstBlockCacheConfig(threshold=0.15)
                apply_first_block_cache(self.pipe.transformer, fb_cfg)
                logger.info("First-Block Cache (FBCache) applied.")
            except Exception as e:
                logger.warning(f"FBCache unavailable: {e}")

        # ── 6. LoRA weights — fuse before compile ────────────────────
        self._apply_loras()

        # ── 7. torch.compile — AFTER all hooks, text encoder stays on GPU ──
        # B2 fix: compile traces the transformer assuming all modules are on
        # the same device. Moving text_encoder to CPU AFTER compile would
        # cause device mismatch on inference. So: compile first, THEN
        # optionally offload text_encoder only if compile is OFF.
        if self.cfg.get("compile"):
            from torch import _dynamo
            _dynamo.config.suppress_errors = True
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune",
                backend="inductor",
                fullgraph=False,
            )
            logger.info("torch.compile applied (mode=max-autotune).")
            # Keep text encoder on GPU — compile needs consistent devices.
            # ~2 GB is trivial on A100 80 GB.
            logger.info("Text encoder stays on GPU (compile compatibility).")
        else:
            # No compile — safe to offload text encoder to CPU
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Text encoder moved to CPU to free VRAM.")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_loras(self) -> None:
        """Load, fuse, and unload LoRA weights from config.
        Fusing bakes LoRA deltas into base weights — zero per-inference overhead.
        """
        loras = self.cfg.get("lora_weights", [])
        if not loras:
            return

        adapter_names: list[str] = []
        adapter_scales: list[float] = []

        for i, lc in enumerate(loras):
            path = lc.get("path") or lc.get("hf_id")
            if not path:
                logger.warning(f"Wan2 LoRA entry {i} missing 'path'/'hf_id' — skipping")
                continue
            scale = float(lc.get("scale", 0.8))
            weight_name = lc.get("weight_name")
            adapter_name = lc.get("name", f"lora_{i}")

            load_kwargs: dict = {
                "pretrained_model_name_or_path_or_dict": path,
                "adapter_name": adapter_name,
            }
            if weight_name:
                load_kwargs["weight_name"] = weight_name

            try:
                self.pipe.load_lora_weights(**load_kwargs)
                adapter_names.append(adapter_name)
                adapter_scales.append(scale)
                logger.info(f"Loaded LoRA '{adapter_name}' from {path} (scale={scale})")
            except Exception as e:
                logger.warning(f"Failed to load LoRA '{adapter_name}' from {path}: {e}")

        if not adapter_names:
            return

        try:
            self.pipe.set_adapters(adapter_names, adapter_weights=adapter_scales)
            self.pipe.fuse_lora()
            self.pipe.unload_lora_weights()
            logger.info(f"Fused {len(adapter_names)} LoRA adapter(s) into Wan2 weights.")
        except Exception as e:
            logger.warning(f"LoRA fuse failed — running without LoRA: {e}")
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

    # ------------------------------------------------------------------

    @property
    def _text_encoder_on_cpu(self) -> bool:
        """True if text encoder was offloaded to CPU (compile is off)."""
        if self.pipe is None:
            return False
        return next(self.pipe.text_encoder.parameters()).device.type == "cpu"

    def warmup(self) -> None:
        """Run a tiny inference to trigger torch.compile cache. Call once per worker startup."""
        if self._warmed_up:
            return
        self.load()
        logger.info("Warmup: triggering torch.compile with dummy inference...")
        if self._text_encoder_on_cpu:
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
        if self._text_encoder_on_cpu:
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

        if self._text_encoder_on_cpu:
            self.pipe.text_encoder.to(self.device)
        embeds = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        if self._text_encoder_on_cpu:
            self.pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        self._prompt_cache[cache_key] = embeds
        return embeds

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = (
            "blurry, out of focus, low resolution, pixelated, compression artifacts, "
            "watermark, text overlay, subtitles, logo, distorted, deformed, "
            "cartoon, anime, illustration, painting, CGI, 3D render, "
            "overexposed, underexposed, washed out, oversaturated, noise, grain, "
            "shaky camera, jerky motion, duplicate frames, worst quality, low quality"
        ),
        duration: float = 5.0,
        output_path: str | Path | None = None,
        fps: int | None = None,
        width: int = 1280,
        height: int = 720,
        guidance_scale: float = 7.0,   # raised from 5.0 — better prompt adherence
        num_inference_steps: int = 20,
        seed: int | None = None,
        progress_callback=None,
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

        def _step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step=step, total_steps=num_inference_steps)
            return callback_kwargs

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
                callback_on_step_end=_step_callback,
            )

        frames = output.frames[0]
        out_path = Path(output_path) if output_path else Path(f"/tmp/wan2_{seed or 0}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=_fps)
        logger.info(f"Wan2.1 -> {out_path}")
        return out_path
