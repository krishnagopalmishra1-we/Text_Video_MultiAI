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
        self.dtype = torch.bfloat16

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import WanPipeline
            from diffusers.utils import export_to_video
        except ImportError:
            raise ImportError("Install: pip install diffusers transformers accelerate")

        logger.info("Loading Wan2.1-14B (bfloat16)…")
        model_id = self.cfg.get("hf_id", _DEFAULT_MODEL_ID)
        try:
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant="bf16",
            ).to(self.device)
        except Exception as e:
            # Some repos do not publish a bf16 variant name even when bf16 weights exist.
            logger.warning(f"Wan2 load with variant=bf16 failed, retrying without variant: {e}")
            self.pipe = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
            ).to(self.device)

        if self.cfg.get("flash_attention"):
            logger.info("Skipping xformers memory-efficient attention for Wan2; it is unstable with this pipeline build.")

        if self.cfg.get("compile"):
            # Keep compile enabled for speed, but allow safe eager fallback if
            # Triton/Inductor compilation fails at runtime.
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune",
                backend="inductor",
            )
        logger.info("Wan2.1-14B loaded.")

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

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
        num_inference_steps: int = 25,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 16)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 81))
        # Wan2 expects odd frame counts
        if num_frames % 2 == 0:
            num_frames += 1

        gen = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with torch.autocast("cuda", dtype=self.dtype):
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
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
        logger.info(f"Wan2.1 → {out_path}")
        return out_path
