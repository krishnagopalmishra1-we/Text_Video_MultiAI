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

_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B"


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
        self.pipe = WanPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=self.dtype,
            variant="bf16",
        ).to(self.device)

        if self.cfg.get("flash_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        if self.cfg.get("compile"):
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
        num_inference_steps: int = 50,
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
