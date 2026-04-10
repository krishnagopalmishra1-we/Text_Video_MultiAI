"""
LTX-Video wrapper — 10GB VRAM, fastest local model.
Used for previews and fallback when VRAM is constrained.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "Lightricks/LTX-Video"


class LTXRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import LTXPipeline
        except ImportError:
            raise ImportError("Install diffusers>=0.32.0")

        logger.info("Loading LTX-Video (bfloat16)…")
        self.pipe = LTXPipeline.from_pretrained(
            _MODEL_ID, torch_dtype=self.dtype
        ).to(self.device)

        if self.cfg.get("compile"):
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="reduce-overhead"
            )
        logger.info("LTX-Video loaded.")

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
        negative_prompt: str = "blurry, low quality, watermark, cartoon, worst quality",
        duration: float = 5.0,
        output_path: str | Path | None = None,
        fps: int | None = None,
        width: int = 768,
        height: int = 512,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 24)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 121))

        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )

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
        out_path = Path(output_path) if output_path else Path(f"/tmp/ltx_{seed or 0}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=_fps)
        logger.info(f"LTX-Video → {out_path}")
        return out_path
