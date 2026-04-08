"""
CogVideoX-5B wrapper — 24GB VRAM, faster than Wan2/Hunyuan.
Used for bulk scenes and when throughput > quality.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "THUDM/CogVideoX-5b"


class CogVideoXRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import CogVideoXPipeline
        except ImportError:
            raise ImportError("Install diffusers>=0.30.0")

        logger.info("Loading CogVideoX-5B (bfloat16)…")
        self.pipe = CogVideoXPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        if self.cfg.get("flash_attention"):
            try:
                self.pipe.transformer.enable_flash_attn()
            except Exception:
                pass

        if self.cfg.get("compile"):
            self.pipe.transformer = torch.compile(
                self.pipe.transformer, mode="reduce-overhead", backend="inductor"
            )
        logger.info("CogVideoX-5B loaded.")

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
        negative_prompt: str = "blurry, low quality, watermark, cartoon",
        duration: float = 6.0,
        output_path: str | Path | None = None,
        fps: int | None = None,
        width: int = 720,
        height: int = 480,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 8)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 49))

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
        out_path = Path(output_path) if output_path else Path(f"/tmp/cogvx_{seed or 0}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=_fps)
        logger.info(f"CogVideoX-5B → {out_path}")
        return out_path
