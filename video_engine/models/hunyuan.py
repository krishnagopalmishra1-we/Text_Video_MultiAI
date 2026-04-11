"""
HunyuanVideo wrapper — cinema quality, ~60GB VRAM on A100 80GB.
Used for hero/key scenes where quality > throughput.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "hunyuanvideo-community/HunyuanVideo"


class HunyuanRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pipe = None
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def load(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import HunyuanVideoPipeline
            from diffusers.models.transformers.transformer_hunyuan_video import (
                HunyuanVideoTransformer3DModel,
            )
        except ImportError:
            raise ImportError("Install diffusers>=0.32.0")

        model_id = self.cfg.get("hf_id", _MODEL_ID)
        quant = self.cfg.get("quantization")  # "int8", "int4", or None

        if quant == "int8":
            logger.info(f"Loading HunyuanVideo INT8-quantized (A100 tensor-core accelerated)…")
            from diffusers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        elif quant == "int4":
            logger.info(f"Loading HunyuanVideo NF4-quantized (~20GB VRAM)…")
            from diffusers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            logger.info("Loading HunyuanVideo (bfloat16, no quantization)…")
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )

        self.pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)
        self.pipe.vae.enable_tiling()
        if self.cfg.get("offload") == "cpu":
            # Optional fallback: keep text encoder on CPU to save VRAM.
            # Must come after pipe.to(self.device) so only the text encoder
            # is moved back, leaving the rest of the pipeline on the GPU.
            self.pipe.text_encoder.to("cpu")

        if self.cfg.get("flash_attention"):
            try:
                self.pipe.transformer.enable_flash_attn()
            except Exception:
                pass

        if self.cfg.get("compile"):
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune",
                backend="inductor",
            )
        logger.info("HunyuanVideo loaded.")

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
        negative_prompt: str = "blurry, low quality, watermark, text",
        duration: float = 5.0,
        output_path: str | Path | None = None,
        fps: int | None = None,
        width: int = 1280,
        height: int = 720,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 24)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 129))

        # Use a CPU generator for deterministic, device-independent seeds.
        # A CUDA generator can produce different random streams across devices;
        # the CPU generator yields identical sequences regardless of GPU model.
        gen = (
            torch.Generator(device="cpu").manual_seed(seed)
            if seed is not None
            else None
        )

        output = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=gen,
        )

        frames = output.frames[0]
        out_path = Path(output_path) if output_path else Path(f"/tmp/hunyuan_{seed or 0}.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=_fps)
        logger.info(f"HunyuanVideo → {out_path}")
        return out_path
