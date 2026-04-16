"""
HunyuanVideo wrapper — cinema quality on A100 80GB.
BF16 (default): ~60GB VRAM — load alone after unloading WAN.
INT8 optional: ~37GB — set quantization: "int8" in config if coexistence needed.

LoRA support:
  Configure in model_config.yaml under models.local.hunyuan.lora_weights:
    - path: "username/my-lora"
      weight_name: "hunyuan_lora.safetensors"
      scale: 0.9
      name: "cinematic"
  LoRAs are fused at load time — no per-inference overhead.
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
        quant = self.cfg.get("quantization")  # "int8", "int4", or None (BF16)

        if quant == "int8":
            logger.info("Loading HunyuanVideo INT8-quantized (~37GB) — BF16 preferred if VRAM allows")
            from diffusers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                model_id,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
        elif quant == "int4":
            logger.info("Loading HunyuanVideo NF4-quantized (~20GB VRAM)")
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
            # Default: BF16 — full precision, best quality, ~60GB on A100 80GB
            logger.info("Loading HunyuanVideo BF16 (~60GB) — full quality, no quantization")
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
            # Optional: keep text encoder on CPU to save ~4GB VRAM.
            # Must come after pipe.to(self.device).
            self.pipe.text_encoder.to("cpu")

        if self.cfg.get("flash_attention"):
            try:
                self.pipe.transformer.enable_flash_attn()
            except Exception:
                pass

        # ── LoRA weights — fuse before compile ───────────────────────
        # Note: quantized (INT8/INT4) models do not support LoRA fusing —
        # skip silently if quantization is active.
        if not quant:
            self._apply_loras()
        else:
            logger.info("LoRA fusing skipped — not supported with quantized models.")

        if self.cfg.get("compile"):
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="max-autotune",
                backend="inductor",
            )
        logger.info("HunyuanVideo loaded.")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _apply_loras(self) -> None:
        """Load, fuse, and unload LoRA weights from config.
        Fusing bakes LoRA deltas into base weights — zero per-inference overhead.
        Only supported on non-quantized (BF16) models.
        """
        loras = self.cfg.get("lora_weights", [])
        if not loras:
            return

        adapter_names: list[str] = []
        adapter_scales: list[float] = []

        for i, lc in enumerate(loras):
            path = lc.get("path") or lc.get("hf_id")
            if not path:
                logger.warning(f"Hunyuan LoRA entry {i} missing 'path'/'hf_id' — skipping")
                continue
            scale = float(lc.get("scale", 0.9))
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
            logger.info(f"Fused {len(adapter_names)} LoRA adapter(s) into HunyuanVideo weights.")
        except Exception as e:
            logger.warning(f"LoRA fuse failed — running without LoRA: {e}")
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

    # ------------------------------------------------------------------

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
        guidance_scale: float = 7.0,   # raised from 6.0 — better cinematic adherence
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> Path:
        from diffusers.utils import export_to_video

        self.load()
        _fps = fps or self.cfg.get("fps", 24)
        num_frames = min(int(duration * _fps) + 1, self.cfg.get("max_frames", 129))

        # Use a CPU generator for deterministic, device-independent seeds.
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
