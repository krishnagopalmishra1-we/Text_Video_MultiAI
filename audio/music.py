"""
Music engine — MusicGen-large (local) for background score generation.
facebook/musicgen-large fits on A100 with ~16GB VRAM.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

_MODEL_ID = "facebook/musicgen-large"
_SAMPLE_RATE = 32000


class MusicEngine:
    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("pip install transformers")

        logger.info(f"Loading {self.model_id}…")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = MusicgenForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        logger.info("MusicGen loaded.")

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model, self._processor
        self._model = None
        self._processor = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        duration: float,
        output_path: str | Path,
        seed: int | None = 42,
    ) -> Path:
        """
        Generate music matching prompt and duration.
        Returns WAV file path.
        Falls back to silence if MusicGen cannot load on current runtime.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.load()

            # MusicGen generates in 30s chunks max; tile for longer durations
            target_tokens = int(duration * 50)  # ~50 tokens/sec at 32kHz

            inputs = self._processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            if seed is not None:
                gen = torch.Generator(device=self.device).manual_seed(seed)
            else:
                gen = None

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=min(target_tokens, 1503),  # model max ~30s
                    do_sample=True,
                    guidance_scale=3.0,
                )

            audio = audio_values[0, 0].cpu().float().numpy()
            sr = self._model.config.audio_encoder.sampling_rate

            # Loop/tile audio if duration > 30s
            if duration > 30:
                audio = self._tile_audio(audio, duration, sr)

            sf.write(str(out), audio, sr)
            logger.info(f"Music: {out} ({len(audio)/sr:.1f}s)")
            return out
        except Exception as e:
            # Keep pipeline alive when model loading is blocked by upstream/runtime constraints.
            logger.warning(f"Music generation failed, using silence fallback: {e}")
            sr = _SAMPLE_RATE
            silence = np.zeros(max(1, int(duration * sr)), dtype=np.float32)
            sf.write(str(out), silence, sr)
            return out

    def generate_for_video(
        self,
        style: str,
        total_duration: float,
        output_path: str | Path,
    ) -> Path:
        """Convenience: build a style-aware prompt and generate."""
        prompt_map = {
            "cinematic": "epic cinematic orchestral score, dramatic strings, Hollywood blockbuster",
            "documentary": "calm ambient documentary underscore, gentle piano, nature sounds",
            "commercial": "upbeat corporate background music, modern, energetic",
            "sci_fi": "futuristic electronic ambient, synthesizers, space atmosphere",
            "nature": "peaceful nature ambient, birds, gentle wind, acoustic guitar",
            "dramatic": "intense dramatic orchestral, percussion, building tension",
        }
        prompt = prompt_map.get(style, "cinematic background music, high quality")
        return self.generate(prompt, total_duration, output_path)

    @staticmethod
    def _tile_audio(audio: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
        """Seamlessly loop audio to reach target duration with crossfade."""
        target_samples = int(target_duration * sr)
        fade_len = min(sr * 2, len(audio) // 4)  # 2s crossfade

        result = np.copy(audio)
        while len(result) < target_samples:
            fade_out = np.linspace(1, 0, fade_len)
            fade_in = np.linspace(0, 1, fade_len)
            result[-fade_len:] = result[-fade_len:] * fade_out + audio[:fade_len] * fade_in
            result = np.concatenate([result, audio[fade_len:]])

        return result[:target_samples]
