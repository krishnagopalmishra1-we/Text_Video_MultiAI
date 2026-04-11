"""
TTS engine — Kokoro only (local, 82M params, free, ★★★★☆ quality).
A100 runs Kokoro at real-time x50+ speed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from scene_splitter import SceneData

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(
        self,
        voice: str = "af_heart",   # af_heart=female warm | am_adam=male
        speed: float = 1.0,
        sample_rate: int = 24000,
    ):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self._pipeline = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._load()

        generator = self._pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
            split_pattern=r"\n+",
        )
        chunks: list[np.ndarray] = [audio for _, _, audio in generator]

        if not chunks:
            sf.write(str(out), np.zeros(self.sample_rate), self.sample_rate)
            return out

        sf.write(str(out), np.concatenate(chunks), self.sample_rate)
        return out

    def synthesize_full(
        self, scenes: list[SceneData], output_path: str | Path
    ) -> Path:
        """
        Concatenate all scene narrations with silence padding to match
        exact scene durations. Returns single WAV aligned to video.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        segments: list[np.ndarray] = []

        for scene in scenes:
            text = scene.narration or ""
            tmp = out.parent / f"_tmp_{scene.scene_id}.wav"
            try:
                if text:
                    self.synthesize(text, tmp)
                    audio, sr = sf.read(str(tmp))
                    if sr != self.sample_rate:
                        # Resample without librosa — use scipy.signal if available,
                        # otherwise simple numpy interpolation
                        target_len = int(len(audio) * self.sample_rate / sr)
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, target_len),
                            np.arange(len(audio)),
                            audio,
                        )
                else:
                    audio = np.zeros(0)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)

            target = int(scene.duration * self.sample_rate)
            if len(audio) < target:
                audio = np.pad(audio, (0, target - len(audio)))
            else:
                audio = audio[:target]
            segments.append(audio)

        full = np.concatenate(segments)
        sf.write(str(out), full, self.sample_rate)
        logger.info(f"TTS: {out} ({len(full)/self.sample_rate:.1f}s)")
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "pip install kokoro>=0.9.4 soundfile\n"
                "Linux also needs: apt-get install espeak-ng"
            )
        self._pipeline = KPipeline(lang_code="a")
        logger.info("Kokoro TTS loaded.")
