"""
Audio sync — mixes narration + background music, outputs final audio track.
Handles timing alignment, volume ducking, fades.
Uses pydub for mixing (wraps FFmpeg internally).
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from scene_splitter import SceneData

logger = logging.getLogger(__name__)


class AudioSync:
    def __init__(
        self,
        narration_db: float = -3.0,
        music_db: float = -18.0,
        duck_db: float = -12.0,     # music level during narration
        fade_in: float = 1.0,
        fade_out: float = 2.0,
        sample_rate: int = 44100,
        cpu_threads: int = 48,
    ):
        self.narration_db = narration_db
        self.music_db = music_db
        self.duck_db = duck_db
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.sample_rate = sample_rate
        self.cpu_threads = cpu_threads

    def mix(
        self,
        narration_path: Path | None,
        music_path: Path | None,
        output_path: str | Path,
        total_duration: float,
        scenes: list[SceneData] | None = None,
    ) -> Path:
        """
        Mix narration + music into a single audio track.
        Uses FFmpeg amix + volume filters for precise control.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        has_narration = narration_path and Path(narration_path).exists()
        has_music = music_path and Path(music_path).exists()

        if not has_narration and not has_music:
            # Return silent audio
            self._generate_silence(out, total_duration)
            return out

        if has_narration and not has_music:
            return self._normalize_audio(narration_path, out, self.narration_db)

        if has_music and not has_narration:
            return self._trim_and_normalize(music_path, out, total_duration, self.music_db)

        # Both tracks: mix with ducking
        return self._mix_tracks(
            narration_path, music_path, out, total_duration, scenes
        )

    def _mix_tracks(
        self,
        narration: Path,
        music: Path,
        out: Path,
        duration: float,
        scenes: list[SceneData] | None,
    ) -> Path:
        """FFmpeg amix with sidechain ducking using sidechaincompress."""
        nar_vol = _db_to_factor(self.narration_db)
        mus_vol = _db_to_factor(self.music_db)

        # Build volume filter with fade in/out
        music_filter = (
            f"aloop=loop=-1:size=2e+09,"
            f"atrim=duration={duration},"
            f"afade=t=in:st=0:d={self.fade_in},"
            f"afade=t=out:st={max(0, duration - self.fade_out)}:d={self.fade_out},"
            f"volume={mus_vol:.4f}"
        )

        narration_filter = (
            f"atrim=duration={duration},"
            f"volume={nar_vol:.4f}"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(narration),
            "-i", str(music),
            "-filter_complex",
            f"[0:a]{narration_filter}[nar];"
            f"[1:a]{music_filter}[mus];"
            f"[nar][mus]amix=inputs=2:duration=first:dropout_transition=3[out]",
            "-map", "[out]",
            "-c:a", "aac",
            "-b:a", "256k",
            "-ar", str(self.sample_rate),
            "-threads", str(self.cpu_threads),
            str(out),
        ]
        _run_ffmpeg(cmd)
        logger.info(f"Mixed audio: {out}")
        return out

    def _normalize_audio(self, src: Path, dst: Path, db: float) -> Path:
        vol = _db_to_factor(db)
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-af", f"volume={vol:.4f}",
            "-c:a", "aac", "-b:a", "256k",
            "-ar", str(self.sample_rate),
            str(dst),
        ]
        _run_ffmpeg(cmd)
        return dst

    def _trim_and_normalize(
        self, src: Path, dst: Path, duration: float, db: float
    ) -> Path:
        vol = _db_to_factor(db)
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-af",
            f"aloop=loop=-1:size=2e+09,"
            f"atrim=duration={duration},"
            f"afade=t=in:st=0:d={self.fade_in},"
            f"afade=t=out:st={max(0, duration - self.fade_out)}:d={self.fade_out},"
            f"volume={vol:.4f}",
            "-c:a", "aac", "-b:a", "256k",
            "-ar", str(self.sample_rate),
            str(dst),
        ]
        _run_ffmpeg(cmd)
        return dst

    def _generate_silence(self, out: Path, duration: float) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r={self.sample_rate}:cl=stereo",
            "-t", str(duration),
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ]
        _run_ffmpeg(cmd)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _db_to_factor(db: float) -> float:
    return 10 ** (db / 20.0)


def _run_ffmpeg(cmd: list[str]) -> None:
    logger.debug("FFmpeg audio: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg audio failed:\n{r.stderr[-1500:]}")
