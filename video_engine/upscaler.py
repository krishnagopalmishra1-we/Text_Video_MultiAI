"""
Real-ESRGAN GPU upscaler — upscales clips from gen resolution to target.
Used by balanced/quality strategies: 848×480 → 1280×720.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Upscaler:
    def __init__(
        self,
        scale: int = 2,
        model: str = "realesrgan-x4plus",
        tile_size: int = 512,
    ):
        self.scale = scale
        self.model = model
        self.tile_size = tile_size

    def upscale_clip(
        self,
        input_path: Path,
        output_path: Path | None = None,
        target_width: int = 1280,
        target_height: int = 720,
    ) -> Path:
        """
        Upscale a video clip frame-by-frame using realesrgan-ncnn-vulkan
        or ffmpeg scale filter as fallback.
        """
        out = output_path or input_path.with_stem(input_path.stem + "_upscaled")
        out.parent.mkdir(parents=True, exist_ok=True)

        # Try GPU-accelerated realesrgan-ncnn-vulkan first
        if self._has_realesrgan():
            return self._upscale_realesrgan(input_path, out)

        # Fallback: ffmpeg lanczos scale
        logger.warning("realesrgan-ncnn-vulkan not found, using ffmpeg lanczos upscale")
        return self._upscale_ffmpeg(input_path, out, target_width, target_height)

    def upscale_batch(
        self,
        clips: list[Path],
        output_dir: Path,
        target_width: int = 1280,
        target_height: int = 720,
    ) -> list[Path]:
        """Upscale all clips in a list. Returns list of upscaled paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for clip in clips:
            out = output_dir / clip.name
            results.append(
                self.upscale_clip(clip, out, target_width, target_height)
            )
        return results

    def _has_realesrgan(self) -> bool:
        try:
            subprocess.run(
                ["realesrgan-ncnn-vulkan", "-h"],
                capture_output=True, timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _upscale_realesrgan(self, input_path: Path, output_path: Path) -> Path:
        """Use realesrgan-ncnn-vulkan binary for GPU-accelerated upscaling."""
        cmd = [
            "realesrgan-ncnn-vulkan",
            "-i", str(input_path),
            "-o", str(output_path),
            "-s", str(self.scale),
            "-n", self.model,
            "-t", str(self.tile_size),
        ]
        logger.info(f"Upscaling {input_path.name} via Real-ESRGAN (scale={self.scale})")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")
        return output_path

    def _upscale_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        width: int,
        height: int,
    ) -> Path:
        """Fallback: ffmpeg lanczos scaling."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ]
        logger.info(f"Upscaling {input_path.name} via ffmpeg lanczos → {width}×{height}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg upscale failed: {result.stderr}")
        return output_path
