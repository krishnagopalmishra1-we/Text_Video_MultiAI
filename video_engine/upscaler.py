"""
Real-ESRGAN GPU upscaler — upscales clips frame-by-frame using Python realesrgan package.
Fixes basicsr/torchvision compatibility via shim before import.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_WEIGHTS_URL = "/root/.cache/realesrgan/RealESRGAN_x4plus.pth"


def _torchvision_shim() -> None:
    import sys
    try:
        import torchvision.transforms.functional_tensor  # noqa
    except ImportError:
        try:
            import torchvision.transforms.functional as _f
            sys.modules["torchvision.transforms.functional_tensor"] = _f
        except Exception:
            pass


def _load_upsampler(tile_size: int):
    _torchvision_shim()
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=_WEIGHTS_URL,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        device=torch.device("cuda"),
        half=True,
    )


class Upscaler:
    def __init__(
        self,
        scale: int = 4,
        model: str = "realesrgan-x4plus",
        tile_size: int = 512,
        target_width: int = 1280,
        target_height: int = 720,
    ):
        self.scale = scale
        self.model = model
        self.tile_size = tile_size
        self.target_width = target_width
        self.target_height = target_height

    def upscale_clip(
        self,
        input_path: Path,
        output_path: Path | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> Path:
        out = output_path or input_path.with_stem(input_path.stem + "_upscaled")
        out.parent.mkdir(parents=True, exist_ok=True)
        tw = target_width or self.target_width
        th = target_height or self.target_height
        try:
            return self._upscale_python(input_path, out, tw, th)
        except Exception as e:
            logger.warning(f"Real-ESRGAN failed ({e}), falling back to lanczos")
            return self._upscale_ffmpeg(input_path, out, tw, th)

    def upscale_batch(
        self,
        clips: list[Path],
        output_dir: Path,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        tw = target_width or self.target_width
        th = target_height or self.target_height
        upsampler = None
        try:
            upsampler = _load_upsampler(self.tile_size)
        except Exception as e:
            logger.warning(f"Could not load Real-ESRGAN: {e}. Using ffmpeg fallback.")

        results = []
        for clip in clips:
            out = output_dir / clip.name
            try:
                if upsampler:
                    results.append(self._upscale_python_with(clip, out, tw, th, upsampler))
                else:
                    results.append(self._upscale_ffmpeg(clip, out, tw, th))
            except Exception as e:
                logger.warning(f"Upscale failed for {clip.name}: {e}, using ffmpeg fallback")
                results.append(self._upscale_ffmpeg(clip, out, tw, th))
        return results

    def _upscale_python(self, input_path: Path, output_path: Path, tw: int, th: int) -> Path:
        upsampler = _load_upsampler(self.tile_size)
        return self._upscale_python_with(input_path, output_path, tw, th, upsampler)

    def _upscale_python_with(self, input_path: Path, output_path: Path, tw: int, th: int, upsampler) -> Path:
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames extracted from {input_path}")

        logger.info(f"Real-ESRGAN: {input_path.name} ({len(frames)} frames) → {tw}×{th}")
        upscaled = []
        for frame in frames:
            out_frame, _ = upsampler.enhance(frame, outscale=4)
            if out_frame.shape[1] != tw or out_frame.shape[0] != th:
                out_frame = cv2.resize(out_frame, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            upscaled.append(out_frame)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (tw, th))
        for frame in upscaled:
            writer.write(frame)
        writer.release()

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_path),
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-c:a", "copy", str(output_path)],
            check=True, capture_output=True,
        )
        tmp_path.unlink(missing_ok=True)
        return output_path

    def _upscale_ffmpeg(self, input_path: Path, output_path: Path, width: int, height: int) -> Path:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg upscale failed: {result.stderr}")
        return output_path
