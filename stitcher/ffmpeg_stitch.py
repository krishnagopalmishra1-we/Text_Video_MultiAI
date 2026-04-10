"""
FFmpeg stitcher — concatenates scene clips, applies transitions,
mixes audio, and outputs final video at 1080p or 4K.
Uses 48 CPU threads and hardware-accelerated encoding where available.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import yaml

from .transitions import build_xfade_chain

logger = logging.getLogger(__name__)

ResolutionMode = Literal["1080p", "4k", "720p"]

_RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def _ffprobe_duration(path: Path) -> float:
    """Get video duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream.get("duration", 0))
    return 0.0


def _scale_filter(w: int, h: int) -> str:
    return f"scale={w}:{h}:flags=lanczos,setsar=1"


def _nvenc_available() -> bool:
    """Check if NVENC encoder exists and can actually initialize."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        if "h264_nvenc" not in result.stdout:
            return False

        # Some environments expose h264_nvenc in ffmpeg, but runtime init still fails.
        probe = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-y",
                "-f", "lavfi", "-i", "testsrc=size=128x72:rate=1",
                "-frames:v", "1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return probe.returncode == 0
    except Exception:
        return False


class Stitcher:
    def __init__(
        self,
        config_path: str | Path = "config/model_config.yaml",
        presets_path: str | Path = "config/presets.yaml",
        cpu_threads: int = 48,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

        self.gen_cfg = cfg["generation"]
        self.cpu_threads = cpu_threads
        self.crf = self.gen_cfg.get("crf", 18)
        self.preset = self.gen_cfg.get("preset", "slow")

        # Use NVENC if available (5-10x faster encoding, same quality)
        if _nvenc_available():
            self.codec = "h264_nvenc"
            self.preset = "p7"  # nvenc highest quality preset
            self.crf_flag = "-cq"  # nvenc uses -cq instead of -crf
            logger.info("Using NVENC hardware encoder (h264_nvenc).")
        else:
            self.codec = self.gen_cfg.get("codec", "libx264")
            self.crf_flag = "-crf"
            logger.info(f"Using software encoder ({self.codec}).")

    def _fallback_to_software_encoder(self) -> None:
        """Switch to a safe software codec if hardware encode fails at runtime."""
        if self.codec == "h264_nvenc":
            self.codec = "libx264"
            self.crf_flag = "-crf"
            self.preset = self.gen_cfg.get("preset", "slow")
            logger.warning("NVENC runtime init failed; falling back to libx264.")

    def _run_encode_with_fallback(self, cmd: list[str]) -> None:
        """Run ffmpeg encode and retry once with libx264 if NVENC fails."""
        try:
            self._run(cmd)
            return
        except RuntimeError as e:
            msg = str(e)
            nvenc_runtime_fail = (
                self.codec == "h264_nvenc"
                and (
                    "No capable devices found" in msg
                    or "OpenEncodeSessionEx failed" in msg
                    or "Error while opening encoder" in msg
                )
            )
            if not nvenc_runtime_fail:
                raise

            self._fallback_to_software_encoder()
            retry_cmd = ["libx264" if t == "h264_nvenc" else t for t in cmd]
            normalized: list[str] = []
            i = 0
            while i < len(retry_cmd):
                token = retry_cmd[i]
                if token == "-cq":
                    normalized.extend(["-crf", str(self.crf)])
                    i += 2
                    continue
                if token == "-preset" and i + 1 < len(retry_cmd) and retry_cmd[i + 1] == "p7":
                    normalized.extend(["-preset", self.preset])
                    i += 2
                    continue
                normalized.append(token)
                i += 1
            self._run(normalized)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def stitch(
        self,
        clip_paths: list[Path],
        output_path: str | Path,
        audio_path: str | Path | None = None,
        transition: str = "crossfade",
        transition_duration: float = 0.8,
        resolution: ResolutionMode = "1080p",
        target_fps: int = 24,
        upscale_to_4k: bool = False,
    ) -> Path:
        """
        Full stitch pipeline:
        1. Normalize all clips (resolution, fps, codec)
        2. Apply transitions via xfade filter_complex
        3. Attach audio track (narration + music mix)
        4. Encode final output
        5. Optional: upscale to 4K via Real-ESRGAN
        """
        if not clip_paths:
            raise ValueError("No clips to stitch")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        w, h = _RESOLUTIONS.get(resolution, _RESOLUTIONS["1080p"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Step 1: Normalize clips
            logger.info(f"Normalizing {len(clip_paths)} clips → {w}x{h} @ {target_fps}fps")
            normalized: list[Path] = []
            for i, clip in enumerate(clip_paths):
                norm = tmp / f"norm_{i:04d}.mp4"
                self._normalize_clip(clip, norm, w, h, target_fps)
                normalized.append(norm)

            # Step 2: Build with transitions
            logger.info(f"Stitching with transition: {transition}")
            stitched = tmp / "stitched.mp4"
            self._concat_with_transitions(
                normalized, stitched, transition, transition_duration, w, h, target_fps
            )

            # Step 3: Attach audio
            if audio_path and Path(audio_path).exists():
                logger.info(f"Attaching audio: {audio_path}")
                final_tmp = tmp / "final_with_audio.mp4"
                self._attach_audio(stitched, Path(audio_path), final_tmp)
            else:
                final_tmp = stitched

            # Step 4: Copy to output
            shutil.copy2(final_tmp, out)

        # Step 5: Optional upscale
        if upscale_to_4k and resolution != "4k":
            logger.info("Upscaling to 4K via Real-ESRGAN…")
            out = self._upscale_4k(out)

        logger.info(f"Final video: {out}")
        return out

    def stitch_concat_only(
        self,
        clip_paths: list[Path],
        output_path: str | Path,
        resolution: ResolutionMode = "1080p",
        target_fps: int = 24,
    ) -> Path:
        """Fast cut-only concat for previews."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = _RESOLUTIONS[resolution]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            concat_list = tmp / "concat.txt"
            norm_clips: list[Path] = []
            for i, c in enumerate(clip_paths):
                n = tmp / f"n{i:04d}.mp4"
                self._normalize_clip(c, n, w, h, target_fps)
                norm_clips.append(n)
            concat_list.write_text(
                "\n".join(f"file '{p}'" for p in norm_clips)
            )
            self._run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(out),
            ])
        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _normalize_clip(self, src: Path, dst: Path, w: int, h: int, fps: int) -> None:
        self._run_encode_with_fallback([
            "ffmpeg", "-y",
            "-i", str(src),
            "-vf", f"{_scale_filter(w, h)},fps={fps}",
            "-c:v", self.codec,
            self.crf_flag, str(self.crf),
            "-preset", self.preset,
            "-an",
            "-threads", str(self.cpu_threads),
            "-movflags", "+faststart",
            str(dst),
        ])

    def _concat_with_transitions(
        self,
        clips: list[Path],
        dst: Path,
        transition: str,
        td: float,
        w: int,
        h: int,
        fps: int,
    ) -> None:
        if transition == "cut" or len(clips) == 1:
            # Simple concat demuxer — fastest path
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("\n".join(f"file '{c}'" for c in clips))
                list_path = f.name
            self._run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                str(dst),
            ])
            return

        # xfade requires individual -i inputs
        durations = [_ffprobe_duration(c) for c in clips]
        filter_complex = build_xfade_chain(durations, transition, td)

        cmd = ["ffmpeg", "-y"]
        for clip in clips:
            cmd += ["-i", str(clip)]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-c:v", self.codec,
            self.crf_flag, str(self.crf),
            "-preset", self.preset,
            "-threads", str(self.cpu_threads),
            "-movflags", "+faststart",
            str(dst),
        ]
        self._run_encode_with_fallback(cmd)

    def _attach_audio(self, video: Path, audio: Path, dst: Path) -> None:
        self._run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(dst),
        ])

    def _upscale_4k(self, video: Path) -> Path:
        """
        Frame-by-frame upscale via Real-ESRGAN, then re-encode.
        Requires: realesrgan-ncnn-vulkan or basicsr.
        """
        try:
            import subprocess
            frames_dir = video.parent / "frames_hr"
            frames_dir.mkdir(exist_ok=True)

            # Extract frames
            self._run([
                "ffmpeg", "-y", "-i", str(video),
                "-threads", str(self.cpu_threads),
                str(frames_dir / "frame_%06d.png"),
            ])

            # Upscale frames
            self._run([
                "realesrgan-ncnn-vulkan",
                "-i", str(frames_dir),
                "-o", str(frames_dir),
                "-n", "realesrgan-x4plus",
                "-s", "4",
            ])

            out_4k = video.parent / (video.stem + "_4k.mp4")
            self._run([
                "ffmpeg", "-y",
                "-framerate", "24",
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx265",
                "-crf", "20",
                "-preset", self.preset,
                "-threads", str(self.cpu_threads),
                "-tag:v", "hvc1",
                "-movflags", "+faststart",
                str(out_4k),
            ])
            return out_4k
        except Exception as e:
            logger.warning(f"4K upscale failed: {e}. Returning 1080p.")
            return video

    @staticmethod
    def _run(cmd: list[str]) -> None:
        logger.debug("FFmpeg: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {result.returncode}):\n{result.stderr[-2000:]}"
            )
