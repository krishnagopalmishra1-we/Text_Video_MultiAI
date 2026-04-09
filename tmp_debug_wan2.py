from pathlib import Path
import traceback

from video_engine.models.wan2 import Wan2Runner

cfg = {
    "hf_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "fps": 16,
    "max_frames": 81,
    "flash_attention": False,
    "compile": False,
}

runner = Wan2Runner(cfg)
try:
    out = runner.generate(
        prompt="A cinematic shot of an astronaut on Mars at sunset.",
        duration=1.0,
        width=832,
        height=480,
        num_inference_steps=2,
        seed=123,
        output_path=Path("/tmp/wan2_debug.mp4"),
    )
    print(f"OK {out}")
except Exception:
    traceback.print_exc()
    raise
