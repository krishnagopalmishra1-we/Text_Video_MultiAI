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
        prompt="A lone astronaut treks across a vast Martian valley at golden hour, footprints trailing in red dust.",
        duration=5.0,
        width=1280,
        height=720,
        num_inference_steps=50,
        seed=123,
        output_path=Path("/tmp/wan2_debug_prod50.mp4"),
    )
    print(f"OK {out}")
except Exception:
    traceback.print_exc()
    raise
