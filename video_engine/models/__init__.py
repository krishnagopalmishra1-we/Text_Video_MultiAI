from .wan2 import Wan2Runner
from .hunyuan import HunyuanRunner
from .cogvideox import CogVideoXRunner
from .ltx import LTXRunner

# Both wan2_14b and wan2_1b use Wan2Runner — same class, different config.
_RUNNER_CLASSES = {
    "wan2_14b": Wan2Runner,
    "wan2_1b": Wan2Runner,
    "hunyuan": HunyuanRunner,
    "cogvideox": CogVideoXRunner,
    "ltx": LTXRunner,
}

__all__ = ["Wan2Runner", "HunyuanRunner", "CogVideoXRunner", "LTXRunner", "_RUNNER_CLASSES"]
