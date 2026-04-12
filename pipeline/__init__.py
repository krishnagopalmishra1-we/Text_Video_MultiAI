"""Pipeline module — DAG orchestration and GPU memory management."""
from .dag import submit_pipeline


def __getattr__(name: str):
    if name == "GPUMemoryManager":
        from .gpu_manager import GPUMemoryManager
        return GPUMemoryManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["submit_pipeline", "GPUMemoryManager"]
