"""
GPU Memory Manager — centralized VRAM lifecycle management.
Tracks loaded models, evicts by priority, verifies free VRAM.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manage GPU model lifecycle with priority-based eviction."""

    def __init__(self, total_vram_gb: float = 80.0, safety_margin_gb: float = 2.0):
        self.total = total_vram_gb
        self.safety = safety_margin_gb
        self._loaded: dict[str, tuple[Any, float, int]] = {}  # name → (model, vram_gb, priority)

    @staticmethod
    def free_vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / 1e9

    def load_model(
        self,
        name: str,
        loader_fn: Callable[[], Any],
        vram_gb: float,
        priority: int = 99,
    ) -> Any:
        """Load a model, evicting lower-priority models if needed."""
        if name in self._loaded:
            return self._loaded[name][0]

        self.ensure_available(vram_gb + self.safety)
        model = loader_fn()
        self._loaded[name] = (model, vram_gb, priority)
        logger.info(
            f"Loaded {name} ({vram_gb:.1f} GB). "
            f"Free VRAM: {self.free_vram_gb():.1f} GB"
        )
        return model

    def unload(self, name: str) -> None:
        """Unload a specific model and verify VRAM freed."""
        if name not in self._loaded:
            return
        model, vram_gb, _ = self._loaded.pop(name)
        if hasattr(model, "unload"):
            model.unload()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.5)
        logger.info(
            f"Unloaded {name} (~{vram_gb:.1f} GB). "
            f"Free VRAM: {self.free_vram_gb():.1f} GB"
        )

    def unload_all(self) -> None:
        for name in list(self._loaded):
            self.unload(name)

    def ensure_available(self, required_gb: float) -> None:
        """Evict lowest-priority models until required VRAM is free."""
        while self.free_vram_gb() < required_gb:
            victim = self._lowest_priority_loaded()
            if victim is None:
                raise RuntimeError(
                    f"Cannot free {required_gb:.1f} GB VRAM. "
                    f"Currently free: {self.free_vram_gb():.1f} GB"
                )
            logger.info(f"Evicting {victim} to free VRAM")
            self.unload(victim)

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def loaded_models(self) -> list[str]:
        return list(self._loaded.keys())

    def _lowest_priority_loaded(self) -> str | None:
        """Return name of loaded model with highest priority number (lowest priority)."""
        if not self._loaded:
            return None
        return max(self._loaded, key=lambda n: self._loaded[n][2])
