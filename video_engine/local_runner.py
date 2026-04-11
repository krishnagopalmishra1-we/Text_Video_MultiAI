"""
Local GPU runner — manages model lifecycle on the A100 80GB.
Only one large model loaded at a time. Smaller models can coexist.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import yaml

from scene_splitter import SceneData
from .models import Wan2Runner, HunyuanRunner, CogVideoXRunner, LTXRunner, _RUNNER_CLASSES

logger = logging.getLogger(__name__)

# VRAM budget thresholds (in GB free) required before loading
_VRAM_THRESHOLDS = {
    "wan2_14b": 42,
    "wan2_1b": 8,
    "hunyuan": 37,
    "cogvideox": 26,
    "ltx": 12,
}

_RUNNERS = _RUNNER_CLASSES


def _free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1e9


class LocalRunner:
    def __init__(self, config_path: str | Path = "config/model_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.model_cfgs: dict = cfg["models"]["local"]
        self.quality_presets: dict = cfg.get("quality_presets", {})
        self.strategies: dict = cfg.get("strategies", {})
        self._runners: dict[str, Wan2Runner | HunyuanRunner | CogVideoXRunner | LTXRunner] = {}
        self._active_model: str | None = None

        # Build runner instances for enabled models
        for name, mcfg in self.model_cfgs.items():
            if mcfg.get("enabled", False) and name in _RUNNERS:
                self._runners[name] = _RUNNERS[name](mcfg)

        # Sorted by priority (ascending = higher priority first)
        self._priority_order: list[str] = sorted(
            self._runners.keys(),
            key=lambda n: self.model_cfgs[n].get("priority", 99),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        scene: SceneData,
        output_dir: str | Path = "outputs/scenes",
        preferred_model: str | None = None,
        quality: str = "high",    # ultra | high | balanced | fast | preview
        seed: int | None = None,
        strategy: str | None = None,
        is_hero: bool = False,
    ) -> Path:
        """Generate a single scene clip. Returns path to output MP4."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"scene_{scene.scene_id:04d}.mp4"

        strat = self.strategies.get(strategy) if strategy else None

        if strat and not preferred_model:
            # Strategy drives model selection
            model_name = strat["hero_model"] if is_hero else strat["bulk_model"]
        else:
            model_name = self._select_model(preferred_model, quality)

        self._switch_model(model_name)
        runner = self._runners[model_name]
        mcfg = self.model_cfgs[model_name]

        # Resolution & steps: strategy overrides > quality preset > model defaults
        if strat:
            if is_hero:
                width, height = strat.get("gen_resolution", mcfg["resolution"])
                steps = strat.get("hero_steps", mcfg.get("default_steps", 30))
            else:
                width, height = strat.get("gen_resolution", mcfg["resolution"])
                steps = strat.get("steps", mcfg.get("default_steps", 20))
        else:
            preset = self.quality_presets.get(quality, {}).get(model_name, {})
            width, height = preset.get("resolution", mcfg["resolution"])
            steps = preset.get("steps", mcfg.get("default_steps", 20))

        logger.info(
            f"Scene {scene.scene_id} → {model_name} "
            f"({width}x{height}, {steps} steps, strategy={strategy or 'none'}, hero={is_hero})"
        )
        return runner.generate(
            prompt=scene.video_prompt or scene.text,
            duration=scene.duration,
            output_path=out_path,
            fps=mcfg.get("fps"),
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed if seed is not None else scene.scene_id,
        )

    def unload_all(self) -> None:
        for r in self._runners.values():
            r.unload()
        self._active_model = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_model(self, preferred: str | None, quality: str) -> str:
        """Choose model based on preference, quality tier, and available VRAM."""
        free = _free_vram_gb()
        logger.debug(f"Free VRAM: {free:.1f} GB")

        # Quality → preferred model hint (only if user didn't specify)
        if preferred is None:
            if quality == "preview":
                preferred = "ltx"
            elif quality == "fast":
                preferred = "cogvideox"
            # ultra, high, balanced: use priority order (wan2_14b > wan2_1b > ...)

        candidates = (
            [preferred] + self._priority_order
            if preferred and preferred in self._runners
            else self._priority_order
        )

        for name in candidates:
            required = _VRAM_THRESHOLDS.get(name, 20)
            if free >= required:
                return name

        # Force-unload active model and retry ltx (smallest)
        if self._active_model:
            self._runners[self._active_model].unload()
            self._active_model = None
            time.sleep(1)
            if "ltx" in self._runners:
                return "ltx"

        raise RuntimeError(
            f"No local model fits in available VRAM ({free:.1f} GB free). "
            "Enable an API fallback model."
        )

    def _switch_model(self, name: str) -> None:
        """Unload current model if different from requested."""
        if self._active_model == name:
            return
        # Unload current model only if new model won't fit alongside it
        if self._active_model:
            req_new = _VRAM_THRESHOLDS.get(name, 20)
            if _free_vram_gb() < req_new:
                logger.info(f"Unloading {self._active_model} to load {name}")
                self._runners[self._active_model].unload()
                self._active_model = None
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.5)
                freed = _free_vram_gb()
                logger.info(f"Post-unload free VRAM: {freed:.1f} GB")
                if freed < _VRAM_THRESHOLDS.get(name, 20):
                    logger.warning(
                        f"VRAM still insufficient after unload: {freed:.1f} GB free, "
                        f"{_VRAM_THRESHOLDS.get(name, 20)} GB needed for {name}"
                    )
        self._active_model = name
