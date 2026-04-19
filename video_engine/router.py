"""
Video Router — routing logic for scene generation.

Strategy:
  Normal scenes:  Local model (priority order) → Runway fallback on failure
  Hero scenes:    Runway Gen-3 directly (best quality)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

from scene_splitter import SceneData
from .local_runner import LocalRunner
from .api_runner import APIRunner
from api.runway import RunwayNoCreditsError

logger = logging.getLogger(__name__)

QualityMode = Literal["ultra", "high", "balanced", "fast", "preview"]


class VideoRouter:
    def __init__(
        self,
        config_path: str | Path = "config/model_config.yaml",
        output_dir: str | Path = "outputs/scenes",
        quality: QualityMode = "high",
        local_retries: int = 2,
        api_fallback: bool = True,
        strategy: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.quality = quality
        self.local_retries = local_retries
        self.api_fallback = api_fallback
        self.strategy = strategy
        self.local = LocalRunner(config_path)
        self.api = APIRunner()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_scene(
        self,
        scene: SceneData,
        preferred_model: str | None = None,
        seed: int | None = None,
        hero: bool = False,
        is_hero: bool = False,
        style: str | None = None,
    ) -> Path:
        """
        Generate one scene clip.
        hero=True  → Runway Gen-3 directly, skip local.
        hero=False → local first, Runway only on failure.
        """
        if hero:
            logger.info(f"Scene {scene.scene_id}: hero → Runway Gen-3")
            return asyncio.run(self.api.generate(scene, self.output_dir))

        local_errors: list[str] = []
        for attempt in range(1, self.local_retries + 1):
            try:
                return self.local.generate(
                    scene=scene,
                    output_dir=self.output_dir,
                    preferred_model=preferred_model,
                    quality=self.quality,
                    seed=seed,
                    strategy=self.strategy,
                    is_hero=is_hero,
                    style=style,
                )
            except Exception as e:
                logger.exception(f"Local attempt {attempt}/{self.local_retries} failed")
                local_errors.append(str(e))

        if not self.api_fallback:
            raise RuntimeError(
                f"Local generation failed and API fallback is disabled. Errors: {' | '.join(local_errors)}"
            )

        logger.info(f"Scene {scene.scene_id}: local failed → Runway fallback")
        try:
            return asyncio.run(self.api.generate(scene, self.output_dir))
        except RunwayNoCreditsError:
            raise  # surface as fatal — no point retrying other scenes

    async def generate_scene_async(
        self,
        scene: SceneData,
        preferred_model: str | None = None,
        seed: int | None = None,
        hero: bool = False,
        is_hero: bool = False,
        style: str | None = None,
    ) -> Path:
        loop = asyncio.get_event_loop()

        if hero:
            return await self.api.generate(scene, self.output_dir)

        local_errors: list[str] = []
        for attempt in range(1, self.local_retries + 1):
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: self.local.generate(
                        scene=scene,
                        output_dir=self.output_dir,
                        preferred_model=preferred_model,
                        quality=self.quality,
                        seed=seed,
                        strategy=self.strategy,
                        is_hero=is_hero,
                        style=style,
                    ),
                )
            except Exception as e:
                logger.exception(f"Local async attempt {attempt} failed")
                local_errors.append(str(e))

        if not self.api_fallback:
            raise RuntimeError(
                f"Local generation failed and API fallback is disabled. Errors: {' | '.join(local_errors)}"
            )

        try:
            return await self.api.generate(scene, self.output_dir)
        except RunwayNoCreditsError:
            raise  # surface as fatal — no point retrying other scenes

    def generate_all(
        self,
        scenes: list[SceneData],
        hero_scene_ids: list[int] | None = None,
        preferred_model: str | None = None,
        resume: bool = True,
    ) -> list[Path]:
        """
        Generate all scenes.
        hero_scene_ids: scene IDs to send to Runway (e.g. [1, 5, 12]).
        resume: skip scenes whose clip file already exists.
        """
        hero_ids = set(hero_scene_ids or [])
        results: list[Path] = []

        for scene in scenes:
            out_path = self.output_dir / f"scene_{scene.scene_id:04d}.mp4"
            if resume and out_path.exists():
                logger.info(f"Scene {scene.scene_id}: skip (exists)")
                results.append(out_path)
                continue
            path = self.generate_scene(
                scene,
                preferred_model=preferred_model,
                seed=scene.scene_id,
                hero=scene.scene_id in hero_ids,
            )
            results.append(path)

        return results

    def cleanup(self) -> None:
        self.local.unload_all()
