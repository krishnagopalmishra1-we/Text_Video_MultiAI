"""
API runner — Runway Gen-3 Alpha Turbo only.
Used for: (1) local model failure fallback, (2) hero scenes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from scene_splitter import SceneData

logger = logging.getLogger(__name__)


class APIRunner:
    async def generate(
        self,
        scene: SceneData,
        output_dir: str | Path = "outputs/scenes",
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        from api.runway import RunwayClient
        client = RunwayClient()
        out_path = out_dir / f"scene_{scene.scene_id:04d}.mp4"
        logger.info(f"Scene {scene.scene_id} → Runway Gen-3")
        return await client.generate(
            prompt=scene.video_prompt or scene.text,
            duration=min(int(scene.duration), 10),
            output_path=out_path,
        )
