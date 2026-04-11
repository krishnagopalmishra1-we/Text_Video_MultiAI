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
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from api.runway import RunwayClient
            self._client = RunwayClient()
        return self._client

    async def generate(
        self,
        scene: SceneData,
        output_dir: str | Path = "outputs/scenes",
    ) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        client = self._get_client()
        out_path = out_dir / f"scene_{scene.scene_id:04d}.mp4"
        logger.info(f"Scene {scene.scene_id} → Runway Gen-3")
        return await client.generate(
            prompt=scene.video_prompt or scene.text,
            duration=min(int(scene.duration), 10),
            output_path=out_path,
        )
