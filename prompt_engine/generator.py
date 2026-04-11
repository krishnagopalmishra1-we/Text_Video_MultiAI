"""
Prompt engine: converts scene text → rich video generation prompts.
Uses Claude API for semantic enrichment, with local fallback.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Literal

import anthropic
import yaml

from scene_splitter import SceneData
from .presets import build_prompt, STYLE_PRESETS, CAMERA_PRESETS


SYSTEM_PROMPT = """You are a cinematic video prompt engineer.
Convert scene text into a single, dense video generation prompt.
Rules:
- Max 120 words
- Lead with the main visual subject and action
- Include lighting, atmosphere, camera motion
- No dialogue, no subtitles, no text on screen
- No first/second person
- Output ONLY the prompt, nothing else"""

USER_TEMPLATE = """Scene text:
{text}

Style: {style}
Camera: {camera}
Duration: {duration}s

Video prompt:"""


class PromptEngine:
    def __init__(
        self,
        presets_path: str | Path = "config/presets.yaml",
        style: str = "cinematic",
        camera: str = "dolly_forward",
        use_llm: bool = True,
        model: str = "claude-haiku-4-5-20251001",  # fast + cheap for bulk
    ):
        self.style = style
        self.camera = camera
        self.use_llm = use_llm
        self.model = model

        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

        self._client: anthropic.AsyncAnthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, scene: SceneData) -> str:
        """Generate a video prompt for a single scene."""
        style = scene.style_hint or self.style
        camera = scene.camera_hint or self.camera

        if self.use_llm:
            try:
                raw = await self._llm_prompt(scene.text, style, camera, scene.duration)
                return build_prompt(raw, style, camera, self.presets)
            except Exception as e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "LLM prompt generation failed for scene %s: %s", scene.scene_id, e
                )
                # fall through to template

        return build_prompt(scene.narration or scene.text, style, camera, self.presets)

    async def generate_batch(
        self, scenes: list[SceneData], concurrency: int = 4
    ) -> list[SceneData]:
        """Enrich all scenes with video_prompt in parallel."""
        sem = asyncio.Semaphore(concurrency)

        async def _enrich(scene: SceneData) -> SceneData:
            async with sem:
                scene.video_prompt = await self.generate(scene)
            return scene

        return await asyncio.gather(*[_enrich(s) for s in scenes])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def _llm_prompt(
        self, text: str, style: str, camera: str, duration: float
    ) -> str:
        client = self._get_client()
        user_msg = USER_TEMPLATE.format(
            text=text[:800],  # guard token limit
            style=style,
            camera=camera.replace("_", " "),
            duration=int(duration),
        )
        response = await client.messages.create(
            model=self.model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()
