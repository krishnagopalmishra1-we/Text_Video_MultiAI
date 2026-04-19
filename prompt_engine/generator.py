"""
Prompt engine: converts scene text → rich video generation prompts.
Uses Gemma 4 31B (Google AI Studio) for semantic enrichment, with local fallback.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml

from scene_splitter import SceneData
from .presets import build_prompt, STYLE_PRESETS, CAMERA_PRESETS


SYSTEM_PROMPT = """You are a cinematic video prompt engineer.
Convert scene text into a visual description for video generation.
Rules:
- Max 80 words
- Describe ONLY: the visual subject, objects in frame, composition, and camera motion
- Do NOT describe lighting, colors, atmosphere, mood, or tone — these are set by the style preset
- Do NOT include moonlight, fog color, shadow color, or any lighting descriptors
- No dialogue, no subtitles, no text on screen
- No first/second person
- Output ONLY the visual description, nothing else"""

USER_TEMPLATE = """Scene text:
{text}

Camera: {camera}
Duration: {duration}s

Visual description (subject and composition only, NO lighting):"""


class PromptEngine:
    def __init__(
        self,
        presets_path: str | Path = "config/presets.yaml",
        style: str = "cinematic",
        camera: str = "dolly_forward",
        use_llm: bool = True,
        model: str = "gemma-4-31b-it",
    ):
        self.style = style
        self.camera = camera
        self.use_llm = use_llm
        self.model = model

        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, scene: SceneData) -> str:
        style = scene.style_hint or self.style
        camera = scene.camera_hint or self.camera

        if self.use_llm:
            try:
                raw = await self._llm_prompt(scene.text, style, camera, scene.duration)
                return build_prompt(raw, style, camera, self.presets)
            except Exception as e:
                import logging as _log, traceback as _tb
                _log.getLogger(__name__).warning(
                    "LLM prompt generation failed for scene %s: %s\n%s",
                    scene.scene_id, e, _tb.format_exc()
                )

        return build_prompt(scene.narration or scene.text, style, camera, self.presets)

    async def generate_batch(
        self, scenes: list[SceneData], concurrency: int = 4
    ) -> list[SceneData]:
        sem = asyncio.Semaphore(concurrency)

        async def _enrich(scene: SceneData) -> SceneData:
            async with sem:
                if not scene.video_prompt:
                    scene.video_prompt = await self.generate(scene)
            return scene

        return await asyncio.gather(*[_enrich(s) for s in scenes])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        key = os.environ.get("Gemini_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError("Gemini_API_KEY not set")
        return key

    async def _llm_prompt(
        self, text: str, style: str, camera: str, duration: float
    ) -> str:
        import requests as _req
        api_key = self._get_api_key()
        user_msg = USER_TEMPLATE.format(
            text=text[:800],
            camera=camera.replace("_", " "),
            duration=int(duration),
        )
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_msg
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: _req.post(url, params={"key": api_key}, json=payload, timeout=30),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
