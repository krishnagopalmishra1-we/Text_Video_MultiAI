"""Runway Gen-4 API wrapper."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


class RunwayNoCreditsError(RuntimeError):
    """Raised when Runway rejects the request due to insufficient credits (HTTP 400)."""

_BASE_URL = "https://api.dev.runwayml.com/v1"
_POLL_INTERVAL = 5
_TIMEOUT = 300


class RunwayClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["RUNWAY_API_KEY"]
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06",
        }

    async def generate(
        self,
        prompt: str,
        duration: int = 10,
        output_path: str | Path = "output.mp4",
        width: int = 1280,
        height: int = 768,
        seed: int | None = None,
    ) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        task_id = await self._submit(prompt, duration, width, height, seed)
        video_url = await self._poll(task_id)
        await self._download(video_url, out)
        logger.info(f"Runway → {out}")
        return out

    @staticmethod
    def _ratio(width: int, height: int) -> str:
        # gen4.5 only accepts 1280:720 or 720:1280
        if height > width:
            return "720:1280"
        return "1280:720"

    async def _submit(
        self, prompt: str, duration: int, width: int, height: int, seed: int | None
    ) -> str:
        body: dict = {
            "model": "gen4.5",
            "promptText": prompt[:512],
            "duration": min(duration, 10),
            "ratio": self._ratio(width, height),
        }
        if seed is not None:
            body["seed"] = seed

        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{_BASE_URL}/text_to_video",
                headers=self._headers,
                json=body,
            )
            if r.is_error:
                if r.status_code == 400 and "credits" in r.text.lower():
                    raise RunwayNoCreditsError(
                        f"Runway has no credits: {r.text[:300]}"
                    )
                raise RuntimeError(f"Runway submit failed {r.status_code}: {r.text[:500]}")
            return r.json()["id"]

    async def _poll(self, task_id: str) -> str:
        deadline = time.time() + _TIMEOUT
        async with httpx.AsyncClient(timeout=30) as client:
            while time.time() < deadline:
                r = await client.get(
                    f"{_BASE_URL}/tasks/{task_id}",
                    headers=self._headers,
                )
                r.raise_for_status()
                data = r.json()
                status = data.get("status")
                if status == "SUCCEEDED":
                    return data["output"][0]
                if status in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"Runway task {task_id} {status}: {data}")
                await asyncio.sleep(_POLL_INTERVAL)
        raise TimeoutError(f"Runway task {task_id} timed out after {_TIMEOUT}s")

    @staticmethod
    async def _download(url: str, path: Path) -> None:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.get(url)
            r.raise_for_status()
            path.write_bytes(r.content)
