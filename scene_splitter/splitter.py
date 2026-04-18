"""
Scene splitter: breaks a long script into timed scenes.
Supports plain text, JSON, and SRT input formats.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml

from .utils import estimate_duration, clean_text, split_sentences

PACING_WPS = {"slow": 2.0, "normal": 2.5, "fast": 3.0}
PACING_BUFFER = {"slow": 2.0, "normal": 1.5, "fast": 1.0}


@dataclass
class SceneData:
    scene_id: int
    text: str
    duration: float          # seconds
    word_count: int
    camera_hint: str = ""    # extracted from [brackets] in script
    style_hint: str = ""
    narration: str = ""      # cleaned TTS text
    video_prompt: str = ""   # filled by prompt_engine

    def to_dict(self) -> dict:
        return asdict(self)


class SceneSplitter:
    def __init__(
        self,
        presets_path: str | Path = "config/presets.yaml",
        pacing: Literal["slow", "normal", "fast"] = "normal",
        min_duration: float = 5.0,
        max_duration: float = 20.0,
        target_duration: float = 10.0,
        max_words_per_scene: int = 80,
    ):
        self.pacing = pacing
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.max_words = max_words_per_scene
        self.wps = PACING_WPS[pacing]
        self.buffer = PACING_BUFFER[pacing]

        with open(presets_path) as f:
            self.presets = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, script: str | Path | dict | list) -> list[SceneData]:
        """Auto-detect format and split into scenes."""
        if isinstance(script, Path):
            script = script.read_text(encoding="utf-8")

        if isinstance(script, list):
            return self._from_json_list(script)

        if isinstance(script, dict):
            return self._from_json_list(script.get("scenes", [script]))

        if isinstance(script, str):
            stripped = script.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                    return self.split(data)
                except json.JSONDecodeError:
                    pass
            if re.match(r"^\d+\n\d{2}:\d{2}:\d{2}", stripped):
                return self._from_srt(stripped)
            return self._from_plain_text(stripped)

        raise ValueError(f"Unsupported script type: {type(script)}")

    def split_file(self, path: str | Path) -> list[SceneData]:
        p = Path(path)
        return self.split(p.read_text(encoding="utf-8"))

    def to_json(self, scenes: list[SceneData], path: str | Path | None = None) -> str:
        data = json.dumps([s.to_dict() for s in scenes], indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(data, encoding="utf-8")
        return data

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _from_plain_text(self, text: str) -> list[SceneData]:
        # Split on double newline (paragraph), then merge short paragraphs
        raw_paragraphs = re.split(r"\n{2,}", text.strip())
        chunks = self._merge_chunks(raw_paragraphs)
        return self._build_scenes(chunks)

    def _from_srt(self, srt_text: str) -> list[SceneData]:
        blocks = re.split(r"\n{2,}", srt_text.strip())
        chunks: list[str] = []
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) >= 3:
                content = " ".join(lines[2:])
                chunks.append(content)
        merged = self._merge_chunks(chunks)
        return self._build_scenes(merged)

    def _from_json_list(self, data: list) -> list[SceneData]:
        scenes: list[SceneData] = []
        for i, item in enumerate(data, start=1):
            if isinstance(item, str):
                text = item
                duration = None
            else:
                text = item.get("text", "")
                duration = item.get("duration")

            video_prompt = item.get("video_prompt", "") if isinstance(item, dict) else ""
            text = clean_text(text)
            wc = len(text.split())
            dur = duration or estimate_duration(wc, self.wps, self.buffer)
            dur = max(self.min_duration, min(self.max_duration, dur))
            scenes.append(SceneData(
                scene_id=i,
                text=text,
                duration=round(dur, 2),
                word_count=wc,
                narration=text,
                video_prompt=video_prompt,
                **self._extract_hints(text),
            ))
        return scenes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_chunks(self, paragraphs: list[str]) -> list[str]:
        """Merge short paragraphs until they approach target word count."""
        chunks: list[str] = []
        current: list[str] = []
        current_wc = 0

        for para in paragraphs:
            para = clean_text(para)
            if not para:
                continue
            wc = len(para.split())

            if current_wc + wc <= self.max_words:
                current.append(para)
                current_wc += wc
            else:
                if current:
                    chunks.append(" ".join(current))
                # If single paragraph exceeds max_words, split by sentences
                if wc > self.max_words:
                    sub = self._split_long_paragraph(para)
                    chunks.extend(sub[:-1])
                    current = [sub[-1]] if sub else []
                    current_wc = len(current[0].split()) if current else 0
                else:
                    current = [para]
                    current_wc = wc

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _split_long_paragraph(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        chunks: list[str] = []
        current: list[str] = []
        current_wc = 0

        for sent in sentences:
            wc = len(sent.split())
            if current_wc + wc <= self.max_words:
                current.append(sent)
                current_wc += wc
            else:
                if current:
                    chunks.append(" ".join(current))
                current = [sent]
                current_wc = wc

        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def _build_scenes(self, chunks: list[str]) -> list[SceneData]:
        scenes: list[SceneData] = []
        for i, chunk in enumerate(chunks, start=1):
            wc = len(chunk.split())
            dur = estimate_duration(wc, self.wps, self.buffer)
            dur = max(self.min_duration, min(self.max_duration, dur))
            hints = self._extract_hints(chunk)
            narration = re.sub(r"\[.*?\]", "", chunk).strip()
            scenes.append(SceneData(
                scene_id=i,
                text=chunk,
                duration=round(dur, 2),
                word_count=wc,
                narration=narration,
                **hints,
            ))
        return scenes

    def _extract_hints(self, text: str) -> dict:
        """Extract [CAMERA: ...] and [STYLE: ...] hints from text."""
        camera = ""
        style = ""
        cam_match = re.search(r"\[CAMERA:\s*([^\]]+)\]", text, re.IGNORECASE)
        style_match = re.search(r"\[STYLE:\s*([^\]]+)\]", text, re.IGNORECASE)
        if cam_match:
            camera = cam_match.group(1).strip().lower().replace(" ", "_")
        if style_match:
            style = style_match.group(1).strip().lower().replace(" ", "_")
        return {"camera_hint": camera, "style_hint": style}
