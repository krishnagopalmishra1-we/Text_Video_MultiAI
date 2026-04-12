#!/usr/bin/env python3
"""Quick single-job test to verify pipeline connectivity."""
import json
import sys
import httpx

r = httpx.post("http://localhost:8000/generate_video", json={
    "script": "A golden sunrise over ancient mountains, mist rolling through valleys. Cinematic aerial camera sweep over peaks and forests.",
    "strategy": "fast",
    "preferred_model": "wan2_1b",
    "api_fallback": False,
    "use_dag": True,
    "min_clip": 5,
    "max_clip": 10,
}, timeout=30)
print(r.status_code, r.text[:500])
