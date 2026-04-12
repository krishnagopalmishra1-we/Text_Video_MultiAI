# Smoke Test Results — 2026-04-12

## Environment
- VM: Azure NCads A100 V4 (`video-gen-a100`, resource group `video-gen-rg`)
- IP: 172.177.139.168
- GPU: NVIDIA A100 80GB
- Stack: Docker Compose (`text_video_multiai`), PostgreSQL, Redis, Celery (concurrency=1, solo pool)
- DB: PostgreSQL (`videogen` db, `videogen` user, via `postgres` hostname inside Docker network)

## Test Matrix
12 runs: 4 models × 3 strategies (fast / balanced / quality)

Script used (single cinematic scene):
> "A futuristic golden-hour street bathed in warm amber light. Dense crowds of humans fill
> the sidewalks — faces illuminated by neon reflections from towering holographic billboards.
> Street performers and dancers move through the crowd, their movements fluid and expressive.
> Close-ups reveal micro-expressions of wonder and joy. The camera glides smoothly through
> the scene, then rises above the rooftops where fireworks burst against the twilight sky.
> Cinematic pacing with narration and ambient music."

## Results

| # | Run | Model | Strategy | Job ID | Status | Gen Time | Final MP4 |
|---|-----|-------|----------|--------|--------|----------|-----------|
| 1 | WAN13B_FAST | wan2_1b | fast | d70cea94 | ✅ done | ~8 min | 4.0 MB, 5.08s @ 1920×1080 |
| 2 | WAN13B_BALANCED | wan2_1b | balanced | 2a873d7f | ✅ done | ~2 min | 3.6 MB, 5.08s @ 1920×1080 |
| 3 | WAN13B_QUALITY | wan2_1b | quality | 6b1eba86 | ✅ done | ~3 min | 4.2 MB, 5.08s @ 1920×1080 |
| 4 | WAN14B_FAST | wan2_14b | fast | 8ec03802 | ✅ done | ~29 min | 5.1 MB, 5.08s @ 1920×1080 |
| 5 | WAN14B_BALANCED | wan2_14b | balanced | 0d06f777 | ✅ done | ~8 min | 4.3 MB, 5.08s @ 1920×1080 |
| 6 | WAN14B_QUALITY | wan2_14b | quality | 4822a4fb | ✅ done | ~12 min | 4.5 MB, 5.08s @ 1920×1080 |
| 7 | HUNYUAN_FAST | hunyuan | fast | 1593c8f6 | ✅ done | ~57 min | 4.3 MB, 5.38s @ 1920×1080 |
| 8 | HUNYUAN_BALANCED | hunyuan | balanced | cccde320 | ✅ done | ~37 min | 4.6 MB, 5.38s @ 1920×1080 |
| 9 | HUNYUAN_QUALITY | hunyuan | quality | d4690e70 | ✅ done | ~26 min | 4.6 MB, 5.38s @ 1920×1080 |
| 10 | COGVID_FAST | cogvideox | fast | 4ab82961 | ❌ failed | instant | — |
| 11 | COGVID_BALANCED | cogvideox | balanced | cee7b10a | ❌ failed | instant | — |
| 12 | COGVID_QUALITY | cogvideox | quality | b6f655ca | ❌ failed | instant | — |

**9/12 passed. All 3 CogVideoX runs failed.**

## Pipeline Verification (for passing runs)

All 9 successful jobs completed the full pipeline end-to-end:
- Scene splitting → prompt generation → video generation → Real-ESRGAN upscale → FFmpeg stitch
- Output: 1920×1080 MP4 confirmed via `ffprobe`
- WAN outputs: 5.084s duration
- Hunyuan outputs: 5.375s duration (higher frame count at 24fps)
- All `scene.clip_path` populated in PostgreSQL, no scene-level errors

## Issues Found

### 1. CogVideoX — `Illegal header value b'Bearer '` (FATAL, model removed)
**Cause**: `HF_TOKEN` environment variable was empty/blank inside the `worker_gpu` Docker container.
This produced a malformed HTTP header `Authorization: Bearer ` (trailing space, no token value),
which `httpx`/`requests` immediately rejected before any model could load.

**Symptoms**: All 3 CogVideoX jobs failed within seconds of submission (no generation attempt made).

**Root cause path**:
- `CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b")` calls HuggingFace Hub API
- Hub uses `HF_TOKEN` env var for `Authorization: Bearer <token>`
- Token was empty → header value was `b'Bearer '` → rejected by httpx

**Resolution**: CogVideoX removed entirely from the model stack. THUDM/CogVideoX-5b requires
HuggingFace authentication even though the repo is public gated. The model also offers lower
quality than WAN 14B at similar VRAM (24GB vs 42GB) with only 8fps output.

**If re-adding later**: Ensure `HF_TOKEN` is set in `docker-compose.yml` under `worker_gpu`
environment, or switch to the non-gated variant `THUDM/CogVideoX-5b-I2V`.

### 2. WAN14B_FAST was slow (29 min vs expected ~10 min)
**Cause**: Fast strategy uses wan2_14b at 1280×720 with 20 steps. This combination is slower
than expected because 14B at full resolution is inherently slower than 1.3B.
The CLAUDE.md strategy table said "~10 min for 12 clips" for fast, but that refers to wan2_1b.
WAN14B_FAST forces the 14B model even in the fast strategy (when user explicitly sets
`preferred_model=wan2_14b`). Normal fast strategy would use wan2_1b.

**Resolution**: Not a bug. Behavior is correct — user explicitly requested wan2_14b in the
smoke test. No code change needed.

### 3. HunyuanVideo FAST was slowest (57 min)
**Observation**: Hunyuan in the fast strategy still runs its full INT8 pipeline (it has no
reduced-step mode for the fast strategy — `hero_steps: 30` is the minimum config).
At 24fps and higher frame count per clip, Hunyuan is inherently slower than WAN.

**Resolution**: Document that Hunyuan is not intended for the fast strategy; it is designed
for hero scenes in the quality strategy only. If a user requests `preferred_model=hunyuan`
with `strategy=fast`, they will get a slow result. Consider adding a warning in the API.

### 4. PostgreSQL `model_used` column always NULL
**Observation**: The `scene.model_used` column was NULL for all 9 completed jobs.

**Root cause**: `server/tasks.py` does not write `model_used` back to the Scene row after
generation. The `LocalRunner.generate()` returns a `Path`, not a model name.

**Impact**: Low — does not affect output quality or pipeline correctness.
**To fix**: Have `run_pipeline` in `tasks.py` record the selected model name when writing the
Scene record, or return it from `VideoRouter.generate_scene()`.

### 5. `smoke_test_12.py` used `httpx` but production container may not have it
**Observation**: `smoke_test_12.py` imports `httpx`. The deployed `smoke_test.py` on the server
correctly used `urllib` only (no extra dependencies). If re-running the 12-run version on the
server, install `httpx` first (`pip install httpx`) or use the `scripts/smoke_test.py` version.

## What Was Removed

- `video_engine/models/cogvideox.py` — deleted
- `video_engine/models/__init__.py` — removed `CogVideoXRunner` import and registry entry
- `video_engine/local_runner.py` — removed `cogvideox` from `_VRAM_THRESHOLDS`, type hints,
  and `_select_model` fallback logic
- `config/model_config.yaml` — removed `cogvideox` model block and all `quality_presets` entries
- `orchestrator.py` — removed `cogvideox` from `--preferred-model` choices
- `server/main.py` — removed `cogvideox` from `preferred_model` comment
- `scripts/smoke_test.py` + `scripts/smoke_test_12.py` — removed COGVID_* runs

## Outstanding Work

1. **Fix `scene.model_used` not being written** — update `server/tasks.py` to record model name
2. **Add Hunyuan fast-strategy warning** — log/surface a warning if hunyuan is used with
   strategy=fast, since it will be significantly slower than wan2_1b
3. **`smoke_test_12.py` httpx dependency** — either replace with urllib or add to requirements
4. **Consider LTX-Video as replacement fast fallback** — 10GB VRAM, no auth required,
   can serve the role cogvideox was meant to fill (fast preview/bulk scenes)
