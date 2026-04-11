# AGENT.md — Long Video Generator

## Project Overview
AI-powered long-form video generation system (10–20 min).
Runs on: NVIDIA A100 80GB, 48 CPU cores.

## Architecture
```
SCRIPT → SceneSplitter → PromptEngine → VideoRouter → Stitcher → AudioSync → FINAL_VIDEO
```

## Key Modules

| Module | File | Role |
|---|---|---|
| SceneSplitter | `scene_splitter/splitter.py` | Split script → timed scenes |
| PromptEngine | `prompt_engine/generator.py` | Scene text → video prompt (via Claude API) |
| VideoRouter | `video_engine/router.py` | Route to best model + fallback chain |
| LocalRunner | `video_engine/local_runner.py` | Manage GPU models on A100 |
| Wan2Runner | `video_engine/models/wan2.py` | Primary: Wan2.1-14B (~40GB VRAM) |
| HunyuanRunner | `video_engine/models/hunyuan.py` | Hero scenes: HunyuanVideo (~60GB) |
| CogVideoXRunner | `video_engine/models/cogvideox.py` | Fast: CogVideoX-5B (~24GB) |
| LTXRunner | `video_engine/models/ltx.py` | Fastest/preview: LTX-Video (~10GB) |
| APIRunner | `video_engine/api_runner.py` | API fallback: Runway/Pika/Veo/Higgsfield |
| Stitcher | `stitcher/ffmpeg_stitch.py` | FFmpeg concat + transitions + 4K upscale |
| TTSEngine | `audio/tts.py` | Kokoro (local) or ElevenLabs (API) |
| MusicEngine | `audio/music.py` | MusicGen-large (local) |
| AudioSync | `audio/sync.py` | Mix narration + music |
| FastAPI server | `server/main.py` | REST API + WebSocket progress |
| Celery tasks | `server/tasks.py` | Async job orchestration |
| DB | `db/models.py` | SQLite job/scene state |

## Model Priority (A100 80GB)
```
1. Wan2.1-14B   → high quality, ~40GB VRAM, default
2. HunyuanVideo → cinema quality, ~60GB VRAM, hero scenes only
3. CogVideoX-5B → fast, ~24GB VRAM
4. LTX-Video    → fastest, ~10GB VRAM, preview
5. Runway Gen-3 → API fallback
6. Pika 2.0     → API fallback
7. Veo 2        → API fallback
8. Higgsfield   → API fallback
```

## CLI Quick Reference
```bash
# Full pipeline (CLI)
python orchestrator.py --script script.txt --style cinematic --quality high

# FastAPI server
uvicorn server.main:app --host 0.0.0.0 --port 8000

# GPU Celery worker
celery -A server.tasks.celery_app worker --queues=gpu --concurrency=1

# CPU Celery workers (uses 8 of 48 cores; increase --concurrency as needed)
celery -A server.tasks.celery_app worker --queues=cpu --concurrency=8

# Docker (full stack)
docker-compose up --build
```

## Environment Variables
```
ANTHROPIC_API_KEY     Claude API (prompt generation)
RUNWAY_API_KEY        Runway Gen-3
PIKA_API_KEY          Pika 2.0
VEO_API_KEY           Google Veo
GCP_PROJECT_ID        Google Cloud project
HIGGSFIELD_API_KEY    Higgsfield
ELEVENLABS_API_KEY    ElevenLabs TTS (fallback)
ELEVENLABS_VOICE_ID   Voice ID (default: Rachel)
DATABASE_URL          SQLite or Postgres URL
CELERY_BROKER_URL     Redis URL
CELERY_RESULT_BACKEND Redis URL
```

## Config Files
- `config/model_config.yaml` — model parameters, VRAM thresholds, resolution
- `config/api_keys.yaml`     — API key templates (use env vars)
- `config/presets.yaml`      — style/camera presets, audio settings

## Output Structure
```
outputs/
  {job_id}/
    scenes/         scene_0001.mp4 … scene_NNNN.mp4
    audio/          narration.wav, music.wav, mixed.aac
    final/          {job_id}.mp4  (or _4k.mp4)
  jobs.db           SQLite state
  pipeline.log      Full run log
```

## Adding a New Model
1. Create `video_engine/models/{name}.py` implementing `.load()`, `.unload()`, `.generate()`
2. Register in `video_engine/models/__init__.py`
3. Add to `_RUNNERS` dict in `video_engine/local_runner.py`
4. Add VRAM threshold to `_VRAM_THRESHOLDS`
5. Add config block in `config/model_config.yaml`

## Performance Notes (A100 80GB, 48 cores)
- Single A100 = sequential GPU inference (one model at a time)
- Wan2.1-14B: ~2–4 min per 10s clip at 1280×720
- CogVideoX-5B: ~1–2 min per 6s clip at 720×480
- LTX-Video: ~30s per 5s clip at 768×512
- FFmpeg uses `--threads 48` for encode/decode
- Celery GPU queue: concurrency=1 (GPU single-tenant)
- Celery CPU queue: concurrency=8 (scale up to 48 for pure CPU tasks)

## Operational Status (2026-04-11)
- VM `video-gen-a100` in `VIDEO-GEN-RG` was deallocated after testing to stop billing.
- WAN smoke generation produced valid 5.06s video artifacts at 1280x720.
- WAN `slow_conv3d_forward` CUDA failure was mitigated by disabling VAE tiling by default in `video_engine/models/wan2.py`.
- Hunyuan model id path was corrected to `hunyuanvideo-community/HunyuanVideo`.
- Hunyuan full-quality (`50` steps, 1280x720, 5s) remains unstable in this stack:
  - `torch.compile` path can fail with invalid huge allocation plans.
  - non-compile path is stable but can be too slow for smoke window completion.
- Current smoke artifacts are model-level scene clips only; full pipeline final video with muxed audio was not produced in this smoke pass.

## Cleanup Notes (2026-04-11)
- Temporary smoke/debug shell scripts and ad-hoc smoke runner files created during troubleshooting were removed from the project root.
- Only reusable model/runtime code changes and this operational note were retained.
