# AGENT.md — Long Video Generator

## Project Overview
AI-powered long-form video generation system (1–2 min target in ~27 min).
Runs on: NVIDIA A100 80GB, 48 CPU cores.

## Architecture
```
SCRIPT → SceneSplitter → PromptEngine → VideoRouter → Upscaler → Stitcher → AudioSync → QualityCheck → FINAL_VIDEO
```

DAG pipeline orchestration via Celery canvas (pipeline/dag.py).

## Generation Strategies

| Strategy | Models | Resolution | Time (12 clips) |
|----------|--------|------------|-----------------|
| Fast | WAN 1.3B | 1280×720 native | ~10 min |
| Balanced | WAN 14B | 848×480 + upscale | ~20 min |
| Quality | WAN 14B + Hunyuan INT8 hero | 848×480 + upscale | ~25 min |

## Key Modules

| Module | File | Role |
|---|---|---|
| SceneSplitter | `scene_splitter/splitter.py` | Split script → timed scenes |
| PromptEngine | `prompt_engine/generator.py` | Scene text → video prompt (via Claude API) |
| VideoRouter | `video_engine/router.py` | Route to best model based on strategy + VRAM |
| LocalRunner | `video_engine/local_runner.py` | Manage GPU models on A100 (singleton) |
| Wan2Runner | `video_engine/models/wan2.py` | WAN 2.1-14B (~42GB) / 1.3B (~8GB) |
| HunyuanRunner | `video_engine/models/hunyuan.py` | HunyuanVideo INT8 (~37GB) — hero scenes |
| CogVideoXRunner | `video_engine/models/cogvideox.py` | CogVideoX-5B (~24GB) — fallback |
| LTXRunner | `video_engine/models/ltx.py` | LTX-Video (~10GB) — fastest/preview |
| APIRunner | `video_engine/api_runner.py` | API fallback: Runway Gen-3 (singleton client) |
| Upscaler | `video_engine/upscaler.py` | Real-ESRGAN 848×480 → 1280×720 |
| Stitcher | `stitcher/ffmpeg_stitch.py` | FFmpeg concat + xfade transitions |
| TTSEngine | `audio/tts.py` | Kokoro (local, 82M params) |
| MusicEngine | `audio/music.py` | MusicGen-large (local) |
| AudioSync | `audio/sync.py` | Mix narration + music |
| DAGBuilder | `pipeline/dag.py` | Celery canvas DAG orchestration |
| GPUMemoryManager | `pipeline/gpu_manager.py` | VRAM tracking + garbage collection |
| QualityCheck | `pipeline/quality_check.py` | Detect black/frozen frames |
| FastAPI server | `server/main.py` | REST API + WebSocket progress + lifespan |
| Celery tasks | `server/tasks.py` | Async job orchestration (singleton router) |
| DB | `db/models.py` | SQLite job/scene state |

## Model Priority (A100 80GB)
```
1. Wan2.1-T2V-14B   → primary, ~42GB VRAM, SageAttn+TeaCache+FBCache+compile
2. Wan2.1-T2V-1.3B  → fast strategy, ~8GB VRAM, SageAttn+TeaCache+compile
3. HunyuanVideo INT8 → hero scenes (quality strategy), ~37GB VRAM
4. CogVideoX-5B     → fallback, ~24GB VRAM
5. LTX-Video        → fastest/preview, ~10GB VRAM
6. Runway Gen-3     → API fallback (only implemented API)
```

## Acceleration Stack (wan2.py)
Applied in order: SageAttention → PAB → TeaCache → FBCache → torch.compile(max-autotune)
Text encoder offloaded to CPU after encoding to free VRAM for generation.

## CLI Quick Reference
```bash
# Full pipeline (CLI)
python orchestrator.py --script script.txt --style cinematic --quality high --strategy balanced

# FastAPI server
uvicorn server.main:app --host 0.0.0.0 --port 8000

# GPU Celery worker
celery -A server.tasks.celery_app worker --queues=gpu --concurrency=1

# CPU Celery workers
celery -A server.tasks.celery_app worker --queues=cpu --concurrency=8

# Docker (full stack)
docker-compose up --build
```

## Environment Variables
```
ANTHROPIC_API_KEY     Claude API (prompt generation)
RUNWAY_API_KEY        Runway Gen-3
HF_TOKEN              Hugging Face (model downloads)
DATABASE_URL          SQLite or Postgres URL
CELERY_BROKER_URL     Redis URL
CELERY_RESULT_BACKEND Redis URL
```

## Config Files
- `config/model_config.yaml` — model parameters, VRAM thresholds, resolution, quality_presets per strategy
- `config/presets.yaml`      — style presets (cinematic, documentary, etc.)
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

## Postmortem Guardrails (2026-04-12)
- Always verify runtime container state before rerunning tests:
  - `docker exec text_video_multiai_worker_gpu_1 grep compile /app/config/model_config.yaml`
  - `docker exec text_video_multiai_worker_gpu_1 grep task_acks /app/server/tasks.py`
  - `docker exec text_video_multiai_redis_1 redis-cli hlen unacked`
- If code changed in worker images, do not rely on `docker-compose restart`; rebuild and recreate worker containers.
- Mount `./config:/app/config:ro` on both GPU and CPU workers so model/runtime config changes apply immediately.
- After crash loops, flush stale broker state before retesting (`redis-cli FLUSHDB`) to remove orphaned `unacked` tasks.
- For this stack/PyTorch build, keep `compile: false` for WAN/Cog/Hunyuan unless compatibility is revalidated.
- Before running long smoke batches, run one complete end-to-end canary (split → prompt → video → audio → stitch → done).
