# CLAUDE.md — Memory & Working Context

## Project
Long-form AI video generator. A100 80GB + 48 CPU cores.
Three generation strategies: fast / balanced / quality.

## Hardware Context
- GPU: NVIDIA A100 80GB (NCads A100 V4 — Azure)
- CPU: 48 cores
- Primary precision: bfloat16 (native A100)
- Attention: SageAttention preferred (1.5–1.7× faster than FlashAttn2), FA2 as fallback

## Model Stack (in priority order)
| Priority | Model | VRAM | Use case | Acceleration |
|---|---|---|---|---|
| 1 | Wan2.1-T2V-14B | ~42GB | Default high quality (balanced/quality) | SageAttn + PAB + TeaCache + FBCache + compile |
| 2 | Wan2.1-T2V-1.3B | ~8GB | Fast strategy | SageAttn + TeaCache + compile |
| 3 | HunyuanVideo INT8 | ~37GB | Hero/cinematic scenes (quality only) | bitsandbytes INT8 quantization |
| 4 | CogVideoX-5B | ~24GB | Fallback | standard |
| 5 | LTX-Video | ~10GB | Preview/fastest | standard |
| 6 | Runway Gen-3 | API | Final fallback (only implemented API) | — |

## Generation Strategies
- **Fast**: WAN 1.3B, 1280×720 native, 20 steps, ~10 min for 12 clips
- **Balanced**: WAN 14B, 848×480 + Real-ESRGAN upscale to 1280×720, 15 steps, ~20 min
- **Quality**: WAN 14B bulk + Hunyuan INT8 hero scenes, 848×480 + upscale, 15/30 steps, ~25 min

## Output Resolutions
- Generation: 848×480 (balanced/quality) or 1280×720 (fast)
- Upscale: Real-ESRGAN 848×480 → 1280×720 (video_engine/upscaler.py)
- Stitch: 1920×1080 (1080p via FFmpeg scale)
- Optional: 3840×2160 (4K upscale)

## Key Design Decisions
- **One model at a time** on GPU — sequential generation, no model parallelism
- **Celery GPU queue concurrency=1** — enforces single-tenant GPU
- **GPUMemoryManager** (pipeline/gpu_manager.py) — tracks VRAM, forces GC after unload
- **DAG orchestration** (pipeline/dag.py) — Celery canvas for parallel CPU tasks
- **Resume mode** on by default — skip already-generated clips on restart
- **Claude Haiku** for prompt generation (fast + cheap for bulk scene processing)
- **Kokoro TTS** only (local, 82M params) — no ElevenLabs dependency
- **MusicGen-large** for background score generation
- **SQLite** for job state (swap DATABASE_URL for Postgres in production)
- **VideoRouter singleton** in tasks.py — avoids recreating per request
- **RunwayClient singleton** in api_runner.py — reuses HTTP connection
- **Compile order matters** — SageAttn → PAB → caches → compile (last)
- **Text encoder offload** — moved to CPU after encoding to free VRAM for generation

## Strict Rules When Modifying
- Never change Wan2 acceleration order: SageAttn → PAB → TeaCache → FBCache → compile
- Never add model CPU offload without updating `_VRAM_THRESHOLDS` in local_runner.py
- Never change `worker_prefetch_multiplier` from 1 on GPU worker (causes OOM)
- FFmpeg threads must be passed as string, not int: `str(self.cpu_threads)`
- All model `.generate()` methods must accept `output_path` and return `Path`
- Scene IDs are 1-indexed and used as filenames: `scene_0001.mp4`
- Model key for WAN is `wan2_14b` or `wan2_1b` in quality_presets, not plain `wan2`

## File Naming Convention
- Scene clips: `outputs/{job_id}/scenes/scene_{id:04d}.mp4`
- Narration: `outputs/{job_id}/audio/narration.wav`
- Music: `outputs/{job_id}/audio/music.wav`
- Mixed audio: `outputs/{job_id}/audio/mixed.aac`
- Final video: `outputs/{job_id}/final/{job_id}.mp4`
- 4K final: `outputs/{job_id}/final/{job_id}_4k.mp4`

## Environment Setup Order
```bash
# 1. CUDA + PyTorch (do first — everything depends on this)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. SageAttention (preferred) + Flash Attention (fallback)
pip install sageattention
pip install flash-attn --no-build-isolation

# 3. Everything else
pip install -r requirements.txt

# 4. Set env vars
export ANTHROPIC_API_KEY=...
export RUNWAY_API_KEY=...
export HF_TOKEN=...
```

## Common Issues
- **OOM on Wan2.1-14B**: text encoder auto-offloads to CPU after encoding; if still OOM, reduce resolution or switch to 1.3B
- **HunyuanVideo OOM**: uses INT8 quantization (~37GB); if still OOM, skip and fallback to CogVideoX
- **SageAttention not found**: falls back to FlashAttn2; install with `pip install sageattention`
- **Slow FFmpeg**: verify `--threads 48` is being passed; check `cpu_threads` in Stitcher
- **Celery not picking up GPU tasks**: ensure worker started with `--queues=gpu`
- **Kokoro not installed**: `pip install kokoro>=0.9.4` — requires espeak-ng on Linux
- **Strategy not applied**: check quality_presets keys in model_config.yaml use `wan2_14b`/`wan2_1b` not `wan2`
