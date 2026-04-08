# CLAUDE.md — Memory & Working Context

## Project
Long-form AI video generator. A100 80GB + 48 CPU cores.

## Hardware Context
- GPU: NVIDIA A100 80GB (NVCads A100 V4 — Azure)
- CPU: 48 cores
- Primary precision: bfloat16 (native A100)
- Flash Attention 2: enabled (sm_80 arch)

## Model Stack (in priority order)
| Priority | Model | VRAM | Use case |
|---|---|---|---|
| 1 | Wan2.1-14B | ~40GB | Default high quality |
| 2 | HunyuanVideo | ~60GB | Hero/cinematic scenes |
| 3 | CogVideoX-5B | ~24GB | Fast bulk scenes |
| 4 | LTX-Video | ~10GB | Preview/fallback |
| 5+ | Runway/Pika/Veo/Higgsfield | API | Final fallback |

## Output Resolutions
- Native generation: 1280×720 (Wan2, Hunyuan, LTX) or 720×480 (CogVideoX)
- Intermediate stitch: 1920×1080 (1080p via FFmpeg scale)
- Optional upscale: 3840×2160 (4K via Real-ESRGAN x4)

## Key Design Decisions
- **One model at a time** on GPU — sequential generation, no model parallelism
- **Celery GPU queue concurrency=1** — enforces single-tenant GPU
- **Resume mode** on by default — skip already-generated clips on restart
- **Claude Haiku** for prompt generation (fast + cheap for bulk scene processing)
- **Kokoro TTS** (local, 82M params) as primary; ElevenLabs as API fallback
- **MusicGen-large** for background score generation
- **SQLite** for job state (swap DATABASE_URL for Postgres in production)

## Strict Rules When Modifying
- Never add model CPU offload to Wan2.1 or HunyuanVideo without updating `_VRAM_THRESHOLDS`
- Never change `worker_prefetch_multiplier` from 1 on GPU worker (causes OOM)
- FFmpeg threads must be passed as string, not int: `str(self.cpu_threads)`
- All model `.generate()` methods must accept `output_path` and return `Path`
- Scene IDs are 1-indexed and used as filenames: `scene_0001.mp4`

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

# 2. Flash Attention (must be after PyTorch)
pip install flash-attn --no-build-isolation

# 3. Everything else
pip install -r requirements.txt

# 4. Copy and fill env file
cp config/api_keys.yaml config/api_keys.local.yaml
export ANTHROPIC_API_KEY=...
```

## Common Issues
- **OOM on Wan2.1**: set `offload: cpu` in model_config.yaml for text_encoder
- **HunyuanVideo OOM**: already offloads text_encoder; if still OOM, switch to CogVideoX
- **Slow FFmpeg**: verify `--threads 48` is being passed; check `cpu_threads` in Stitcher
- **Celery not picking up GPU tasks**: ensure worker started with `--queues=gpu`
- **Kokoro not installed**: `pip install kokoro>=0.9.4` — requires espeak-ng on Linux
