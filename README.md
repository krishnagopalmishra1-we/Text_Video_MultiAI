# Long Video Generator

AI-powered long-form video generation (1–2 min target in ~27 min) on a single A100 80GB.

## Stack
- **Video models**: Wan2.1-14B (primary) · Wan2.1-1.3B (fast) · HunyuanVideo INT8 (hero scenes) · CogVideoX-5B · LTX-Video · Runway (API fallback)
- **Acceleration**: SageAttention · PAB · TeaCache · FBCache · torch.compile
- **TTS**: Kokoro (local, 82M params)
- **Music**: MusicGen-large (local)
- **Upscale**: Real-ESRGAN (848×480 → 1280×720)
- **Stitch**: FFmpeg (48-thread) + optional 4K upscale
- **API**: FastAPI + Celery + Redis + SQLite
- **Pipeline**: DAG orchestration (Celery canvas), GPUMemoryManager

## Generation Strategies

| Strategy | Models | Resolution | Time (12 clips) |
|----------|--------|------------|-----------------|
| **Fast** | WAN 1.3B | 1280×720 native | ~10 min |
| **Balanced** | WAN 14B | 848×480 + upscale | ~20 min |
| **Quality** | WAN 14B + Hunyuan INT8 hero | 848×480 + upscale | ~25 min |

User selects strategy in the UI dropdown. Each strategy is wired through the full stack.

## Quick Start

### 1. Install
```bash
# PyTorch for A100 (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Flash Attention (optional — SageAttention preferred)
pip install flash-attn --no-build-isolation

# SageAttention (1.5-1.7× faster than FA2)
pip install sageattention

# Project deps
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your API keys
# Add HF_TOKEN for faster first-time local model downloads
```

### 3. Run (CLI — simplest)
```bash
python orchestrator.py \
  --script my_script.txt \
  --style cinematic \
  --quality high \
  --strategy balanced \
  --transition crossfade
```

### 4. Run (Docker — full stack)
```bash
docker-compose up --build
# API available at http://localhost:8000
# Flower dashboard at http://localhost:5555
```

### 5. API Usage
```bash
# Submit job
curl -X POST http://localhost:8000/generate_video \
  -H "Content-Type: application/json" \
  -d '{"script": "...", "style": "cinematic", "strategy": "balanced"}'

# Check status
curl http://localhost:8000/status/{job_id}

# Health check (GPU/Redis/DB/Disk)
curl http://localhost:8000/health

# Download
curl -O http://localhost:8000/download/{job_id}
```

## Azure Deployment

Deploy on an Azure GPU VM using the scripts in `deploy/azure`:

1. Provision VM (PowerShell): `./deploy/azure/create_vm.ps1`
2. SSH to VM and bootstrap: `bash deploy/azure/bootstrap_vm.sh`
3. Configure env (`ANTHROPIC_API_KEY`, `RUNWAY_API_KEY`, `HF_TOKEN`) and start: `docker compose up -d --build`

Full guide: `deploy/azure/README.md`

## Pipeline

```
SCRIPT
  ↓ SceneSplitter    — split into 5–20s scenes with timing
  ↓ PromptEngine     — Claude Haiku converts text → video prompt
  ↓ VideoRouter      — selects model based on strategy + VRAM
  ↓   Wan2.1-14B     — primary (42GB VRAM, 848×480 + upscale)
  ↓   Wan2.1-1.3B    — fast (8GB VRAM, 1280×720 native)
  ↓   HunyuanVideo   — hero scenes (37GB VRAM INT8, quality strategy)
  ↓   CogVideoX-5B   — fallback (24GB VRAM)
  ↓   LTX-Video      — fastest (10GB VRAM, preview)
  ↓   Runway Gen-3   — API fallback
  ↓ Upscaler         — Real-ESRGAN 848×480 → 1280×720
  ↓ Stitcher         — FFmpeg concat + xfade transitions
  ↓ AudioSync        — Kokoro TTS + MusicGen + mix
  ↓ QualityCheck     — detect black/frozen frames
  ↓ FINAL VIDEO      — 1080p MP4 (optional 4K upscale)
```

## Script Format

Plain text (one paragraph = one scene candidate):
```
The sun rises over the ancient city, casting long golden shadows
across the stone streets below.

A lone figure walks through the market, weaving between merchants
and their colorful wares.
```

Or JSON:
```json
[
  {"text": "The sun rises...", "duration": 10},
  {"text": "A lone figure...", "duration": 12}
]
```

Inline hints:
```
[STYLE: documentary] [CAMERA: aerial]
A vast migration of wildebeest crosses the sun-scorched plains.
```

## Output

```
outputs/{job_id}/
  scenes/          individual scene clips
  audio/           narration.wav, music.wav, mixed.aac
  final/           {job_id}.mp4  (or _4k.mp4)
```

## Performance (A100 80GB, 48 cores, with acceleration stack)

| Model | VRAM | Speed (5s clip) | Quality | Acceleration |
|---|---|---|---|---|
| Wan2.1-14B | ~42GB | ~90s | ★★★★★ | SageAttn + TeaCache + FBCache + compile |
| Wan2.1-1.3B | ~8GB | ~20s | ★★★ | SageAttn + TeaCache + compile |
| HunyuanVideo INT8 | ~37GB | ~3 min | ★★★★★ | bitsandbytes INT8 quantization |
| CogVideoX-5B | ~24GB | ~1.5 min | ★★★★ | standard |
| LTX-Video | ~10GB | ~15s | ★★★ | standard |

12-clip balanced video ≈ 20 min end-to-end (including TTS, music, stitching).
