# Long Video Generator

AI-powered long-form video generation (10–20 min) on a single A100 80GB.

## Stack
- **Video models**: Wan2.1-14B → HunyuanVideo → CogVideoX-5B → LTX-Video → Runway/Pika/Veo/Higgsfield
- **TTS**: Kokoro (local) + ElevenLabs fallback
- **Music**: MusicGen-large (local)
- **Stitch**: FFmpeg (48-thread) + optional Real-ESRGAN 4K upscale
- **API**: FastAPI + Celery + Redis + SQLite

## Quick Start

### 1. Install
```bash
# PyTorch for A100 (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Flash Attention
pip install flash-attn --no-build-isolation

# Project deps
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run (CLI — simplest)
```bash
python orchestrator.py \
  --script my_script.txt \
  --style cinematic \
  --quality high \
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
  -d '{"script": "...", "style": "cinematic", "quality": "high"}'

# Check status
curl http://localhost:8000/status/{job_id}

# Download
curl -O http://localhost:8000/download/{job_id}
```

## Azure Deployment

Deploy on an Azure GPU VM using the scripts in `deploy/azure`:

1. Provision VM (PowerShell): `./deploy/azure/create_vm.ps1`
2. SSH to VM and bootstrap: `bash deploy/azure/bootstrap_vm.sh`
3. Configure env and start: `docker compose up -d --build`

Full guide: `deploy/azure/README.md`

## Pipeline

```
SCRIPT
  ↓ SceneSplitter    — split into 5–20s scenes with timing
  ↓ PromptEngine     — Claude Haiku converts text → video prompt
  ↓ VideoRouter      — selects model based on VRAM + quality tier
  ↓   Wan2.1-14B     — primary (40GB VRAM, 1280×720)
  ↓   HunyuanVideo   — hero scenes (60GB VRAM)
  ↓   CogVideoX-5B   — fast (24GB VRAM)
  ↓   LTX-Video      — fastest (10GB VRAM)
  ↓   API fallback   — Runway / Pika / Veo / Higgsfield
  ↓ Stitcher         — FFmpeg concat + xfade transitions
  ↓ AudioSync        — Kokoro TTS + MusicGen + mix
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

## Performance (A100 80GB, 48 cores)

| Model | VRAM | Speed (10s clip) | Quality |
|---|---|---|---|
| Wan2.1-14B | ~40GB | ~3 min | ★★★★★ |
| HunyuanVideo | ~60GB | ~5 min | ★★★★★ |
| CogVideoX-5B | ~24GB | ~1.5 min | ★★★★ |
| LTX-Video | ~10GB | ~30s | ★★★ |

20-minute video @ 10s scenes ≈ 120 scenes ≈ 6h on Wan2.1-14B
