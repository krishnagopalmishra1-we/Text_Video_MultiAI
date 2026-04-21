# CLAUDE.md — Memory & Working Context

## Standing Rule: Think Holistically
**Always trace the full system impact before any change.** No short-sighted fixes.
Before touching config, pipeline, or model code: follow the chain through router → local_runner → model → strategy → stitcher. Update CLAUDE.md and memory when architectural decisions are made. Never propose a change without reading the affected code.

## Standing Rule: No Guesswork
**Never guess, assume, or speculate about errors, behavior, or state.** Read the actual logs, source code, or live system before drawing conclusions. If the answer requires a tool call (log read, source inspect, live test), make that call first. Do not fabricate explanations that sound plausible — only report what the evidence shows. If evidence is insufficient, say so explicitly and fetch more before proceeding.

## Project
Long-form AI video generator. A100 80GB + 48 CPU cores.
Three generation strategies: fast / balanced / quality.
**Speed target: 1-minute video in ≤ 15 minutes end-to-end.**

## Hardware Context
- GPU: NVIDIA A100 80GB (NCads A100 V4 — Azure)
- CPU: 48 cores
- Primary precision: bfloat16 (native A100)
- Attention: SageAttention preferred (1.5–1.7× faster than FlashAttn2), FA2 as fallback

## Model Stack (in priority order)
| Priority | Model | VRAM | Use case | Acceleration |
|---|---|---|---|---|
| 1 | Wan2.1-T2V-14B | ~42GB | Default (balanced/quality) | PyTorch SDPA (FlashAttn2 auto on A100). SageAttn ENABLED — SIGSEGV fixed by PyTorch 2.5.1+cu124 |
| 2 | Wan2.1-T2V-1.3B | ~8GB | Not used — produces blurry/low quality output | — |
| 3 | HunyuanVideo INT8 | ~37GB | Hero/cinematic scenes (quality only) | bitsandbytes INT8 quantization |
| 4 | LTX-Video | ~10GB | Preview/fastest (not yet wired) | standard |
| 5 | Runway Gen-3 | API | Final fallback (only implemented API) | — |

**Removed**: CogVideoX-5B — failed smoke test 2026-04-12 with `Illegal header value b'Bearer '`
(empty HF_TOKEN in Docker worker). Also lower quality than WAN14B at similar VRAM. See
`docs/smoke_test_2026_04_12.md` for full postmortem.

## Generation Strategies
- **Fast**: WAN 1.3B — DO NOT USE. Produces blurry low-quality output. Smoke test confirmed.
- **Balanced**: WAN 14B, 848×480 + Real-ESRGAN upscale to 1280×720, 15 steps, PAB+TeaCache+FBCache — target ~10-12 min for 1-min video
- **Quality**: WAN 14B bulk + Hunyuan INT8 hero scenes, 848×480 + upscale, 15/30 steps, ~20-22 min

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
- **Audio parallelism** — orchestrator starts audio in background thread after prompt gen; joins before stitch. Audio runs on CPU while GPU does video. Zero time cost.
- **WAN 1.3B retired** — do not route production traffic to wan2_1b; quality is unacceptable (blurry output confirmed in smoke test)
- **Text encoder offload** — moved to CPU after encoding to free VRAM for generation

## Strict Rules When Modifying
- Must enable sage_attention — SIGSEGV fixed with PyTorch 2.5.1 + CUDA 12.4.1 + SageAttention 2.2.0.
- Wan2 acceleration order: SageAttn → FasterCache → FBCache → compile (PAB removed — conflicts with FasterCache on diffusers 0.37+)
- FasterCache `is_guidance_distilled` must be `False` for WAN 14B (standard CFG, not distilled)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True required in docker-compose.yml + run scripts (PAB+FasterCache+FBCache reserved-memory fragmentation fix)
- Never add model CPU offload without updating `_VRAM_THRESHOLDS` in local_runner.py
- Never change `worker_prefetch_multiplier` from 1 on GPU worker (causes OOM)
- FFmpeg threads must be passed as string, not int: `str(self.cpu_threads)`
- All model `.generate()` methods must accept `output_path` and return `Path`
- Scene IDs are 1-indexed and used as filenames: `scene_0001.mp4`
- Model key for WAN is `wan2_14b` or `wan2_1b` in quality_presets, not plain `wan2`
- Do not re-add CogVideoX without fixing HF_TOKEN in docker-compose.yml worker_gpu env

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
- **HunyuanVideo OOM**: uses INT8 quantization (~37GB); if still OOM, fallback to Runway API
- **SageAttention not found**: falls back to FlashAttn2; install with `pip install sageattention`
- **Slow FFmpeg**: verify `--threads 48` is being passed; check `cpu_threads` in Stitcher
- **Celery not picking up GPU tasks**: ensure worker started with `--queues=gpu`
- **Kokoro not installed**: `pip install kokoro>=0.9.4` — requires espeak-ng on Linux
- **Strategy not applied**: check quality_presets keys in model_config.yaml use `wan2_14b`/`wan2_1b` not `wan2`
- **Hunyuan slow on fast strategy**: Hunyuan is designed for hero scenes in quality strategy; using it with fast strategy produces ~57 min/run. Use wan2_1b for fast.
- **scene.model_used always NULL**: known issue — tasks.py does not write model name back to Scene row

## Smoke Test History
- **2026-04-12**: First full 12-run smoke test on Azure A100. 9/12 passed (all WAN + Hunyuan).
  CogVideoX 3/3 failed (HF_TOKEN empty in Docker env). CogVideoX removed from stack.
  See `docs/smoke_test_2026_04_12.md` for full results and outstanding issues.
- **2026-04-16**: WAN 14B balanced smoke test (sushi chef prompt, 848×480, 15 steps, PAB+TeaCache+FBCache).
  **Result: 0-second output, blurry, no detail. FAILED.** Speed also unacceptable: ~4 min/clip → ~48 min for 1-min video (target is 15–20 min — 3× over).
  **Root cause not yet diagnosed** — may be export_to_video bug, WAV frame count issue, or upscaler returning empty frames.
  **Decision**: Pivot to anime/animation-style video as primary use case (see Animation Pivot section below).

## Speed Problem Analysis (Real-video path)
- WAN 14B @ 848×480, 15 steps: ~4 min/clip
- 12 clips for 1-min video = ~48 min. **3× over the 20-min target.**
- TeaCache/FBCache working (step drops 22s→10s after warmup) but only ~25% speedup total
- To hit 20-min target with WAN 14B: need ≤100s/clip → requires ~5–6 steps (unacceptable quality)
- **Only viable real-video path**: Hunyuan INT8 at 15–20 steps (test pending)

## Animation Pivot — Plan for Next Session
**Decision 2026-04-16**: Anime/animation-style video is the preferred use case.
Reasons: better storytelling, consistent visual style, smaller/faster models, no uncanny valley.

### Target: 1-min anime/animation video in ≤20 min on A100 80GB

### Models to Evaluate (priority order)
| Model | VRAM | Speed estimate | Style |
|---|---|---|---|
| WAN 14B + anime LoRA | ~42GB | ~4 min/clip | anime/cinematic |
| WAN 1.3B + anime LoRA | ~8GB | ~50s/clip | anime (need quality check) |
| CogVideoX-5B (anime checkpoint) | ~20GB | ~2–3 min/clip | anime |
| AnimateDiff v3 (SDXL) | ~12GB | ~30–60s/clip | anime/stylized |
| Mochi-1 | ~22GB | ~2 min/clip | stylized |

### Animation Style Requirements
1. **Anime / Studio Ghibli style** — soft lines, painted BGs, expressive characters
2. **2D animated** — flat color, motion comics, illustrated characters
3. **3D animated** — Pixar-style, cel-shaded
4. **Motion comic** — manga-panel style with subtle animation
5. **Cinematic anime** — Makoto Shinkai / AoT style: detailed, filmic

### Prompt Engineering for Animation
- Add style tokens: `anime style, Studio Ghibli, cel-shaded, 2D animation, hand-drawn`
- Avoid: `photorealistic, live action, real person`
- Negative: add `photorealistic, live action, 3D render, CGI` when targeting 2D anime
- For Ghibli: `soft pastel colors, watercolor background, gentle animation`
- For action anime: `dynamic motion lines, expressive faces, vibrant colors, dramatic lighting`

### Next Session Action Items
1. Diagnose 0-second output bug in current WAN 14B run (check `export_to_video`, frame count, upscaler)
2. Test WAN 14B with anime-style prompt at 848×480, 15 steps — does quality improve?
3. Test WAN 1.3B with anime LoRA (smaller model, faster — acceptable if anime style hides low-res)
4. Research and test CogVideoX anime checkpoints (may need HF_TOKEN fix first)
5. Evaluate AnimateDiff — fastest option but different pipeline
6. Pick best model, set as default strategy for animation content
7. Add `animation_style` parameter to scene prompts and strategy config
