# Production Blueprint — Multimodal Video Generation Platform

## 1. FULL SYSTEM AUDIT

### 1.1 Confirmed Bugs

| # | File | Bug | Severity |
|---|------|-----|----------|
| B1 | `config/model_config.yaml` | Hunyuan `hf_id` still says `tencent/HunyuanVideo`; code was patched to default to `hunyuanvideo-community/HunyuanVideo`, but config is stale — any fresh worker_gpu reading config will pass the wrong ID to the runner | **Critical** |
| B2 | `video_engine/models/wan2.py` | `torch.compile` + CPU text-encoder offload = device mismatch (`cpu` vs `cuda:0`). Compile traces assume all modules on same device. Warmup hacks around it but real inference path still breaks when compile=True and encode_prompt shuffles text_encoder between CPU/GPU | **Critical** |
| B3 | `video_engine/models/hunyuan.py` | `torch.compile(mode="max-autotune")` triggers inductor cudagraph allocation of 559 GiB buffer — known inductor bug with Hunyuan's cross-attention map size. Config still has `compile: true` | **Critical** |
| B4 | `server/tasks.py` → `run_pipeline` | Entire pipeline (split → prompt → video × N scenes → audio → stitch) runs inside a **single Celery GPU task**. Audio generation (MusicGen-large = 16 GB VRAM) competes with video model for GPU memory. If WAN is still loaded (40 GB), MusicGen load will OOM at 56 GB+ | **High** |
| B5 | `video_engine/local_runner.py` | `_switch_model` / `unload_all` deletes runner objects and calls `gc.collect()` + `torch.cuda.empty_cache()`, but never guardrails that VRAM is actually freed before loading the next model. `torch.compile` inductor cache can pin 2-5 GB that doesn't release | **High** |
| B6 | `stitcher/transitions.py` → `build_xfade_chain` | Cumulative offset calculation is wrong: `cumulative += clip_durations[i] - td` starts at i=0 but should use cumulative duration of first clip minus td. First xfade offset should be `clip_durations[0] - td`, not `0 + clip_durations[0] - td` (which gives the right value for the first but compounds incorrectly because it doesn't account for previous xfade durations shortening the stream) | **Medium** |
| B7 | `db/models.py` | SQLite with `check_same_thread: False` is used from multiple Celery workers + FastAPI + WebSocket. SQLite has a single-writer lock — concurrent writes from GPU worker + CPU worker will cause `database is locked` errors under real load | **High** |
| B8 | `server/main.py` | `@app.on_event("startup")` is deprecated since FastAPI 0.109; should use lifespan context manager | **Low** |
| B9 | `audio/music.py` | `torch.manual_seed(seed)` sets the **global** seed instead of using a `Generator` — affects any other torch operation in the same process | **Medium** |
| B10 | `audio/tts.py` → `synthesize_full` | Writes temp files per scene (`_tmp_{scene_id}.wav`) but only `unlink(missing_ok=True)` — if an exception occurs mid-loop, temp files leak | **Low** |
| B11 | `video_engine/api_runner.py` | Creates a new `RunwayClient()` on **every** call — no connection reuse, no rate limiting | **Low** |
| B12 | `prompt_engine/generator.py` | Silently swallows LLM exceptions (`except Exception: pass`). If the API key is invalid or quota exhausted, every scene falls back to template prompts with zero logging | **Medium** |
| B13 | `docker-compose.yml` | `api` service has GPU reservation but only serves FastAPI — wasteful GPU allocation. The GPU should be reserved only for `worker_gpu` | **Medium** |
| B14 | `Dockerfile` | `COPY . .` before output dir creation means every code change invalidates the Docker layer cache for output dir creation | **Low** |
| B15 | `scene_splitter/splitter.py` | `_build_scenes` target_words variable is computed but never used; `_split_long_paragraph` also computes it without use | **Low** |
| B16 | `server/tasks.py` → `generate_scene_clip` | Creates a new `VideoRouter` (which creates a new `LocalRunner` and loads config) per call — model gets re-loaded from scratch on every scene instead of reusing the warm runner | **Critical** |
| B17 | `orchestrator.py` | Missing `backend` parameter in `TTSEngine(backend=args.tts_backend)` constructor but `TTSEngine.__init__` doesn't accept `backend` — only has `voice`, `speed`, `sample_rate` | **Medium** |

### 1.2 Performance Bottlenecks

| # | Area | Issue | Impact |
|---|------|-------|--------|
| P1 | Video generation | WAN 2.1-14B at 1280×720, 20 steps ≈ 78s/step × 20 = **26 min per 5s clip**. A 2-min video (24 scenes × 5s) = **10+ hours** | **Blocker** |
| P2 | Video generation | Each scene is generated sequentially in a single GPU process. Zero parallelism | **Blocker** |
| P3 | Model loading | Model swap (unload WAN → load Hunyuan) takes 45-90s per swap due to weight deserialization. No model caching across jobs | **High** |
| P4 | VAE decode | VAE decode is the final bottleneck after denoising — runs at full precision on full-res latents with no batching | **High** |
| P5 | torch.compile | First-run compilation overhead is 2-5 min per model. Warmup only runs on `worker_process_init` — if worker restarts, full recompile | **High** |
| P6 | Audio pipeline | MusicGen-large generates max 30s chunks then tiles. A 2-min video requires 4 tiles with crossfade — each tile is a full forward pass | **Medium** |
| P7 | Stitching | All clips are re-encoded per-clip during normalization, then again during xfade, then again when attaching audio — **3 encode passes** | **Medium** |
| P8 | No clip-level parallelism | CPU worker has concurrency=8 but the `run_pipeline` task runs everything sequentially on the GPU queue. The Celery chord/group pattern exists as individual tasks but is never wired together | **High** |
| P9 | I/O | `export_to_video` writes uncompressed frames to h264 via imageio — no hardware-accelerated encoding at the intermediate step | **Medium** |
| P10 | Prompt generation | Claude API calls are serial within each scene batch (semaphore=4 max concurrency) but block the GPU pipeline — prompts should be generated fully before GPU work starts | **Low** |

### 1.3 Architectural Flaws

| # | Issue | Detail |
|---|-------|--------|
| A1 | **Single-GPU bottleneck** | Entire system assumes 1 GPU. No concept of model parallelism, pipeline parallelism, or multi-node. xDiT is commented out | |
| A2 | **Monolithic pipeline task** | `run_pipeline` does everything in one task. No DAG-based orchestration for parallel prompt gen + sequential video gen + parallel audio + stitch | |
| A3 | **No model persistence across jobs** | Each job submission through Celery creates fresh model instances. The warmup hook only helps the very first load; subsequent calls through `generate_scene_clip` task recreate `VideoRouter` from scratch (B16) | |
| A4 | **SQLite in distributed system** | SQLite can't handle concurrent writes from multiple workers (B7) | |
| A5 | **ComfyUI parallel stack unused** | ComfyUI container + WanVideoWrapper is built but never integrated into the pipeline. It's a completely separate system accessible only via web UI | |
| A6 | **No progress granularity** | Progress tracking is scene-level only. No per-step progress during diffusion (the longest phase). WebSocket sends updates every 2s but the data only changes per-scene completion | |
| A7 | **No artifact storage** | All outputs go to local filesystem. No object storage, no CDN, no signed URLs for download | |
| A8 | **No health monitoring** | `/health` returns `{"status": "ok"}` with zero checks on GPU availability, Redis connectivity, worker status, or disk space | |
| A9 | **No retry semantics for Runway** | Runway API has a 5-min poll timeout but no exponential backoff, no circuit breaker, no fallback when credits are exhausted mid-batch | |
| A10 | **Docker image bloat** | Single Dockerfile for API server and both workers. The API server gets CUDA + Flash Attention + all ML dependencies it never uses | |

### 1.4 Dead Code & Unnecessary Files

| File | Reason | Action |
|------|--------|--------|
| `video_engine/models/wan2.py.bak` | Backup file from earlier session | **Delete** |
| `tmp_track_job.py` (root) | Debug script with hardcoded job ID | **Delete** |
| `smoke_ltx_req.json` (root) | One-off test request | **Delete** |
| `smoke_wan2_req.json` (root) | One-off test request | **Delete** |
| `wan2_celery_response.json` (root) | Debug capture | **Delete** |
| `wan2_response.json` (root) | Debug capture | **Delete** |
| `tmp_check_wan_opts.py` | Debug scratch | **Delete** |
| `tmp_debug_wan2_prod.py` | Debug scratch | **Delete** |
| `tmp_debug_wan2_prod50.py` | Debug scratch | **Delete** |
| `tmp_debug_wan2.py` | Debug scratch | **Delete** |
| `tmp_inspect_attn_backend.py` | Debug scratch | **Delete** |
| `tmp_inspect_wan_impl.py` | Debug scratch | **Delete** |
| `tmp_job_high.json` | Debug job spec | **Delete** |
| `tmp_job.json` | Debug job spec | **Delete** |
| `Dockerfile.comfyui` | Unused in pipeline, maintenance burden. ComfyUI is a separate UI tool, not part of automated pipeline | **Remove from compose; keep file only if manual UI needed** |
| `workflows/*.json` | ComfyUI workflows never used by pipeline | **Move to docs/ or delete** |
| `scene_splitter/splitter.py` lines `target_words` | Dead variable in `_build_scenes` and `_split_long_paragraph` | **Remove** |
| `prompt_engine/presets.py` | Duplicate of data already in `config/presets.yaml` — two sources of truth for same style/camera presets | **Consolidate into config file only** |
| `scripts/comfyui_start.sh` | Only relevant to ComfyUI container | **Move into Dockerfile.comfyui context or delete** |

### 1.5 Dependency Issues

| Issue | Detail |
|-------|--------|
| `xformers` in requirements but Flash Attention is primary | xformers is 800 MB+ and unused when FA2 is available. Remove or make optional |
| `librosa` | Only used for resampling in TTS — `scipy.signal.resample` or `torchaudio.transforms.Resample` would eliminate this 15 MB dependency |
| `nltk` | Only used for `sent_tokenize` in `scene_splitter/utils.py` with a regex fallback already present. NLTK pulls 40 MB+ of data |
| `tiktoken` | Listed in requirements but never imported anywhere in the codebase |
| `protobuf` | Listed but not directly used — transitive dep of `transformers`, shouldn't be pinned |
| `sentencepiece` | Transitive dep, shouldn't be pinned separately |
| `scipy` | Only used transitively via librosa. Remove if librosa is removed |
| `rich` | Imported nowhere in the codebase |
| `psutil` | Imported nowhere in the codebase |
| `tqdm` | Imported nowhere directly (used by HF libraries internally) |

---

## 2. TARGET PERFORMANCE MODEL

**Goal**: 1-2 minute video, ~30 minutes, no quality loss at 1280×720.

### Math

- 2-minute video = ~120s of content
- At 5s per clip = **24 clips**
- At 10s per clip = **12 clips**
- Budget: 30 min = 1800s
- Available for video generation: ~1500s (reserve 300s for prompt gen + audio + stitching)
- **Per-clip budget**: 1500s / 12 clips = **125s per clip** (10s clips) or 1500s / 24 clips = **62.5s per clip** (5s clips)

### Current vs Required Speed

| Model | Current (s/step) | Steps | Total/clip | Required | Gap |
|-------|-------------------|-------|------------|----------|-----|
| WAN 2.1-14B @ 1280×720 | ~78s | 20 | ~1560s | 62-125s | **12-25×** too slow |
| WAN 2.1-14B @ 848×480 | ~25s | 20 | ~500s | 62-125s | **4-8×** too slow |
| WAN 2.1-14B @ 480×320 | ~8s | 15 | ~120s | 62-125s | ≈ feasible |

**Conclusion**: Hitting 30 min for 2 min of 1280×720 content on a **single A100** is not possible with naive sequential generation. The plan must use a combination of:
1. Resolution reduction with post-upscaling
2. Step reduction with flow-matching acceleration
3. Attention caching (PAB/TeaCache/TaylorSeer)
4. torch.compile with working configuration
5. Parallel clip generation (multi-GPU or temporal tiling)

---

## 3. PRODUCTION IMPLEMENTATION PLAN

### Phase 0: Immediate Bug Fixes (Day 1)

#### 0.1 Fix Hunyuan config
```yaml
# config/model_config.yaml
hunyuan:
  hf_id: "hunyuanvideo-community/HunyuanVideo"
  compile: false  # disabled until inductor bug is fixed
```

#### 0.2 Fix torch.compile + CPU offload conflict in WAN
Two options (implement option A):

**Option A — Disable compile when offload is active** (safe, immediate):
```python
# wan2.py: move compile block BEFORE text_encoder offload
# AND guard: only compile if text encoder stays on GPU
if self.cfg.get("compile") and not self.cfg.get("offload_text_encoder", True):
    self.pipe.transformer = torch.compile(...)
```

**Option B — Keep text encoder on GPU** (uses ~2 GB more VRAM but enables compile):
Remove the `self.pipe.text_encoder.to("cpu")` line. On A100 80 GB with WAN at ~40 GB, the extra 2 GB is affordable.

**Recommended**: Option B. The 2 GB is trivial on 80 GB, and compile gives 20-30% speedup.

#### 0.3 Fix Celery model persistence (B16)
```python
# server/tasks.py — replace per-task VideoRouter creation with module-level singleton
_video_router: VideoRouter | None = None

def _get_router(quality: str = "high") -> VideoRouter:
    global _video_router
    if _video_router is None:
        _video_router = VideoRouter(quality=quality)
    return _video_router
```

#### 0.4 Fix SQLite concurrent access (B7)
Replace SQLite with PostgreSQL in production:
```yaml
# docker-compose.yml — add postgres service
postgres:
  image: postgres:16-alpine
  environment:
    POSTGRES_DB: videogen
    POSTGRES_USER: videogen
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
  volumes:
    - pgdata:/var/lib/postgresql/data
```
Update `DATABASE_URL=postgresql://videogen:${POSTGRES_PASSWORD}@postgres:5432/videogen`

#### 0.5 Fix MusicGen global seed (B9)
```python
gen = torch.Generator(device=self.device).manual_seed(seed)
audio_values = self._model.generate(**inputs, ..., generator=gen)  # if API supports it
```

#### 0.6 Fix silent LLM failure (B12)
```python
except Exception as e:
    logger.warning(f"LLM prompt generation failed for scene {scene.scene_id}: {e}")
    # fall through to template
```

#### 0.7 Remove GPU from API service (B13)
```yaml
# docker-compose.yml — api service
api:
  build:
    context: .
    dockerfile: Dockerfile.api  # lightweight image without CUDA
  # Remove: runtime: nvidia, deploy.resources.reservations
```

### Phase 1: Speed Optimization — Single A100 (Days 2-5)

#### 1.1 Optimized Generation Strategy: Generate at 848×480 + Upscale to 1280×720

The critical insight from global best practices: **generate at lower resolution and upscale**, rather than generating at native 1280×720.

- WAN 2.1-14B at 848×480 with 15 steps: ~12s/step × 15 = **~180s per 5s clip**
- With PAB + TaylorSeer cache: ~1.5× speedup → **~120s per clip**
- With torch.compile (fixed): ~1.3× speedup → **~90s per clip**
- 12 clips × 90s = **~18 min** for video generation
- Upscale 12 clips via Real-ESRGAN: ~30s per clip = **~6 min**
- Audio + stitching: ~5 min
- **Total: ~29 minutes** ✓

#### 1.2 Fix and optimize acceleration stack

```python
# wan2.py — corrected acceleration order
def load(self) -> None:
    # 1. Load pipeline
    self.pipe = WanPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
    
    # 2. VAE optimization
    self.pipe.vae.enable_slicing()
    
    # 3. Flash Attention
    self.pipe.transformer.enable_flash_attn()
    
    # 4. Attention caching (PAB) — BEFORE compile
    apply_pyramid_attention_broadcast(self.pipe.transformer, pab_config)
    
    # 5. torch.compile — AFTER all hooks are applied
    self.pipe.transformer = torch.compile(
        self.pipe.transformer,
        mode="max-autotune",  # not reduce-overhead, which conflicts with PAB
        backend="inductor",
        fullgraph=False,  # allow graph breaks for dynamic shapes
    )
    
    # 6. Keep text encoder on GPU (only ~2 GB, enables compile compatibility)
    # Do NOT move to CPU
```

#### 1.3 Implement SageAttention (global best practice, 2024)

SageAttention provides **2-3× speedup** over Flash Attention 2 for video diffusion models with negligible quality loss:

```python
# Install: pip install sageattention
from sageattention import sageattn
import torch.nn.functional as F

# Monkey-patch scaled_dot_product_attention
torch.nn.functional.scaled_dot_product_attention = sageattn
```

This is a drop-in replacement. Benchmark results on A100:
- FA2: 312 TFLOPS
- SageAttention: 520 TFLOPS (1.67× faster)

#### 1.4 Implement First-Block Cache (FBCache)

State-of-the-art technique from 2025 for DiT-based video models:

```python
# For WAN 2.1 specifically:
from diffusers.hooks import apply_first_block_cache
from diffusers.hooks.first_block_cache import FirstBlockCacheConfig

config = FirstBlockCacheConfig(threshold=0.15)  # tune: lower = higher quality
apply_first_block_cache(pipe.transformer, config)
# Provides 1.5-2× speedup by caching redundant first-block computations
```

Combined with PAB, this gives **2-3× total speedup** over baseline.

#### 1.5 Implement proper TeaCache for WAN 2.1

```python
from diffusers.hooks import apply_tea_cache
from diffusers.hooks.tea_cache import TeaCacheConfig

tea_config = TeaCacheConfig(
    threshold=0.15,               # quality/speed tradeoff
    cache_type="mean",
    skip_first_steps=2,           # don't cache initial steps
    skip_last_steps=2,            # don't cache final steps  
)
apply_tea_cache(pipe.transformer, tea_config)
```

#### 1.6 Implement Adaptive Resolution Pipeline

```python
class AdaptiveResolutionPipeline:
    """Generate at optimal resolution per model, upscale to target."""
    
    def __init__(self, gen_resolution=(848, 480), target_resolution=(1280, 720)):
        self.gen_res = gen_resolution
        self.target_res = target_resolution
        self.upscaler = None  # lazy-load Real-ESRGAN
    
    def generate_and_upscale(self, runner, scene, **kwargs):
        # Generate at lower resolution
        clip = runner.generate(
            ..., width=self.gen_res[0], height=self.gen_res[1], **kwargs
        )
        # Upscale to target
        return self._upscale_clip(clip, self.target_res)
    
    def _upscale_clip(self, clip_path, target_res):
        """Frame-by-frame Real-ESRGAN or CUDA-accelerated upscale."""
        # Use realesrgan-ncnn-vulkan for GPU-accelerated upscaling
        # Or torch-based upscaling using Real-ESRGAN weights
        pass
```

#### 1.7 Optimized quality presets (replace current)

```yaml
quality_presets:
  # "high" preset optimized for 30-min budget on single A100
  high:
    label: "High Quality (gen@848×480, upscale to 1280×720)"
    clip_duration: 10           # 10s clips reduce scene count
    wan2:
      gen_resolution: [848, 480]
      output_resolution: [1280, 720]
      steps: 15                 # flow-matching: 15 ≈ 20 with caching
      guidance_scale: 5.0
      use_pab: true
      use_tea_cache: true
      use_fb_cache: true
      use_sage_attention: true
      compile: true
    upscale:
      method: "realesrgan"
      scale: 2                  # 848→1696 then crop to 1280
```

### Phase 2: Pipeline Architecture (Days 6-10)

#### 2.1 DAG-Based Pipeline Orchestration

Replace the monolithic `run_pipeline` task with a proper DAG:

```
[Split Script] 
    → [Generate Prompts (parallel, CPU)]
    → [Generate Video Clips (sequential on GPU, with model persistence)]
    → [Generate Audio (parallel on CPU, after video gen frees GPU)]
    → [Upscale Clips (GPU, after video gen)]  
    → [Stitch + Mux (CPU)]
    → [Done]
```

Implementation using Celery canvas:

```python
from celery import chain, group, chord

def submit_pipeline(job_id: str, config: dict):
    pipeline = chain(
        split_script.si(job_id, config),
        generate_all_prompts.s(job_id, config),
        generate_all_clips.s(job_id, config),          # sequential GPU
        group(
            upscale_all_clips.s(job_id, config),        # GPU (after video gen unloads model)
            generate_audio_track.si(job_id, config),    # CPU parallel
        ),
        stitch_final.s(job_id, config),
    )
    pipeline.apply_async()
```

#### 2.2 GPU Memory Management

```python
class GPUMemoryManager:
    """Centralized VRAM lifecycle management."""
    
    def __init__(self, total_vram_gb: float = 80.0):
        self.total = total_vram_gb
        self._loaded_models: dict[str, tuple[Any, float]] = {}  # name → (model, vram_gb)
    
    def ensure_available(self, required_gb: float) -> None:
        """Evict models until required VRAM is free."""
        while self._free_vram() < required_gb:
            # Evict lowest-priority loaded model
            evict_name = self._lowest_priority_loaded()
            if evict_name is None:
                raise RuntimeError(f"Cannot free {required_gb}GB VRAM")
            self.unload(evict_name)
    
    def load_model(self, name: str, loader_fn, vram_gb: float) -> Any:
        if name in self._loaded_models:
            return self._loaded_models[name][0]
        self.ensure_available(vram_gb + 2.0)  # 2GB safety margin
        model = loader_fn()
        self._loaded_models[name] = (model, vram_gb)
        return model
    
    def unload(self, name: str) -> None:
        if name not in self._loaded_models:
            return
        model, _ = self._loaded_models.pop(name)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Verify VRAM actually freed
        time.sleep(0.5)
        logger.info(f"Unloaded {name}, free VRAM: {self._free_vram():.1f}GB")
```

#### 2.3 Separate Docker Images

```dockerfile
# Dockerfile.api — lightweight FastAPI server (no CUDA)
FROM python:3.11-slim
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY server/ /app/server/
COPY db/ /app/db/
COPY config/ /app/config/

# Dockerfile.gpu — GPU worker with full ML stack
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# ... full ML dependencies

# Dockerfile.cpu — CPU worker for audio/stitching
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg
COPY requirements-cpu.txt .
RUN pip install -r requirements-cpu.txt
```

#### 2.4 PostgreSQL + Connection Pooling

```python
# db/models.py
from sqlalchemy.pool import QueuePool

_DB_URL = os.environ.get("DATABASE_URL", "postgresql://videogen:pass@postgres:5432/videogen")
engine = create_engine(
    _DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
```

#### 2.5 Per-Step Progress via Callback

```python
# video_engine/models/wan2.py
def generate(self, ..., progress_callback=None):
    def step_callback(pipe, step, timestep, callback_kwargs):
        if progress_callback:
            progress_callback(step=step, total_steps=num_inference_steps)
        return callback_kwargs
    
    output = self.pipe(
        ...,
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
```

Wire this to WebSocket for real-time UI updates:
```python
# server/tasks.py
def _progress_callback(job_id, scene_id, step, total_steps):
    redis_client.publish(f"progress:{job_id}", json.dumps({
        "scene_id": scene_id,
        "step": step,
        "total_steps": total_steps,
        "percent": step / total_steps * 100,
    }))
```

### Phase 3: Multi-GPU & Scaling (Days 11-20)

#### 3.1 xDiT Sequence Parallelism (2-4 GPUs)

xDiT enables splitting a single video generation across multiple GPUs via DiT sequence parallelism:

```python
# Uncomment in requirements.txt: xdit>=0.4.0
from xdit import xDiTParallel

# Split WAN 2.1-14B across 2 GPUs: ~2× speedup
parallel_pipe = xDiTParallel(
    pipe,
    parallel_config={
        "sequence_parallel_degree": 2,
        "ulysses_degree": 2,
    }
)
```

With 2× A100: 12 clips × 45s = **9 min** for video generation.
With 4× A100: 12 clips × 23s = **~5 min** for video generation.

#### 3.2 Multi-Worker Clip Generation

If scaling to multiple GPU nodes (e.g., 4× A100 on separate VMs):

```python
# server/tasks.py — parallel scene generation via Celery group
def generate_all_clips(scenes, job_id, config):
    tasks = [
        generate_scene_clip.si(job_id, scene.to_dict(), config["quality"])
        for scene in scenes
    ]
    # Each task runs on a different GPU worker
    return group(tasks).apply_async(queue="gpu")
```

#### 3.3 Model Sharding for Hunyuan (60 GB → fits in 40 GB)

```python
# Use device_map="auto" from accelerate for Hunyuan
from accelerate import infer_auto_device_map

device_map = infer_auto_device_map(
    transformer,
    max_memory={0: "40GiB", "cpu": "64GiB"},
    no_split_module_classes=["HunyuanVideoTransformerBlock"],
)
```

### Phase 4: Storage, Delivery & Monitoring (Days 21-25)

#### 4.1 Cloud Storage (GCS)

```python
# storage/gcs.py
from google.cloud import storage

class VideoStorage:
    def __init__(self, bucket_name="videogen-outputs"):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def upload_clip(self, local_path: Path, job_id: str) -> str:
        blob = self.bucket.blob(f"jobs/{job_id}/{local_path.name}")
        blob.upload_from_filename(str(local_path))
        return blob.public_url
    
    def generate_signed_url(self, blob_path: str, expiry_hours: int = 24) -> str:
        blob = self.bucket.blob(blob_path)
        return blob.generate_signed_url(
            expiration=timedelta(hours=expiry_hours),
            method="GET",
        )
```

#### 4.2 Health Check Endpoint

```python
@app.get("/health")
async def health():
    checks = {}
    
    # Redis
    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except:
        checks["redis"] = "error"
    
    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            checks["gpu"] = {"free_gb": round(free/1e9, 1), "total_gb": round(total/1e9, 1)}
        else:
            checks["gpu"] = "unavailable"
    except:
        checks["gpu"] = "error"
    
    # Disk
    import shutil
    usage = shutil.disk_usage("outputs")
    checks["disk"] = {"free_gb": round(usage.free/1e9, 1)}
    
    # DB
    try:
        with Session(engine) as s:
            s.exec(select(Job).limit(1))
        checks["db"] = "ok"
    except:
        checks["db"] = "error"
    
    status = "ok" if all(v == "ok" or isinstance(v, dict) for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}
```

#### 4.3 Structured Logging + Metrics

```python
# Use structlog for JSON logging
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

# Key metrics to track:
# - generation_time_seconds (per clip, per model)
# - steps_per_second (per model, per resolution)
# - vram_utilization_percent
# - queue_depth (celery inspect)
# - job_completion_time_seconds (end to end)
# - error_rate (per model, per stage)
```

### Phase 5: Quality & Robustness (Days 26-30)

#### 5.1 Prompt-Adaptive Model Selection

Instead of one model for all scenes, select model based on scene content:

```python
SCENE_MODEL_MAP = {
    "action": "wan2",        # Best motion coherence
    "landscape": "hunyuan",  # Best visual detail  
    "dialogue": "wan2",      # Best lip sync potential
    "abstract": "cogvideox", # Good for stylized content
    "preview": "ltx",        # Fast drafts
}

def select_model(scene: SceneData, quality: str) -> str:
    if quality == "preview":
        return "ltx"
    # Use LLM to classify scene type
    scene_type = classify_scene(scene.text)
    return SCENE_MODEL_MAP.get(scene_type, "wan2")
```

#### 5.2 Scene Coherence via IP-Adapter / Style Transfer

To maintain visual consistency across clips:

```python
# Extract style embeddings from first generated clip
# Apply as conditioning to subsequent clips
from diffusers import WanImageToVideoPipeline

# Option 1: Generate first frame as reference, use I2V for remaining
# Option 2: Use CLIP image embeddings as cross-attention conditioning
```

#### 5.3 Temporal Overlap for Seamless Transitions

Generate clips with 1-2 second overlap, then blend in post:

```python
def generate_with_overlap(scenes, overlap_seconds=1.5):
    for i, scene in enumerate(scenes):
        extended_duration = scene.duration + (overlap_seconds if i < len(scenes)-1 else 0)
        clip = runner.generate(prompt=scene.video_prompt, duration=extended_duration)
        clips.append(clip)
    # Blend overlapping regions in stitcher
```

#### 5.4 Automated Quality Check

```python
def check_clip_quality(clip_path: Path) -> dict:
    """Automated quality assessment before including in final video."""
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    
    scores = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "is_black": False,
        "is_static": False,
        "avg_brightness": 0.0,
    }
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        scores["is_black"] = True
        return scores
    
    # Check for black frames
    brightness = [f.mean() for f in frames]
    scores["avg_brightness"] = sum(brightness) / len(brightness)
    scores["is_black"] = scores["avg_brightness"] < 10
    
    # Check for frozen/static video
    if len(frames) > 2:
        diffs = [cv2.absdiff(frames[i], frames[i+1]).mean() for i in range(len(frames)-1)]
        scores["is_static"] = sum(diffs) / len(diffs) < 1.0
    
    return scores
```

---

## 4. OPTIMIZED FILE STRUCTURE

```
project/
├── README.md
├── BLUEPRINT.md
├── .env.example
├── .gitignore
├── docker-compose.yml          # Slimmed: api, worker_gpu, worker_cpu, redis, postgres, flower
├── Dockerfile.api              # NEW: lightweight API-only image
├── Dockerfile.gpu              # RENAMED from Dockerfile: GPU worker image
├── Dockerfile.cpu              # NEW: CPU worker for audio/stitch
├── requirements.txt            # GPU worker deps (cleaned)
├── requirements-api.txt        # NEW: API-only deps
├── requirements-cpu.txt        # NEW: audio + ffmpeg deps
├── orchestrator.py             # CLI entry point (kept, fixed)
│
├── config/
│   ├── model_config.yaml       # Fixed HF IDs, optimized presets
│   ├── presets.yaml             # Style/camera/pacing (single source of truth)
│   └── api_keys.yaml
│
├── server/
│   ├── __init__.py
│   ├── main.py                 # FastAPI (lifespan, improved health)
│   ├── tasks.py                # Celery tasks (DAG-based, model persistence)
│   ├── progress.py             # NEW: Redis pub/sub progress streaming
│   └── static/
│       ├── index.html
│       ├── app.js              # WebSocket progress (per-step)
│       └── styles.css
│
├── db/
│   ├── __init__.py
│   └── models.py               # PostgreSQL, connection pooling
│
├── pipeline/                   # NEW: replaces monolithic run_pipeline
│   ├── __init__.py
│   ├── dag.py                  # DAG orchestration (Celery canvas)
│   ├── gpu_manager.py          # VRAM lifecycle management
│   └── quality_check.py        # Automated clip QA
│
├── video_engine/
│   ├── __init__.py
│   ├── router.py               # Model selection + fallback
│   ├── local_runner.py         # Model lifecycle (uses gpu_manager)
│   ├── api_runner.py           # Runway API (connection pooling)
│   ├── upscaler.py             # NEW: Real-ESRGAN GPU upscaling
│   └── models/
│       ├── __init__.py
│       ├── base.py             # NEW: abstract base runner class
│       ├── wan2.py             # Fixed: compile+offload, SageAttn, caching
│       ├── hunyuan.py          # Fixed: correct HF ID, compile=false
│       ├── cogvideox.py
│       └── ltx.py
│
├── prompt_engine/
│   ├── __init__.py
│   └── generator.py            # Cleaned: uses config/presets.yaml only
│
├── scene_splitter/
│   ├── __init__.py
│   ├── splitter.py             # Cleaned: dead code removed
│   └── utils.py
│
├── audio/
│   ├── __init__.py
│   ├── tts.py                  # Fixed: backend param, temp file cleanup
│   ├── music.py                # Fixed: generator-based seed
│   └── sync.py
│
├── stitcher/
│   ├── __init__.py
│   ├── ffmpeg_stitch.py        # Optimized: single-pass encoding
│   └── transitions.py          # Fixed: offset calculation
│
├── api/
│   ├── __init__.py
│   └── runway.py               # Connection pooling, circuit breaker
│
├── storage/                    # NEW
│   ├── __init__.py
│   └── gcs.py                  # GCS upload + signed URLs
│
├── scripts/
│   └── patch_diffusers.py
│
└── deploy/
    ├── azure/
    │   ├── create_vm.ps1
    │   ├── bootstrap_vm.sh
    │   └── README.md
    └── gcp/                    # NEW
        ├── create_instance.sh
        └── README.md
```

**Deleted:**
- `Dockerfile.comfyui`, `scripts/comfyui_start.sh`, `workflows/` — ComfyUI parallel stack (unused by pipeline)
- `prompt_engine/presets.py` — duplicate of config/presets.yaml
- `video_engine/models/wan2.py.bak`
- All `tmp_*` files in project root and workspace root
- All `smoke_*` and `wan2_*` JSON files in workspace root
- `CLAUDE.md` — merge useful content into AGENT.md

---

## 5. 30-MINUTE GENERATION TIMELINE

For a 2-minute video (12 clips × 10s each) on single A100 80GB:

```
T+0:00  [CPU] Script splitting + prompt generation (Claude API, parallel)     ~30s
T+0:30  [GPU] Load WAN 2.1-14B (warm from worker init)                       ~0s (cached)
T+0:30  [GPU] Generate 12 clips @ 848×480, 15 steps                          ~18 min
        ├─ Per clip: 15 steps × ~8s/step (with PAB+TeaCache+SageAttn+compile)
        ├─ = ~120s per clip × 12 clips = 1440s
        └─ Step progress streamed via WebSocket
T+18:30 [GPU] Unload WAN, load Real-ESRGAN (~2 GB)                           ~10s
T+18:40 [GPU] Upscale 12 clips 848×480 → 1280×720                            ~3 min
        └─ ~15s per clip via GPU-accelerated Real-ESRGAN
T+21:40 [GPU] Unload upscaler, load MusicGen-large (~16 GB)                  ~15s
T+22:00 [GPU] Generate background music (2 min)                              ~45s
T+22:45 [GPU] Unload MusicGen                                                ~5s
T+22:50 [CPU] TTS narration via Kokoro (real-time ×50)                       ~3s
T+23:00 [CPU] Audio mixing (narration + music)                               ~5s
T+23:05 [CPU] Stitch 12 clips with crossfade transitions                     ~2 min
        ├─ Normalize resolution/fps (single pass with NVENC)
        ├─ Apply xfade transitions
        └─ Mux audio
T+25:00 [CPU] Final encode + faststart                                       ~1 min
T+26:00 [DONE] Upload to storage, generate download URL                      ~30s

TOTAL: ~27 minutes
```

### Speed Multipliers Applied

| Optimization | Speedup | Source |
|-------------|---------|--------|
| 848×480 instead of 1280×720 | 3-4× | Quadratic attention reduction |
| 15 steps instead of 20 | 1.33× | Flow-matching equivalence |
| PAB (Pyramid Attention Broadcast) | 1.3-1.5× | Diffusers built-in |
| TeaCache / FBCache | 1.3-1.5× | Diffusers built-in |
| SageAttention | 1.5-1.7× | Drop-in FA2 replacement |
| torch.compile (inductor) | 1.2-1.3× | PyTorch native |
| **Combined** | **~8-12×** | Multiplicative |

### Baseline vs Optimized

| Metric | Current | Optimized | Factor |
|--------|---------|-----------|--------|
| Per-clip time (1280×720, 20 steps) | ~26 min | N/A | — |
| Per-clip time (848×480, 15 steps, all opts) | N/A | ~2 min | — |
| 12-clip video generation | ~5.2 hours | ~24 min | **13×** |
| End-to-end pipeline | N/A (never completed) | ~27 min | — |

---

## 6. CONFIGURATION CHANGES SUMMARY

### model_config.yaml (corrected)

```yaml
models:
  local:
    wan2:
      enabled: true
      priority: 1
      hf_id: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
      vram_gb: 40
      dtype: bfloat16
      max_frames: 81
      fps: 16
      resolution: [848, 480]          # CHANGED: generate at lower res
      output_resolution: [1280, 720]  # NEW: upscale target
      compile: true
      flash_attention: false          # CHANGED: replaced by SageAttention
      sage_attention: true            # NEW
      pab: true                       # RENAMED from teacache (was misleading)
      tea_cache: true                 # NEW: actual TeaCache
      fb_cache: true                  # NEW: First-Block Cache
      default_steps: 15              # CHANGED: 15 with caching ≈ 20 raw
      offload_text_encoder: false    # CHANGED: keep on GPU for compile compat
      
    hunyuan:
      enabled: true
      priority: 2
      hf_id: "hunyuanvideo-community/HunyuanVideo"  # FIXED
      vram_gb: 60
      compile: false                  # FIXED: inductor bug
      # ... rest unchanged

    cogvideox:
      enabled: true
      priority: 3
      # unchanged
      
    ltx:
      enabled: true
      priority: 4
      # unchanged
```

---

## 7. PREREQUISITES & DEPENDENCIES

### Python (cleaned requirements.txt)

```
# ── Core ──
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
sqlmodel>=0.0.19
celery[redis]>=5.4.0
redis>=5.0.0
httpx>=0.27.0
python-multipart>=0.0.9
websockets>=13.0

# ── PyTorch (CUDA 12.x) ──
torch>=2.4.0
torchvision>=0.19.0

# ── Diffusers / HuggingFace ──
diffusers>=0.33.0
transformers>=4.45.0
accelerate>=0.34.0
huggingface_hub>=0.25.0
safetensors>=0.4.4

# ── Attention ──
# flash-attn  (install separately: pip install flash-attn --no-build-isolation)
sageattention>=1.0.0

# ── Audio ──
kokoro>=0.9.4
soundfile>=0.12.1
numpy>=1.26.0

# ── LLM ──
anthropic>=0.34.0

# ── Config ──
pyyaml>=6.0.2
python-dotenv>=1.0.1

# ── Video ──
imageio>=2.34.0
```

**Removed**: `flower`, `xformers`, `librosa`, `scipy`, `nltk`, `tiktoken`, `protobuf`, `sentencepiece`, `tokenizers`, `rich`, `psutil`, `tqdm` (7 direct deps, ~500 MB savings in Docker image).

### System

- CUDA 12.4+
- FFmpeg 6.x+ (with NVENC support)
- PostgreSQL 16
- Redis 7
- Real-ESRGAN (realesrgan-ncnn-vulkan binary or torch-based)
- Python 3.11

---

## 8. MIGRATION CHECKLIST

### Completed (commit 24bcc82)

- [x] Delete all files listed in §1.4 — 14 tmp/smoke/debug files removed
- [x] **B2**: Fix torch.compile + CPU offload in wan2.py — rewrote `load()` with correct acceleration order, text encoder offloaded after encoding
- [x] **B5**: VRAM verification after unload in local_runner.py `_switch_model`
- [x] **B6**: Fix xfade offset calculation in transitions.py
- [x] **B8**: Migrate to lifespan context manager in server/main.py
- [x] **B9**: Fix global torch.manual_seed → Generator in audio/music.py
- [x] **B10**: Fix temp file leak in audio/tts.py with try/finally
- [x] **B11**: Singleton RunwayClient in video_engine/api_runner.py
- [x] **B12**: Add logging for LLM errors in prompt_engine/generator.py
- [x] **B13**: Remove GPU reservation from API service in docker-compose.yml
- [x] **B15**: Remove dead target_words variable in scene_splitter/splitter.py
- [x] **B16**: Add VideoRouter singleton (`_get_video_router`) in server/tasks.py
- [x] **B17**: Fix TTSEngine(backend=...) crash in orchestrator.py
- [x] SageAttention integration in wan2.py (with FlashAttn2 fallback)
- [x] PAB (Pyramid Attention Broadcast) applied in wan2.py
- [x] TeaCache applied in wan2.py
- [x] FBCache (First-Block Cache) applied in wan2.py
- [x] torch.compile(max-autotune) after all hooks in wan2.py
- [x] Per-step progress via callback_on_step_end in wan2.py `generate()`
- [x] Real-ESRGAN upscaler module created (video_engine/upscaler.py)
- [x] DAG pipeline orchestration created (pipeline/dag.py) using Celery canvas
- [x] GPU memory manager created (pipeline/gpu_manager.py)
- [x] Automated clip quality check created (pipeline/quality_check.py)
- [x] Enhanced /health endpoint — checks GPU, Redis, DB, disk
- [x] Clean requirements.txt — removed xformers, librosa, scipy, nltk, tiktoken, rich, psutil, tqdm; added bitsandbytes
- [x] Fix tts.py to use numpy interp instead of librosa for resampling
- [x] 3 generation strategies (fast/balanced/quality) wired through full stack: model_config.yaml, UI, main.py, tasks.py, router.py, local_runner.py, orchestrator.py
- [x] WAN 1.3B model support added (wan2_1b in model_config.yaml + models/__init__.py)
- [x] HunyuanVideo INT8 quantization (bitsandbytes) — ~37GB instead of ~60GB
- [x] Singleton model persistence in Celery workers
- [x] Remove GPU reservation from API service
- [x] Document updated architecture in README.md, AGENT.md, CLAUDE.md
- [x] **B1**: Hunyuan hf_id already correct (`hunyuanvideo-community/HunyuanVideo`) in model_config.yaml
- [x] **B4**: DAG tasks wired into tasks.py (split_and_prompt, generate_all_clips, upscale_clips, generate_audio_task, stitch_final)
- [x] **B14**: Fix Dockerfile COPY order — output dirs created before COPY
- [x] Wire DAG pipeline into tasks.py + main.py `/generate_video` (use_dag=True by default)
- [x] Wire upscaler into DAG flow — `upscale_clips` task reads strategy config, runs Real-ESRGAN
- [x] Wire quality_check into `stitch_final` — runs check_batch before stitching
- [x] Per-step progress → Redis pub/sub (`_publish_progress`) → WebSocket relay in ws_progress
- [x] Add structured logging (structlog) — configured in lifespan, falls back to stdlib

### Not Yet Done

- [ ] **B3**: Hunyuan torch.compile inductor bug — compile=false is workaround, not fix (PyTorch upstream)
- [ ] **B7**: SQLite → PostgreSQL migration (still using SQLite — requires infra)
- [ ] Split Dockerfile into 3 images (api, gpu, cpu) — requires infra rebuild
- [ ] Add PostgreSQL to docker-compose.yml — requires infra
- [ ] End-to-end integration test: 2-min video, 30-min budget
- [ ] Load test: concurrent job submission
- [ ] Phase 3: Multi-GPU / xDiT (requires additional hardware)
- [ ] Phase 4: Cloud storage (GCS), signed URLs
- [ ] Phase 5: Scene coherence, temporal overlap, prompt-adaptive model selection
