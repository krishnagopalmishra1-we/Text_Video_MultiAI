"""
Celery task definitions.
Workers:
  - GPU worker (concurrency=1): scene video generation
  - CPU workers (concurrency=8): prompt generation, audio, stitching
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init
from sqlmodel import Session, select

logger = logging.getLogger(__name__)

_BROKER = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "video_gen",
    broker=_BROKER,
    backend=_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # critical for GPU tasks
    task_routes={
        "server.tasks.generate_scene_clip": {"queue": "gpu"},
        "server.tasks.generate_prompts": {"queue": "cpu"},
        "server.tasks.generate_audio": {"queue": "cpu"},
        "server.tasks.stitch_video": {"queue": "cpu"},
        "server.tasks.run_pipeline": {"queue": "gpu"},
        # DAG tasks
        "server.tasks.split_and_prompt": {"queue": "cpu"},
        "server.tasks.generate_all_clips": {"queue": "gpu"},
        "server.tasks.upscale_clips": {"queue": "gpu"},
        "server.tasks.generate_audio_task": {"queue": "cpu"},
        "server.tasks.stitch_final": {"queue": "cpu"},
    },
)


# Warmup disabled — standalone runner was never reused by VideoRouter,
# causing double model loading and VRAM/crash issues.
# The first task will cold-load the requested model instead.
# @worker_process_init.connect
# def _warmup_gpu_worker(**kwargs):
#     ...


# ------------------------------------------------------------------
# VideoRouter singleton — reuse across generate_scene_clip calls
# to avoid reloading models from scratch per scene (B16 fix).
# ------------------------------------------------------------------
_video_router = None
_video_router_quality = None


def _get_video_router(quality: str = "high"):
    global _video_router, _video_router_quality
    if _video_router is None or _video_router_quality != quality:
        from video_engine import VideoRouter
        _video_router = VideoRouter(quality=quality)
        _video_router_quality = quality
    return _video_router


@celery_app.task(
    name="server.tasks.generate_prompts",
    bind=True,
    max_retries=2,
)
def generate_prompts(self, job_id: str, scenes_json: list[dict], style: str) -> list[dict]:
    import asyncio
    from scene_splitter import SceneData
    from prompt_engine import PromptEngine

    scenes = [SceneData(**s) for s in scenes_json]
    engine = PromptEngine(style=style)
    enriched = asyncio.run(engine.generate_batch(scenes))
    return [s.to_dict() for s in enriched]


@celery_app.task(
    name="server.tasks.generate_scene_clip",
    bind=True,
    max_retries=3,
    queue="gpu",
)
def generate_scene_clip(self, job_id: str, scene_dict: dict, quality: str) -> dict:
    from scene_splitter import SceneData
    from db.models import Scene, engine as db_engine

    scene = SceneData(**scene_dict)
    router = _get_video_router(quality=quality)

    with Session(db_engine) as session:
        db_scene = session.exec(
            select(Scene).where(
                Scene.job_id == job_id, Scene.scene_id == scene.scene_id
            )
        ).first()
        if db_scene:
            db_scene.status = "running"
            db_scene.updated_at = datetime.utcnow()
            session.add(db_scene)
            session.commit()

    try:
        clip_path = router.generate_scene(scene)
        with Session(db_engine) as session:
            db_scene = session.exec(
                select(Scene).where(
                    Scene.job_id == job_id, Scene.scene_id == scene.scene_id
                )
            ).first()
            if db_scene:
                db_scene.status = "done"
                db_scene.clip_path = str(clip_path)
                db_scene.updated_at = datetime.utcnow()
                session.add(db_scene)

                # Update job progress
                from db.models import Job
                job = session.get(Job, job_id)
                if job:
                    job.completed_scenes += 1
                    job.updated_at = datetime.utcnow()
                    session.add(job)
                session.commit()
        return {"scene_id": scene.scene_id, "clip_path": str(clip_path)}
    except Exception as e:
        with Session(db_engine) as session:
            db_scene = session.exec(
                select(Scene).where(
                    Scene.job_id == job_id, Scene.scene_id == scene.scene_id
                )
            ).first()
            if db_scene:
                db_scene.status = "failed"
                db_scene.error = str(e)[:500]
                db_scene.updated_at = datetime.utcnow()
                session.add(db_scene)
                session.commit()
        raise self.retry(exc=e, countdown=10)


@celery_app.task(
    name="server.tasks.generate_audio",
    bind=True,
    max_retries=2,
)
def generate_audio(
    self,
    job_id: str,
    scenes_json: list[dict],
    style: str,
    output_dir: str,
) -> dict:
    from scene_splitter import SceneData
    from audio import TTSEngine, MusicEngine, AudioSync

    scenes = [SceneData(**s) for s in scenes_json]
    total_duration = sum(s.duration for s in scenes)
    out_dir = Path(output_dir)

    tts = TTSEngine()
    narration_path = out_dir / "narration.wav"
    tts.synthesize_full(scenes, narration_path)

    music = MusicEngine()
    music_path = out_dir / "music.wav"
    music.generate_for_video(style, total_duration, music_path)

    syncer = AudioSync()
    final_audio = out_dir / "final_audio.aac"
    syncer.mix(narration_path, music_path, final_audio, total_duration, scenes)

    return {"audio_path": str(final_audio), "duration": total_duration}


@celery_app.task(
    name="server.tasks.stitch_video",
    bind=True,
    max_retries=2,
)
def stitch_video(
    self,
    job_id: str,
    clip_paths: list[str],
    audio_path: str | None,
    output_path: str,
    transition: str = "crossfade",
    upscale_4k: bool = False,
) -> str:
    from stitcher import Stitcher

    clips = [Path(p) for p in clip_paths]
    stitcher = Stitcher()
    final = stitcher.stitch(
        clip_paths=clips,
        output_path=output_path,
        audio_path=audio_path,
        transition=transition,
        upscale_to_4k=upscale_4k,
    )
    return str(final)


@celery_app.task(
    name="server.tasks.run_pipeline",
    bind=True,
    max_retries=1,
    queue="gpu",
    soft_time_limit=3600,   # 60 min soft limit (raises SoftTimeLimitExceeded)
    time_limit=3900,        # 65 min hard kill
)
def run_pipeline(self, job_id: str, config: dict) -> str:
    """
    Full end-to-end pipeline orchestration as a single Celery task.
    For simpler deployments without chord/group complexity.
    """
    import asyncio
    from scene_splitter import SceneSplitter
    from prompt_engine import PromptEngine
    from video_engine import VideoRouter
    from audio import TTSEngine, MusicEngine, AudioSync
    from stitcher import Stitcher
    from db.models import Job, Scene, engine as db_engine

    cfg = config
    out_dir = Path("outputs") / job_id
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    def _update_job(status: str, **kwargs):
        with Session(db_engine) as session:
            job = session.get(Job, job_id)
            if job:
                job.status = status
                job.updated_at = datetime.utcnow()
                for k, v in kwargs.items():
                    setattr(job, k, v)
                session.add(job)
                session.commit()

    try:
        _update_job("running")

        # 1. Split
        splitter = SceneSplitter(
            pacing=cfg.get("pacing", "normal"),
            min_duration=cfg.get("min_clip", 5),
            max_duration=cfg.get("max_clip", 20),
        )
        scenes = splitter.split(cfg["script"])
        _update_job("running", total_scenes=len(scenes))

        with Session(db_engine) as session:
            for s in scenes:
                session.add(Scene(
                    job_id=job_id, scene_id=s.scene_id,
                    text=s.text, duration=s.duration,
                ))
            session.commit()

        # 2. Generate prompts
        engine = PromptEngine(style=cfg.get("style", "cinematic"))
        scenes = asyncio.run(engine.generate_batch(scenes))

        # 3. Generate video clips (sequential — single A100)
        strategy = cfg.get("strategy", "balanced")
        router = VideoRouter(
            quality=cfg.get("quality", "high"),
            output_dir=str(out_dir / "scenes"),
            api_fallback=cfg.get("api_fallback", True),
            strategy=strategy,
        )
        clip_paths = []
        resume = cfg.get("resume", True)
        preferred_model = cfg.get("preferred_model")

        # Determine hero scenes from strategy config
        import yaml as _yaml
        with open("config/model_config.yaml") as _f:
            _strat_cfg = _yaml.safe_load(_f).get("strategies", {}).get(strategy, {})
        _hero_ids_cfg = _strat_cfg.get("hero_scene_ids", [])
        if _hero_ids_cfg == "auto":
            # auto = first and last scene
            hero_scene_ids = {scenes[0].scene_id, scenes[-1].scene_id} if scenes else set()
        elif _hero_ids_cfg:
            hero_scene_ids = set(_hero_ids_cfg)
        else:
            hero_scene_ids = set()

        try:
            for scene in scenes:
                out_path = out_dir / "scenes" / f"scene_{scene.scene_id:04d}.mp4"

                with Session(db_engine) as session:
                    db_scene = session.exec(
                        select(Scene).where(
                            Scene.job_id == job_id,
                            Scene.scene_id == scene.scene_id,
                        )
                    ).first()

                    if resume and out_path.exists():
                        if db_scene:
                            db_scene.status = "done"
                            db_scene.clip_path = str(out_path)
                            db_scene.updated_at = datetime.utcnow()
                            session.add(db_scene)

                        job = session.get(Job, job_id)
                        if job:
                            done_scenes = session.exec(
                                select(Scene).where(
                                    Scene.job_id == job_id,
                                    Scene.status == "done",
                                )
                            ).all()
                            job.completed_scenes = len(done_scenes)
                            job.updated_at = datetime.utcnow()
                            session.add(job)
                        session.commit()

                        clip_paths.append(out_path)
                        continue

                    if db_scene:
                        db_scene.status = "running"
                        db_scene.updated_at = datetime.utcnow()
                        session.add(db_scene)
                        session.commit()

                try:
                    clip_path = router.generate_scene(
                        scene,
                        preferred_model=preferred_model,
                        seed=scene.scene_id,
                        is_hero=scene.scene_id in hero_scene_ids,
                    )
                except Exception as e:
                    with Session(db_engine) as session:
                        db_scene = session.exec(
                            select(Scene).where(
                                Scene.job_id == job_id,
                                Scene.scene_id == scene.scene_id,
                            )
                        ).first()
                        if db_scene:
                            db_scene.status = "failed"
                            db_scene.error = str(e)[:500]
                            db_scene.updated_at = datetime.utcnow()
                            session.add(db_scene)
                            session.commit()
                    raise

                # Quality check — detect black/static/garbage clips
                from pipeline.quality_check import check_clip_quality
                qc = check_clip_quality(clip_path)
                if not qc.get("passed", True) and not qc.get("skipped", False):
                    logger.warning(
                        f"Scene {scene.scene_id} QC FAILED: {qc} — regenerating"
                    )
                    clip_path = router.generate_scene(
                        scene,
                        preferred_model=preferred_model,
                        seed=(scene.scene_id + 1000),
                        is_hero=scene.scene_id in hero_scene_ids,
                    )

                with Session(db_engine) as session:
                    db_scene = session.exec(
                        select(Scene).where(
                            Scene.job_id == job_id,
                            Scene.scene_id == scene.scene_id,
                        )
                    ).first()
                    if db_scene:
                        db_scene.status = "done"
                        db_scene.clip_path = str(clip_path)
                        db_scene.updated_at = datetime.utcnow()
                        session.add(db_scene)

                    job = session.get(Job, job_id)
                    if job:
                        done_scenes = session.exec(
                            select(Scene).where(
                                Scene.job_id == job_id,
                                Scene.status == "done",
                            )
                        ).all()
                        job.completed_scenes = len(done_scenes)
                        job.updated_at = datetime.utcnow()
                        session.add(job)

                    session.commit()

                clip_paths.append(clip_path)
        finally:
            # Always cleanup GPU VRAM — prevents progressive leaks across jobs
            try:
                router.cleanup()
            except Exception:
                logger.warning("router.cleanup() failed during finally block")
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # 4. Audio
        tts = TTSEngine()
        total_dur = sum(s.duration for s in scenes)
        narration = tts.synthesize_full(scenes, out_dir / "audio" / "narration.wav")
        music_engine = MusicEngine()
        music = music_engine.generate_for_video(
            cfg.get("style", "cinematic"), total_dur, out_dir / "audio" / "music.wav"
        )
        music_engine.unload()
        syncer = AudioSync()
        final_audio = syncer.mix(
            narration, music, out_dir / "audio" / "mixed.aac", total_dur, scenes
        )

        # 5. Stitch
        stitcher = Stitcher()
        final_video = stitcher.stitch(
            clip_paths=clip_paths,
            output_path=out_dir / f"{job_id}_final.mp4",
            audio_path=final_audio,
            transition=cfg.get("transition", "crossfade"),
            upscale_to_4k=cfg.get("upscale_4k", False),
        )

        _update_job("done", output_path=str(final_video))
        return str(final_video)

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        _update_job("failed", error=str(e)[:1000])
        raise


# ==================================================================
# Redis pub/sub progress helper
# ==================================================================

def _publish_progress(job_id: str, **kwargs):
    """Publish per-step progress to Redis channel for WebSocket relay."""
    try:
        import redis as _redis
        r = _redis.from_url(_BROKER, decode_responses=True)
        r.publish(f"progress:{job_id}", json.dumps(kwargs))
    except Exception:
        pass  # non-critical — don't break generation


def _step_progress_callback(job_id: str, scene_id: int, total_scenes: int):
    """Return a callback_on_step_end function for diffusion progress."""
    def callback(pipe, step, timestep, callback_kwargs):
        _publish_progress(
            job_id,
            type="step",
            scene_id=scene_id,
            step=step,
            total_scenes=total_scenes,
        )
        return callback_kwargs
    return callback


# ==================================================================
# DAG pipeline tasks — used by pipeline/dag.py
# ==================================================================

@celery_app.task(name="server.tasks.split_and_prompt", bind=True, max_retries=2)
def split_and_prompt(self, job_id: str, config: dict) -> list[dict]:
    """Step 1+2: Split script into scenes and generate video prompts (CPU)."""
    import asyncio
    from scene_splitter import SceneSplitter
    from prompt_engine import PromptEngine
    from db.models import Job, Scene, engine as db_engine

    cfg = config
    splitter = SceneSplitter(
        pacing=cfg.get("pacing", "normal"),
        min_duration=cfg.get("min_clip", 5),
        max_duration=cfg.get("max_clip", 20),
    )
    scenes = splitter.split(cfg["script"])

    # Persist scenes to DB
    with Session(db_engine) as session:
        job = session.get(Job, job_id)
        if job:
            job.status = "running"
            job.total_scenes = len(scenes)
            job.updated_at = datetime.utcnow()
            session.add(job)
        for s in scenes:
            session.add(Scene(
                job_id=job_id, scene_id=s.scene_id,
                text=s.text, duration=s.duration,
            ))
        session.commit()

    # Generate prompts
    engine = PromptEngine(style=cfg.get("style", "cinematic"))
    enriched = asyncio.run(engine.generate_batch(scenes))

    _publish_progress(job_id, type="phase", phase="prompts_done", count=len(enriched))
    return [s.to_dict() for s in enriched]


@celery_app.task(name="server.tasks.generate_all_clips", bind=True, max_retries=1, queue="gpu")
def generate_all_clips(self, scenes_json: list[dict], job_id: str, config: dict) -> list[str]:
    """Step 3: Generate all video clips sequentially on GPU."""
    import yaml as _yaml
    from scene_splitter import SceneData
    from db.models import Job, Scene, engine as db_engine

    cfg = config
    strategy = cfg.get("strategy", "balanced")
    out_dir = Path("outputs") / job_id / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    router = _get_video_router(quality=cfg.get("quality", "high"))
    scenes = [SceneData(**s) for s in scenes_json]

    # Determine hero scenes
    with open("config/model_config.yaml") as _f:
        strat_cfg = _yaml.safe_load(_f).get("strategies", {}).get(strategy, {})
    hero_ids_cfg = strat_cfg.get("hero_scene_ids", [])
    if hero_ids_cfg == "auto":
        hero_scene_ids = {scenes[0].scene_id, scenes[-1].scene_id} if scenes else set()
    elif hero_ids_cfg:
        hero_scene_ids = set(hero_ids_cfg)
    else:
        hero_scene_ids = set()

    clip_paths = []
    for scene in scenes:
        out_path = out_dir / f"scene_{scene.scene_id:04d}.mp4"

        # Resume check
        if cfg.get("resume", True) and out_path.exists():
            clip_paths.append(str(out_path))
            continue

        # Update scene status
        with Session(db_engine) as session:
            db_scene = session.exec(
                select(Scene).where(Scene.job_id == job_id, Scene.scene_id == scene.scene_id)
            ).first()
            if db_scene:
                db_scene.status = "running"
                db_scene.updated_at = datetime.utcnow()
                session.add(db_scene)
                session.commit()

        try:
            clip_path = router.generate_scene(
                scene,
                preferred_model=cfg.get("preferred_model"),
                seed=scene.scene_id,
                is_hero=scene.scene_id in hero_scene_ids,
            )
        except Exception as e:
            with Session(db_engine) as session:
                db_scene = session.exec(
                    select(Scene).where(Scene.job_id == job_id, Scene.scene_id == scene.scene_id)
                ).first()
                if db_scene:
                    db_scene.status = "failed"
                    db_scene.error = str(e)[:500]
                    db_scene.updated_at = datetime.utcnow()
                    session.add(db_scene)
                    session.commit()
            raise

        # Mark done
        with Session(db_engine) as session:
            db_scene = session.exec(
                select(Scene).where(Scene.job_id == job_id, Scene.scene_id == scene.scene_id)
            ).first()
            if db_scene:
                db_scene.status = "done"
                db_scene.clip_path = str(clip_path)
                db_scene.updated_at = datetime.utcnow()
                session.add(db_scene)
            job = session.get(Job, job_id)
            if job:
                job.completed_scenes = len(clip_paths) + 1
                job.updated_at = datetime.utcnow()
                session.add(job)
            session.commit()

        clip_paths.append(str(clip_path))
        _publish_progress(
            job_id, type="clip_done",
            scene_id=scene.scene_id, done=len(clip_paths), total=len(scenes),
        )

    router.cleanup()
    return clip_paths


@celery_app.task(name="server.tasks.upscale_clips", bind=True, max_retries=1, queue="gpu")
def upscale_clips(self, clip_paths_or_prev: list[str] | None, job_id: str, config: dict) -> list[str]:
    """Step 4a: Upscale clips if strategy requires it (GPU)."""
    import yaml as _yaml

    strategy = config.get("strategy", "balanced")
    with open("config/model_config.yaml") as _f:
        strat_cfg = _yaml.safe_load(_f).get("strategies", {}).get(strategy, {})

    if not strat_cfg.get("upscale", False):
        logger.info(f"Strategy '{strategy}' — upscale disabled, skipping")
        return clip_paths_or_prev or []

    from video_engine.upscaler import Upscaler

    target_w, target_h = strat_cfg.get("output_resolution", [1280, 720])
    upscaler = Upscaler()

    out_dir = Path("outputs") / job_id / "upscaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = clip_paths_or_prev or []
    upscaled = []
    for cp in clip_paths:
        p = Path(cp)
        out = out_dir / p.name
        try:
            result = upscaler.upscale_clip(p, out, target_width=target_w, target_height=target_h)
            upscaled.append(str(result))
        except Exception as e:
            logger.warning(f"Upscale failed for {p.name}: {e} — using original")
            upscaled.append(cp)

    _publish_progress(job_id, type="phase", phase="upscale_done", count=len(upscaled))
    return upscaled


@celery_app.task(name="server.tasks.generate_audio_task", bind=True, max_retries=2)
def generate_audio_task(self, job_id: str, config: dict) -> dict:
    """Step 4b: Generate TTS + music (CPU, runs in parallel with upscale)."""
    from scene_splitter import SceneData
    from audio import TTSEngine, MusicEngine, AudioSync
    from db.models import Scene, engine as db_engine

    out_dir = Path("outputs") / job_id / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve scenes from DB
    with Session(db_engine) as session:
        db_scenes = session.exec(
            select(Scene).where(Scene.job_id == job_id).order_by(Scene.scene_id)
        ).all()
        scenes = [
            SceneData(
                scene_id=s.scene_id, text=s.text, duration=s.duration,
                word_count=len(s.text.split()), video_prompt=s.text, narration=s.text,
            )
            for s in db_scenes
        ]

    total_dur = sum(s.duration for s in scenes)

    tts = TTSEngine()
    narration = tts.synthesize_full(scenes, out_dir / "narration.wav")

    music_engine = MusicEngine()
    music = music_engine.generate_for_video(
        config.get("style", "cinematic"), total_dur, out_dir / "music.wav"
    )
    music_engine.unload()

    syncer = AudioSync()
    final_audio = syncer.mix(narration, music, out_dir / "mixed.aac", total_dur, scenes)

    _publish_progress(job_id, type="phase", phase="audio_done")
    return {"audio_path": str(final_audio), "duration": total_dur}


@celery_app.task(name="server.tasks.stitch_final", bind=True, max_retries=2)
def stitch_final(self, _prev_results, job_id: str, config: dict) -> str:
    """Step 5: Quality check + stitch + mux audio (CPU)."""
    from stitcher import Stitcher
    from pipeline.quality_check import check_batch
    from db.models import Job, Scene, engine as db_engine

    out_dir = Path("outputs") / job_id

    # Collect clip paths — prefer upscaled, fall back to raw scenes
    upscaled_dir = out_dir / "upscaled"
    scenes_dir = out_dir / "scenes"

    with Session(db_engine) as session:
        db_scenes = session.exec(
            select(Scene).where(Scene.job_id == job_id).order_by(Scene.scene_id)
        ).all()

    clip_paths = []
    for s in db_scenes:
        upscaled = upscaled_dir / f"scene_{s.scene_id:04d}.mp4"
        raw = scenes_dir / f"scene_{s.scene_id:04d}.mp4"
        clip_paths.append(upscaled if upscaled.exists() else raw)

    # Quality check
    qc_results = check_batch(clip_paths)
    failed = [r for r in qc_results if not r.get("passed", True) and not r.get("skipped")]
    if failed:
        logger.warning(f"Quality check: {len(failed)}/{len(qc_results)} clips failed: "
                       f"{[r['path'] for r in failed]}")

    # Get audio path
    audio_path = out_dir / "audio" / "mixed.aac"
    if not audio_path.exists():
        audio_path = None

    # Stitch
    stitcher = Stitcher()
    final_video = stitcher.stitch(
        clip_paths=clip_paths,
        output_path=out_dir / f"{job_id}_final.mp4",
        audio_path=str(audio_path) if audio_path else None,
        transition=config.get("transition", "crossfade"),
        upscale_to_4k=config.get("upscale_4k", False),
    )

    # Update job as done
    with Session(db_engine) as session:
        job = session.get(Job, job_id)
        if job:
            job.status = "done"
            job.output_path = str(final_video)
            job.updated_at = datetime.utcnow()
            session.add(job)
            session.commit()

    _publish_progress(job_id, type="phase", phase="done", output=str(final_video))
    return str(final_video)
