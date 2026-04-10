"""
Celery task definitions.
Workers:
  - GPU worker (concurrency=1): scene video generation
  - CPU workers (concurrency=8): prompt generation, audio, stitching
"""
from __future__ import annotations

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
    },
)


# Pre-load and warmup the primary model on GPU worker startup
# so torch.compile overhead is paid once, not on every job.
@worker_process_init.connect
def _warmup_gpu_worker(**kwargs):
    import os
    queue = os.environ.get("CELERY_QUEUES", "")
    # Only warmup on GPU workers
    if "cpu" in queue:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        import yaml
        with open("config/model_config.yaml") as f:
            cfg = yaml.safe_load(f)
        wan_cfg = cfg.get("models", {}).get("local", {}).get("wan2", {})
        if not wan_cfg.get("enabled"):
            return
        from video_engine.models.wan2 import Wan2Runner
        runner = Wan2Runner(wan_cfg)
        runner.warmup()
        logger.info("GPU worker warmup complete.")
    except Exception as e:
        logger.warning(f"GPU worker warmup failed (non-fatal): {e}")


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
    from video_engine import VideoRouter
    from db.models import Scene, engine as db_engine

    scene = SceneData(**scene_dict)
    router = VideoRouter(quality=quality)

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
        router = VideoRouter(
            quality=cfg.get("quality", "high"),
            output_dir=str(out_dir / "scenes"),
            api_fallback=cfg.get("api_fallback", True),
        )
        clip_paths = []
        resume = cfg.get("resume", True)
        preferred_model = cfg.get("preferred_model")

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

        router.cleanup()

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
