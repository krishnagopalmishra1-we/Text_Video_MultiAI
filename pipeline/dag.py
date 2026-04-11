"""
DAG-based pipeline orchestration using Celery canvas.

Replaces monolithic run_pipeline with a proper task DAG:
  [Split] → [Prompts] → [Video Clips (sequential GPU)] →
  [Upscale (GPU) + Audio (CPU) in parallel] → [Stitch (CPU)]
"""
from __future__ import annotations

import logging

from celery import chain, chord, group

logger = logging.getLogger(__name__)


def submit_pipeline(job_id: str, config: dict) -> str:
    """
    Build and submit the video generation DAG.
    Returns the Celery AsyncResult ID.
    """
    from server.tasks import (
        split_and_prompt,
        generate_all_clips,
        upscale_clips,
        generate_audio,
        stitch_video,
    )

    pipeline = chain(
        # Step 1+2: Split script and generate prompts (CPU)
        split_and_prompt.si(job_id, config),

        # Step 3: Generate all video clips sequentially (GPU)
        generate_all_clips.s(job_id, config),

        # Step 4: Upscale + Audio in parallel, then stitch
        # chord runs the group in parallel, then calls stitch
        _build_post_pipeline(job_id, config),
    )

    result = pipeline.apply_async(task_id=f"dag-{job_id}")
    logger.info(f"DAG pipeline submitted: {result.id}")
    return result.id


def _build_post_pipeline(job_id: str, config: dict):
    """
    Build the post-video-generation phase:
    parallel(upscale, audio) → stitch.
    
    Returns a Celery canvas primitive.
    """
    from server.tasks import (
        upscale_clips,
        generate_audio_task,
        stitch_final,
    )

    return chord(
        group(
            upscale_clips.s(job_id, config),
            generate_audio_task.si(job_id, config),
        ),
        stitch_final.si(job_id, config),
    )
