"""
DAG-based pipeline orchestration using Celery canvas.

Replaces monolithic run_pipeline with a proper task DAG:
  [Split+Prompt (CPU)] → [Video Clips (sequential GPU)] →
  [Upscale (GPU) + Audio (CPU) in parallel] → [Quality Check + Stitch (CPU)]
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
        generate_audio_task,
        stitch_final,
    )

    pipeline = chain(
        # Step 1+2: Split script and generate prompts (CPU)
        split_and_prompt.si(job_id, config),

        # Step 3: Generate all video clips sequentially (GPU)
        generate_all_clips.s(job_id, config),

        # Step 4: Upscale (GPU) + Audio (CPU) in parallel, then stitch
        chord(
            group(
                upscale_clips.s(job_id, config),
                generate_audio_task.si(job_id, config),
            ),
            stitch_final.si(None, job_id, config),
        ),
    )

    result = pipeline.apply_async(task_id=f"dag-{job_id}")
    logger.info(f"DAG pipeline submitted: {result.id}")
    return result.id
