"""
FastAPI server — video generation API.
Endpoints:
  POST /generate_video     → submit full pipeline job
  POST /generate_scene     → generate a single scene clip
  GET  /status/{job_id}    → job/scene status + progress
  GET  /health             → liveness check
  WS   /ws/{job_id}        → real-time progress stream
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from db.models import Job, Scene, init_db, get_session
from server.tasks import run_pipeline, generate_scene_clip

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Long Video Generator",
    version="1.0.0",
    description="Multi-model AI long-form video generation system",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/app", StaticFiles(directory=str(STATIC_DIR)), name="app")


@app.on_event("startup")
def on_startup():
    init_db()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class VideoJobRequest(BaseModel):
    script: str = Field(..., min_length=10)
    style: str = Field(default="cinematic")
    quality: str = Field(default="high")           # ultra | high | balanced | fast | preview
    preferred_model: Optional[str] = Field(default=None)  # wan2 | hunyuan | cogvideox | ltx
    api_fallback: bool = Field(default=True)        # Runway fallback only after local failures
    pacing: str = Field(default="normal")          # slow | normal | fast
    transition: str = Field(default="crossfade")
    min_clip: float = Field(default=5.0)
    max_clip: float = Field(default=20.0)
    upscale_4k: bool = Field(default=False)
    resume: bool = Field(default=True)


class SceneRequest(BaseModel):
    scene_id: int
    text: str
    video_prompt: Optional[str] = None
    duration: float = Field(default=10.0)
    style: str = Field(default="cinematic")
    quality: str = Field(default="high")


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    total_scenes: int
    completed_scenes: int
    output_path: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="UI not found")
    return index_path.read_text(encoding="utf-8")


@app.post("/generate_video", response_model=JobStatusResponse)
def generate_video(
    req: VideoJobRequest,
    session: Session = Depends(get_session),
):
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        status="pending",
        script=req.script[:5000],
        style=req.style,
        quality=req.quality,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Submit to Celery
    run_pipeline.apply_async(
        args=[job_id, req.model_dump()],
        task_id=job_id,
    )

    return _job_to_response(job)


@app.post("/generate_scene")
def generate_single_scene(
    req: SceneRequest,
    session: Session = Depends(get_session),
):
    from scene_splitter import SceneData
    job_id = f"scene_{req.scene_id}_{uuid.uuid4().hex[:8]}"

    scene = SceneData(
        scene_id=req.scene_id,
        text=req.text,
        duration=req.duration,
        word_count=len(req.text.split()),
        video_prompt=req.video_prompt or req.text,
        narration=req.text,
    )

    task = generate_scene_clip.apply_async(
        args=[job_id, scene.to_dict(), req.quality],
    )
    return {"task_id": task.id, "job_id": job_id, "status": "submitted"}


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@app.get("/status/{job_id}/scenes")
def get_scene_status(job_id: str, session: Session = Depends(get_session)):
    scenes = session.exec(
        select(Scene).where(Scene.job_id == job_id).order_by(Scene.scene_id)
    ).all()
    return [
        {
            "scene_id": s.scene_id,
            "status": s.status,
            "clip_path": s.clip_path,
            "model_used": s.model_used,
            "error": s.error,
        }
        for s in scenes
    ]


@app.get("/download/{job_id}")
def download_video(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.output_path:
        raise HTTPException(status_code=400, detail=f"Job not ready: {job.status}")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


# ------------------------------------------------------------------
# WebSocket progress stream
# ------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    import asyncio
    from db.models import engine as db_engine

    await websocket.accept()
    try:
        while True:
            with Session(db_engine) as session:
                job = session.get(Job, job_id)
                if not job:
                    await websocket.send_json({"error": "Job not found"})
                    break
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "completed_scenes": job.completed_scenes,
                    "total_scenes": job.total_scenes,
                })
                if job.status in ("done", "failed"):
                    break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _job_to_response(job: Job) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        total_scenes=job.total_scenes,
        completed_scenes=job.completed_scenes,
        output_path=job.output_path,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,           # single worker — GPU state is process-local
        log_level="info",
    )
