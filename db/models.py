"""
SQLModel database models for job/scene state tracking.
Uses SQLite for local dev; swap DATABASE_URL for Postgres in prod.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy.pool import QueuePool, NullPool

_DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./outputs/jobs.db")

# Use different pool strategies based on database
if _DB_URL.startswith("postgresql"):
    # PostgreSQL: use QueuePool for concurrent access
    engine = create_engine(
        _DB_URL,
        echo=False,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=3600,    # Recycle connections every hour
    )
else:
    # SQLite: use NullPool (no connection pooling)
    engine = create_engine(
        _DB_URL,
        echo=False,
        poolclass=NullPool,
        connect_args={"check_same_thread": False}
    )


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)              # UUID
    status: str = Field(default="pending")          # pending | running | done | failed
    script: str = Field(default="")
    style: str = Field(default="cinematic")
    quality: str = Field(default="high")
    total_scenes: int = Field(default=0)
    completed_scenes: int = Field(default=0)
    output_path: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def progress(self) -> float:
        if self.total_scenes == 0:
            return 0.0
        return self.completed_scenes / self.total_scenes


class Scene(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    scene_id: int
    status: str = Field(default="pending")          # pending | running | done | failed
    text: str = Field(default="")
    video_prompt: str = Field(default="")
    clip_path: Optional[str] = Field(default=None)
    model_used: Optional[str] = Field(default=None)
    duration: float = Field(default=0.0)
    error: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
