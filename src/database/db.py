"""
Database connection and session management
"""
import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, Dataset, TrainingJob, TrainingResult

# Database location
DB_DIR = Path("data")
DB_FILE = DB_DIR / "training.db"

# Ensure data directory exists
DB_DIR.mkdir(exist_ok=True)

# Database URL
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db() -> Session:
    """
    Get database session (context manager)

    Usage:
        with get_db() as db:
            datasets = db.query(Dataset).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get database session (for FastAPI dependency injection)

    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db_session)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_stale_jobs():
    """
    Reset any jobs with 'running' status to 'halted' on startup.

    This handles the case where the server was restarted while jobs were running.
    Those jobs are now dead processes and should be marked as halted.

    Returns:
        int: Number of jobs that were reset
    """
    from datetime import datetime

    db = SessionLocal()
    try:
        # Find all jobs with 'running' status
        stale_jobs = db.query(TrainingJob).filter(
            TrainingJob.status == "running"
        ).all()

        count = len(stale_jobs)

        for job in stale_jobs:
            job.status = "halted"
            job.error_message = "Server restarted while job was running. Job was halted."
            job.completed_at = datetime.utcnow()

        db.commit()

        if count > 0:
            print(f"⚠️  Reset {count} stale job(s) from 'running' to 'halted'")

        return count

    except Exception as e:
        db.rollback()
        print(f"❌ Failed to reset stale jobs: {e}")
        return 0
    finally:
        db.close()
