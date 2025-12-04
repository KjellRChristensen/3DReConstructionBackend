"""
Training worker that processes training jobs from the database
"""
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..database import get_db, TrainingJob, Dataset, Model

logger = logging.getLogger(__name__)


class TrainingWorker:
    """Background worker that processes training jobs"""

    def __init__(self, check_interval: int = 5):
        """
        Initialize training worker

        Args:
            check_interval: How often to check for new jobs (seconds)
        """
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the worker in a background thread"""
        if self.running:
            logger.warning("Worker already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info("Training worker started")

    def stop(self):
        """Stop the worker"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Training worker stopped")

    def _worker_loop(self):
        """Main worker loop that checks for pending jobs"""
        logger.info("Worker loop starting...")

        while self.running:
            try:
                # Check for pending jobs
                pending_job_id = self._get_next_pending_job()

                if pending_job_id:
                    logger.info(f"Found pending job: {pending_job_id}")
                    self._process_job(pending_job_id)
                else:
                    # No pending jobs, sleep
                    time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(self.check_interval)

    def _get_next_pending_job(self) -> Optional[str]:
        """Get the next pending training job ID from database"""
        try:
            with get_db() as db:
                job = (
                    db.query(TrainingJob)
                    .filter(TrainingJob.status == "pending")
                    .order_by(TrainingJob.created_at.asc())
                    .first()
                )
                # Return job_id while session is still active
                return job.job_id if job else None
        except Exception as e:
            logger.error(f"Error fetching pending job: {e}")
            return None

    def _process_job(self, job_id: str):
        """Process a training job"""
        from .trainer import VLMTrainer

        try:
            # Update job status to running
            with get_db() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if not job:
                    logger.error(f"Job {job_id} not found")
                    return

                job.status = "running"
                job.started_at = datetime.utcnow()
                job.current_stage = "Initializing"
                db.commit()

            logger.info(f"Starting training for job {job_id}")

            # Load job details
            with get_db() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
                model = db.query(Model).filter(Model.id == job.model_id).first()

                if not dataset or not model:
                    raise ValueError("Dataset or model not found")

                # Create trainer
                logger.info(f"Creating trainer with device={job.device or 'auto'}")
                trainer = VLMTrainer(
                    job_id=job_id,
                    dataset_path=dataset.path,
                    model_name=model.huggingface_id,
                    output_dir=f"data/training/outputs/{job_id}",
                    epochs=job.epochs,
                    batch_size=job.batch_size,
                    learning_rate=job.learning_rate,
                    use_lora=job.use_lora,
                    device=job.device or "auto",
                )
                logger.info(f"Trainer created, starting training on device: {trainer.device}")

                # Run training
                trainer.train()

            # Update job status to completed
            with get_db() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.progress = 1.0
                    job.current_stage = "Completed"
                    db.commit()

            logger.info(f"Training completed for job {job_id}")

        except Exception as e:
            print(f"[WORKER] Training failed for job {job_id}: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Training failed for job {job_id}: {e}", exc_info=True)

            # Update job status to failed
            try:
                with get_db() as db:
                    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                    if job:
                        job.status = "failed"
                        job.completed_at = datetime.utcnow()
                        job.error_message = str(e)
                        db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update job status: {db_error}")
