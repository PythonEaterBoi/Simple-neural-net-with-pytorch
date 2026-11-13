"""Background training-job management for the web server."""

import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from config import get_task
from training.trainer import TRAIN_FUNCS

CHECKPOINTS_ROOT = "checkpoints"
UPLOADS_ROOT = "uploads"


@dataclass
class Job:
    id: str
    modality: str
    task: str
    config: dict
    status: str = "queued"  # queued | running | done | error | cancelled
    created_at: str = ""
    finished_at: str = ""
    logs: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    checkpoint: str = ""
    error: str = ""
    cancel_requested: bool = False

    def summary(self) -> dict:
        task_label = get_task(self.modality, self.task)["label"]
        mod_label = self.modality
        try:
            from config import MODALITIES
            mod_label = MODALITIES[self.modality]["label"]
        except KeyError:
            pass
        return {"id": self.id, "modality": self.modality, "task": self.task,
                "title": f"{mod_label} · {task_label}", "status": self.status,
                "created_at": self.created_at, "finished_at": self.finished_at,
                "config": self.config, "checkpoint": self.checkpoint,
                "error": self.error, "n_metrics": len(self.metrics),
                "epochs": self.config.get("epochs", 0),
                "last": self.metrics[-1] if self.metrics else None}

    def detail(self) -> dict:
        d = self.summary()
        d.update(logs=self.logs, metrics=self.metrics)
        return d


class _JobCallback:
    def __init__(self, manager: "JobManager", job: Job):
        self.manager, self.job = manager, job

    def log(self, msg: str):
        with self.manager.lock:
            self.job.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def metric(self, row: dict):
        with self.manager.lock:
            self.job.metrics.append(row)

    def cancelled(self) -> bool:
        return self.job.cancel_requested


class JobManager:
    def __init__(self, uploads_root: str = UPLOADS_ROOT,
                 checkpoints_root: str = CHECKPOINTS_ROOT, max_workers: int = 2):
        self.uploads_root, self.checkpoints_root = uploads_root, checkpoints_root
        os.makedirs(uploads_root, exist_ok=True)
        os.makedirs(checkpoints_root, exist_ok=True)
        self.jobs: dict = {}
        self.lock = threading.RLock()  # reentrant: callbacks also take this lock
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------- lifecycle
    def create(self, modality: str, task: str, config: dict) -> Job:
        get_task(modality, task)  # validates the pair
        job = Job(id=uuid.uuid4().hex[:8], modality=modality, task=task,
                  config=config, created_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        with self.lock:
            self.jobs[job.id] = job
        os.makedirs(os.path.join(self.uploads_root, job.id), exist_ok=True)
        return job

    def start(self, job_id: str):
        job = self.get(job_id)
        self.pool.submit(self._run, job)

    def _run(self, job: Job):
        with self.lock:
            job.status = "running"
        cb = _JobCallback(self, job)
        try:
            data_dir = os.path.join(self.uploads_root, job.id)
            ckpt_name = f"{job.id}_{job.modality}_{job.task}.pt"
            out_path = os.path.join(self.checkpoints_root, ckpt_name)
            cb.log(f"Starting {job.modality}/{job.task} training "
                   f"(job {job.id}, device={'cuda' if self._has_cuda() else 'cpu'}).")
            func = TRAIN_FUNCS[(job.modality, job.task)]
            func(data_dir, job.config, cb, out_path)
            with self.lock:
                was_cancelled = job.cancel_requested
                if was_cancelled:
                    job.status = "cancelled"
                else:
                    job.status = "done"
                    job.checkpoint = ckpt_name
            # NOTE: log outside the locked block (cb.log takes the lock itself).
            cb.log("Training cancelled by user." if was_cancelled
                   else f"Done. Checkpoint saved: {ckpt_name}")
        except Exception as e:  # noqa: BLE001 - surfaced to the UI
            with self.lock:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.logs.append(f"[{time.strftime('%H:%M:%S')}] ERROR: {job.error}")
                job.logs.append(traceback.format_exc(limit=3))
        finally:
            with self.lock:
                job.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ---------------------------------------------------------------- access
    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(f"Unknown job: {job_id}")
            return self.jobs[job_id]

    def list_jobs(self) -> list:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.summary() for j in jobs]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self.lock:
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
        return job

    def remove(self, job_id: str):
        job = self.get(job_id)
        if job.status in ("running", "queued"):
            raise RuntimeError("Cancel the job before deleting it.")
        with self.lock:
            del self.jobs[job_id]
        shutil.rmtree(os.path.join(self.uploads_root, job_id), ignore_errors=True)
