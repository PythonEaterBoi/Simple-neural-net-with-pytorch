"""OmniTrain web server: train + prompt neural nets across all modalities.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
then open http://localhost:8000
"""

import json
import os
import shutil
import tempfile
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import MODALITIES, default_config
from inference.predict import (FileError, delete_checkpoint, list_checkpoints,
                               predict)
from training.jobs import JobManager

app = FastAPI(title="OmniTrain", description="Multi-modal neural net training interface")
jobs = JobManager()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ------------------------------------------------------------------- helpers
def _safe_name(name: str) -> str:
    return os.path.basename(name or "upload").replace("..", "_") or "upload"


def _extract_zips(root: str):
    """Recursively extract any zips dropped into the upload dir."""
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".zip"):
                zp = os.path.join(dirpath, f)
                dest = os.path.join(dirpath, os.path.splitext(f)[0])
                os.makedirs(dest, exist_ok=True)
                try:
                    with zipfile.ZipFile(zp) as zf:
                        for member in zf.namelist():
                            target = os.path.normpath(os.path.join(dest, member))
                            if target.startswith(os.path.abspath(dest)) or \
                               os.path.abspath(target).startswith(os.path.abspath(dest)):
                                zf.extract(member, dest)
                except zipfile.BadZipFile:
                    pass


# -------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join("static", "index.html"))


@app.get("/api/health")
def health():
    try:
        import torch
        cuda = torch.cuda.is_available()
        ver = torch.__version__
    except ImportError:
        cuda, ver = False, None
    return {"status": "ok", "torch": ver, "cuda": cuda}


@app.get("/api/modalities")
def modalities():
    return MODALITIES


@app.get("/api/defaults/{modality}/{task}")
def defaults(modality: str, task: str):
    try:
        return default_config(modality, task)
    except KeyError:
        raise HTTPException(404, f"Unknown modality/task: {modality}/{task}")


# --------------------------------------------------------------- training
@app.post("/api/train")
async def start_training(modality: str = Form(...), task: str = Form(...),
                         config: str = Form("{}"),
                         files: list[UploadFile] = File(default=[])):
    try:
        cfg = json.loads(config or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid config JSON.")
    merged = default_config(modality, task) if modality in MODALITIES else {}
    merged.update({k: v for k, v in cfg.items() if v != ""})
    try:
        job = jobs.create(modality, task, merged)
    except KeyError:
        raise HTTPException(400, f"Unknown modality/task: {modality}/{task}")
    data_dir = os.path.join(jobs.uploads_root, job.id)
    for uf in files:
        if not uf.filename:
            continue
        dest = os.path.join(data_dir, _safe_name(uf.filename))
        with open(dest, "wb") as f:
            shutil.copyfileobj(uf.file, f)
    _extract_zips(data_dir)
    jobs.start(job.id)
    return {"job_id": job.id}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    try:
        return jobs.get(job_id).detail()
    except KeyError:
        raise HTTPException(404, "Unknown job.")


@app.get("/api/jobs/{job_id}/logs")
def job_logs(job_id: str, since: int = 0):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job.")
    return {"status": job.status, "logs": job.logs[since:], "next": len(job.logs),
            "metrics": job.metrics}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        job = jobs.cancel(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job.")
    return {"status": job.status}


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str):
    try:
        jobs.remove(job_id)
    except KeyError:
        raise HTTPException(404, "Unknown job.")
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


# -------------------------------------------------------------- checkpoints
@app.get("/api/checkpoints")
def checkpoints():
    return list_checkpoints()


@app.delete("/api/checkpoints/{name}")
def remove_checkpoint(name: str):
    try:
        delete_checkpoint(_safe_name(name))
    except FileError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


# ------------------------------------------------------------------ prompting
@app.post("/api/prompt")
async def prompt(checkpoint: str = Form(...), prompt_text: str = Form(""),
                 values: str = Form(""), options: str = Form("{}"),
                 file: UploadFile = File(default=None)):
    try:
        opts = json.loads(options or "{}")
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid options JSON.")
    vals = None
    if values.strip():
        try:
            vals = json.loads(values)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid values JSON.")
    media_path = None
    tmpdir = None
    try:
        if file is not None and file.filename:
            tmpdir = tempfile.mkdtemp(prefix="prompt_")
            media_path = os.path.join(tmpdir, _safe_name(file.filename))
            with open(media_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            if media_path.lower().endswith(".zip"):  # frame-folder clip zips
                _extract_zips(tmpdir)
                # If the zip contained a single folder, point at it.
                entries = [os.path.join(tmpdir, e) for e in os.listdir(tmpdir)]
                dirs = [e for e in entries if os.path.isdir(e)]
                if dirs:
                    media_path = dirs[0]
        result = predict(checkpoint, prompt_text=prompt_text,
                         media_path=media_path, values=vals, options=opts)
        return JSONResponse({"ok": True, "result": result})
    except FileError as e:
        raise HTTPException(404, str(e))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
