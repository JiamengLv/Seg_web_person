
from __future__ import annotations

import base64
import fnmatch
import io
import os
import random
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from fastapi import (
    Body,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from utils import crop_img, detection_img, fits_to_jpg, read_fits
from utils_training import (
    detection_img_full,
    load_validation_model,
    mask_to_b64,
    nparray_to_b64,
    remove_white_border,
    run_detection,
    start_training_impl,
)

def safe_str(obj):
    if isinstance(obj, str):
        return obj.encode('utf-8', 'replace').decode('utf-8')
    elif isinstance(obj, Path):
        return str(obj).encode('utf-8', 'replace').decode('utf-8')
    elif isinstance(obj, dict):
        return {safe_str(k): safe_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_str(x) for x in obj]
    else:
        return obj

save_model_path_every_times_training_unet = Path(
    r'D:\Program\月球大模型-地球化学所\标签修改\model_path_new'
)

ALLOWED_ROOTS: Dict[str, Path] = {
    'data': Path(r'D:\Program\月球大模型-地球化学所\标签修改\data'),
    'home': Path(r'D:\Program\月球大模型-地球化学所'),
    'service': Path(r'D:\Program\月球大模型-地球化学所\标签修改'),
}


API_TOKEN = os.getenv('API_TOKEN', 'change-me')


app = FastAPI(title='Remote File Browser')
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


class SelectReq(BaseModel):
    root_key: str
    relpaths: List[str]


class TrainReq(BaseModel):
    training_data_path: str
    masks: Dict[str, List[List[Any]]]
    image_coords: Dict[str, List[List[Any]]]


class SaveCorrectionReq(BaseModel):
    mask_b64: str  
    filename: str  
    training_data_path: str 


class InferByPathReq(BaseModel):
    path: str

def _auth(x_token: Optional[str]):
    if x_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


def _get_root(root_key: str) -> Path:
    if root_key not in ALLOWED_ROOTS:
        raise HTTPException(status_code=400, detail=f"unknown root_key: {root_key}")
    root = ALLOWED_ROOTS[root_key].resolve()
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"root not exists: {root}")
    return root


def _safe_join(root: Path, rel: str) -> Path:

    rel = rel.strip().lstrip("/").lstrip("\\")
    p = (root / rel).resolve()
    if root == p or root in p.parents:
        return p
    raise HTTPException(status_code=400, detail="path traversal detected")


def _iter_files(base: Path, pattern: str, recursive: bool):
    if recursive:
        for p in base.rglob("*"):
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                yield p
    else:
        for p in base.iterdir():
            if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                yield p


def _job_set(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(kwargs)


def _job_get(job_id):
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="job not found")
        return dict(JOBS[job_id])


@app.get("/roots")
def list_roots(x_token: Optional[str] = Header(default=None)):
    _auth(x_token)
    return safe_str({"roots": {k: str(v.resolve()) for k, v in ALLOWED_ROOTS.items()}})


@app.get("/files")
def list_files(
    root_key: str = Query(..., description="which allowed root"),
    subdir: str = Query("", description="relative subdir under root"),
    pattern: str = Query("*", description="filename pattern, e.g. *.fits"),
    recursive: bool = Query(False),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
    x_token: Optional[str] = Header(default=None),
):
    """
    返回文件列表：relpath + 一些元信息，供你本地选择。
    """
    _auth(x_token)
    root = _get_root(root_key)
    base = _safe_join(root, subdir) if subdir else root
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"base not exists: {base}")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"base not a dir: {base}")

    items: List[Dict[str, Any]] = []
    for p in _iter_files(base, pattern=pattern, recursive=recursive):
        rel = str(p.relative_to(root))
        st = p.stat()
        items.append(
            {
                "relpath": rel,
                "name": p.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            }
        )

    total = len(items)
    page = items[offset : offset + limit]
    return safe_str({
        "root_key": root_key,
        "root": str(root),
        "base": str(base),
        "pattern": pattern,
        "recursive": recursive,
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": page,
    })


@app.post("/files/select")
def select_files(req: SelectReq, x_token: Optional[str] = Header(default=None)):
    """
    你本地把 relpaths 发回来，服务端返回绝对路径 full_paths。
    """
    _auth(x_token)
    root = _get_root(req.root_key)

    full_paths: List[str] = []
    missing: List[str] = []
    for rel in req.relpaths:
        p = _safe_join(root, rel)
        if p.exists() and p.is_file():
            full_paths.append(str(p))
        else:
            missing.append(rel)

    return safe_str({
        "root_key": req.root_key,
        "root": str(root),
        "full_paths": full_paths,
        "missing": missing,
    })


@app.get('/jobs/{job_id}')
def job_status(job_id: str, x_token: Optional[str] = Header(default=None)):
    _auth(x_token)
    j = _job_get(job_id)
    j.pop("traceback", None)
    return j


@app.get('/jobs/{job_id}/result')
def job_result(job_id: str, x_token: Optional[str] = Header(default=None)):
    _auth(x_token)
    j = _job_get(job_id)
    if j.get("status") != "done":
        raise HTTPException(status_code=400, detail=f"job not done: {j.get('status')}")
    return j.get("result", {})


@app.post('/train/start')
def start_training(
    req: TrainReq = Body(...),
    x_token: str = Header(..., alias='X-Token'),
):
    _auth(x_token)
    job_id = uuid.uuid4().hex
    _job_set(
        job_id,
        status="queued",
        stage="train",
        progress=0,
        message="queued",
        created_at=time.time(),
    )
    t = threading.Thread(
        target=_run_train_job,
        args=(job_id, req.model_dump()),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


def _run_train_job(job_id, payload):
    try:
        _job_set(job_id, status='running', stage='train', progress=1, message='preparing data')

        def progress_cb(pct: int, msg: str):
            pct = max(0, min(100, int(pct)))
            _job_set(job_id, status="running", stage="train", progress=pct, message=msg)

        result = start_training_impl(
            payload,
            progress_cb=progress_cb,
            p=save_model_path_every_times_training_unet,
        )
        _job_set(
            job_id,
            status="done",
            stage="done",
            progress=100,
            message="done",
            result=result,
            finished_at=time.time(),
        )
    except Exception as e:
        _job_set(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=str(e),
            traceback=traceback.format_exc(),
            finished_at=time.time(),
        )
import re

def clean_filename(filename: str) -> str:

    cleaned = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_.')
    if not cleaned:
        cleaned = "unnamed"
    return cleaned

@app.post('/save_correction')
def save_correction(req: SaveCorrectionReq, x_token: str = Header(..., alias='X-Token')):
    _auth(x_token)

    mask_bytes = base64.b64decode(req.mask_b64.split(',')[-1])
    img = Image.open(io.BytesIO(mask_bytes)).convert('L')
    mask_np = np.array(img, dtype=np.uint8)

    corrections_dir = Path(req.training_data_path) / 'Corrections'
    corrections_dir.mkdir(parents=True, exist_ok=True)
    fits_path = corrections_dir / (clean_filename(Path(req.filename).stem) + '.fits')

    fits.PrimaryHDU(mask_np).writeto(str(fits_path), overwrite=True)

    print(f"Saved correction mask to {fits_path} {req.filename}")
    return safe_str({"msg": "saved", "path": str(fits_path)})


@app.post('/infer_by_path')
def infer_by_path(req: InferByPathReq, x_token: Optional[str] = Header(default=None, alias="X-Token")):
    _auth(x_token)
    suffix = os.path.splitext(req.path)[-1].lower()
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail="file not found")

    if suffix in ['.fits', '.fit', '.fts']:
        image_data = read_fits(req.path).astype(np.float32)
        y0, y1, x0, x1 = remove_white_border(image_data, white_value=255)
        image_data = image_data[y0:y1, x0:x1]
    else:
        image_data = np.array(Image.open(req.path).convert('L'), dtype=np.float32)

    model = load_validation_model(str(save_model_path_every_times_training_unet))
    mask = detection_img_full(image_data, model)
    img_b64 = nparray_to_b64(image_data)
    mask_b64 = mask_to_b64(mask)

    return safe_str({
        "filename": os.path.basename(req.path),
        "original": img_b64,
        "mask": mask_b64,
    })


