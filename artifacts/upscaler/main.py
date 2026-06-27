import asyncio
import logging
import os
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import psutil
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from utils import database, job_queue
from utils.auth import require_api_key
from utils.image_upscaler import upscale_image
from utils.video_upscaler import upscale_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
MAX_UPLOAD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB

VALID_SCALES = {2, 4}
VALID_IMAGE_FMTS = {"PNG", "JPG", "WEBP"}
VALID_VIDEO_FMTS = {"MP4"}
VALID_COMPRESSIONS = {"high_quality", "balanced", "small_size"}

STARTUP_TIME = datetime.utcnow()


@asynccontextmanager
async def lifespan(app: FastAPI):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Initialising database...")
    await database.init_db()

    logger.info("Setting up Real-ESRGAN (clone + models + load)...")
    try:
        from utils.models_manager import setup
        await asyncio.get_event_loop().run_in_executor(None, setup)
        logger.info("Real-ESRGAN ready.")
    except Exception as e:
        logger.error(f"Real-ESRGAN setup failed: {e}", exc_info=True)

    job_queue.start_worker()
    logger.info("Job queue worker started.")

    asyncio.create_task(_cleanup_loop())

    yield

    job_queue.stop_worker()
    logger.info("Shutdown complete.")


async def _cleanup_loop():
    while True:
        await asyncio.sleep(3600)
        try:
            await database.cleanup_old_jobs()
            _cleanup_temp()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


def _cleanup_temp():
    for item in TEMP_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(str(item), ignore_errors=True)
        except OSError:
            pass


app = FastAPI(
    title="Real-ESRGAN Upscaler API",
    description=(
        "Production-ready AI image & video upscaling using the official Real-ESRGAN models. "
        "Supports 2x/4x upscaling, PNG/JPG/WEBP/MP4 output, and multiple compression levels."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/download", StaticFiles(directory=str(OUTPUT_DIR)), name="download")


def _make_download_url(filename: str, request_base: str = "") -> str:
    return f"/download/{filename}"


def _job_download_url(job_id: str) -> str:
    return f"/api/v1/download/{job_id}"


# ─────────────────────────────────────────────
# Root & health
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Real-ESRGAN Upscaler API",
        "version": "1.0.0",
        "description": "AI image and video upscaling powered by Real-ESRGAN",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "upscale_image": "POST /api/v1/upscale-image",
            "upscale_video": "POST /api/v1/upscale-video",
            "job_status": "GET /api/v1/status/{job_id}",
            "download": "GET /api/v1/download/{job_id}",
            "generate_api_key": "POST /api/v1/generate-api-key",
            "list_keys": "GET /api/v1/api-keys",
            "health": "GET /health",
            "metrics": "GET /metrics",
        },
    }


@app.get("/health", tags=["Info"])
async def health():
    return {"status": "online", "uptime_seconds": (datetime.utcnow() - STARTUP_TIME).total_seconds()}


@app.get("/metrics", tags=["Info"])
async def metrics():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except Exception:
        gpu_available = False
        gpu_name = None

    return {
        "cpu_percent": cpu,
        "ram_used_mb": mem.used / 1024 / 1024,
        "ram_total_mb": mem.total / 1024 / 1024,
        "ram_percent": mem.percent,
        "disk_used_gb": disk.used / 1024 / 1024 / 1024,
        "disk_total_gb": disk.total / 1024 / 1024 / 1024,
        "disk_percent": disk.percent,
        "queue_size": job_queue.queue_size(),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }


# ─────────────────────────────────────────────
# API key management
# ─────────────────────────────────────────────

@app.post("/api/v1/generate-api-key", tags=["API Keys"])
async def generate_api_key(name: Optional[str] = None):
    key_info = await database.generate_api_key(name or "")
    return {
        "success": True,
        "api_key": key_info["key"],
        "name": key_info["name"],
        "created_at": key_info["created_at"],
        "daily_limit": key_info["daily_limit"],
        "note": "Store this key securely. Pass it as the X-API-Key header on every request.",
    }


@app.get("/api/v1/api-keys", tags=["API Keys"])
async def list_api_keys():
    keys = await database.list_api_keys()
    return {"success": True, "count": len(keys), "api_keys": keys}


@app.delete("/api/v1/api-keys/{key}", tags=["API Keys"])
async def delete_api_key(key: str):
    deleted = await database.delete_api_key(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": "API key deleted"}


@app.post("/api/v1/api-keys/{key}/disable", tags=["API Keys"])
async def disable_api_key(key: str):
    ok = await database.set_key_enabled(key, False)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": "API key disabled"}


@app.post("/api/v1/api-keys/{key}/enable", tags=["API Keys"])
async def enable_api_key(key: str):
    ok = await database.set_key_enabled(key, True)
    if not ok:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": "API key enabled"}


@app.get("/api/v1/api-keys/{key}/usage", tags=["API Keys"])
async def get_key_usage(key: str):
    key_info = await database.validate_key(key)
    if not key_info:
        raise HTTPException(status_code=404, detail="API key not found")
    usage = await database.get_usage_for_key(key)
    return {
        "success": True,
        "key": key,
        "daily_limit": key_info["daily_limit"],
        "used_today": usage["used_today"],
        "remaining": max(0, key_info["daily_limit"] - usage["used_today"]),
        "date": usage["date"],
    }


# ─────────────────────────────────────────────
# Image upscaling
# ─────────────────────────────────────────────

async def _run_image_job(job_id: str, input_path: str, scale: int, fmt: str, compression: str):
    await database.update_job_status(job_id, "upscaling")
    ext_map = {"PNG": ".png", "JPG": ".jpg", "WEBP": ".webp"}
    ext = ext_map.get(fmt.upper(), ".png")
    output_filename = f"{job_id}{ext}"
    output_path = str(OUTPUT_DIR / output_filename)

    start = time.time()
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, upscale_image, input_path, output_path, scale, fmt, compression
        )
        elapsed = time.time() - start
        logger.info(f"Job {job_id} completed in {elapsed:.1f}s")
        await database.update_job_status(job_id, "completed", output_file=output_filename)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.post("/api/v1/upscale-image", tags=["Upscaling"])
async def upscale_image_endpoint(
    file: UploadFile = File(...),
    scale: int = Form(4),
    output_format: str = Form("PNG"),
    compression: str = Form("balanced"),
    key_info: dict = Depends(require_api_key),
):
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. Allowed: JPEG, PNG, WEBP"
        )
    if scale not in VALID_SCALES:
        raise HTTPException(status_code=400, detail=f"Invalid scale: {scale}. Choose 2 or 4.")
    fmt = output_format.upper()
    if fmt not in VALID_IMAGE_FMTS:
        raise HTTPException(status_code=400, detail=f"Invalid format: {fmt}. Choose PNG, JPG, WEBP.")
    if compression.lower() not in VALID_COMPRESSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid compression: {compression}.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 1 GB limit.")

    job_id = secrets.token_hex(16)
    suffix = Path(file.filename or "image.png").suffix or ".png"
    input_path = str(TEMP_DIR / f"{job_id}_input{suffix}")
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(content)

    logger.info(f"Image upload received: {file.filename} ({len(content)/1024:.1f} KB), job={job_id}")

    await database.create_job(job_id, key_info["key"], "image", scale, fmt, compression.lower())
    await job_queue.enqueue(job_id, _run_image_job, input_path, scale, fmt, compression.lower())

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "check_status": f"/api/v1/status/{job_id}",
        "download_url": f"/api/v1/download/{job_id}",
        "expires_in": "24h",
    }


# ─────────────────────────────────────────────
# Video upscaling
# ─────────────────────────────────────────────

async def _run_video_job(job_id: str, input_path: str, scale: int, compression: str):
    await database.update_job_status(job_id, "extracting_frames")
    output_filename = f"{job_id}.mp4"
    output_path = str(OUTPUT_DIR / output_filename)

    start = time.time()
    try:
        loop = asyncio.get_event_loop()

        def _status_cb(stage, frame_count):
            asyncio.run_coroutine_threadsafe(
                database.update_job_status(job_id, stage), loop
            )

        await loop.run_in_executor(
            None, upscale_video, input_path, output_path, scale, compression, _status_cb
        )
        elapsed = time.time() - start
        logger.info(f"Video job {job_id} completed in {elapsed:.1f}s")
        await database.update_job_status(job_id, "completed", output_file=output_filename)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


@app.post("/api/v1/upscale-video", tags=["Upscaling"])
async def upscale_video_endpoint(
    file: UploadFile = File(...),
    scale: int = Form(4),
    compression: str = Form("balanced"),
    key_info: dict = Depends(require_api_key),
):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {file.content_type}. Allowed: MP4, MOV, MKV, AVI"
        )
    if scale not in VALID_SCALES:
        raise HTTPException(status_code=400, detail=f"Invalid scale: {scale}. Choose 2 or 4.")
    if compression.lower() not in VALID_COMPRESSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid compression: {compression}.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 1 GB limit.")

    job_id = secrets.token_hex(16)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = str(TEMP_DIR / f"{job_id}_input{suffix}")
    async with aiofiles.open(input_path, "wb") as f:
        await f.write(content)

    logger.info(f"Video upload received: {file.filename} ({len(content)/1024/1024:.1f} MB), job={job_id}")

    await database.create_job(job_id, key_info["key"], "video", scale, "MP4", compression.lower())
    await job_queue.enqueue(job_id, _run_video_job, input_path, scale, compression.lower())

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "check_status": f"/api/v1/status/{job_id}",
        "download_url": f"/api/v1/download/{job_id}",
        "expires_in": "24h",
    }


# ─────────────────────────────────────────────
# Status & download
# ─────────────────────────────────────────────

@app.get("/api/v1/status/{job_id}", tags=["Jobs"])
async def job_status(job_id: str, key_info: dict = Depends(require_api_key)):
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    resp = {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "job_type": job["job_type"],
        "scale": job["scale"],
        "output_format": job["fmt"],
        "compression": job["compression"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    if job["status"] == "completed" and job["output_file"]:
        resp["download_url"] = f"/api/v1/download/{job_id}"
    if job["status"] == "failed":
        resp["error"] = job["error"]
    return resp


@app.get("/api/v1/download/{job_id}", tags=["Jobs"])
async def download_job(job_id: str, key_info: dict = Depends(require_api_key)):
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready. Current status: {job['status']}"
        )
    output_file = job["output_file"]
    if not output_file:
        raise HTTPException(status_code=404, detail="Output file not found")
    file_path = OUTPUT_DIR / output_file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file has been removed (expired)")
    return FileResponse(
        path=str(file_path),
        filename=output_file,
        media_type="application/octet-stream",
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
