"""Video tracking, movement, density, and inactivity analysis routes."""
from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.core.density import DEFAULT_MIN_NEIGHBORS
from app.core.detector import validate_threshold
from app.core.inactivity import METRICS
from app.core.video_processor import process_video
from app.state import (
    IMG_TEST_DIR,
    OUTPUTS_DIR,
    REPO_ROOT,
    TEMP_UPLOADS_DIR,
    get_detector,
    latest_analyses,
    video_jobs,
)

router = APIRouter(prefix="/api", tags=["video"])

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_TRACKERS = {"bytetrack", "centroid"}
MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
REPO_VIDEOS_DIR = (
    REPO_ROOT / "data" / "sample_videos"
    if (REPO_ROOT / "data" / "sample_videos").is_dir()
    else REPO_ROOT / "Tracker_Distance_Measure"
)


def _validate_video_params(
    conf: float,
    iou: float,
    tracker_type: str,
    max_lost: int,
    max_distance: Optional[float],
    jump_threshold_px: Optional[float],
    proximity_threshold_px: float,
    inactivity_window_s: float,
    inactivity_thresh_px: Optional[float],
    inactivity_metric: str,
    density_radius_px: Optional[float],
    density_min_neighbors: int,
    max_frames: Optional[int],
):
    validate_threshold("conf", conf)
    validate_threshold("iou", iou)
    if tracker_type.lower() not in ALLOWED_TRACKERS:
        raise ValueError(f"tracker_type must be one of {sorted(ALLOWED_TRACKERS)}")
    if max_lost < 0 or max_lost > 120:
        raise ValueError("max_lost must be between 0 and 120")
    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be > 0")
    if jump_threshold_px is not None and jump_threshold_px <= 0:
        raise ValueError("jump_threshold_px must be > 0")
    if proximity_threshold_px <= 0:
        raise ValueError("proximity_threshold_px must be > 0")
    if inactivity_window_s <= 0 or inactivity_window_s > 600:
        raise ValueError("inactivity_window_s must be in (0, 600]")
    if inactivity_thresh_px is not None and inactivity_thresh_px < 0:
        raise ValueError("inactivity_thresh_px must be >= 0")
    if inactivity_metric not in METRICS:
        raise ValueError(f"inactivity_metric must be one of {METRICS}")
    if density_radius_px is not None and density_radius_px <= 0:
        raise ValueError("density_radius_px must be > 0")
    if density_min_neighbors < 2:
        raise ValueError("density_min_neighbors must be >= 2")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")


@router.get("/analyze/video/progress/{job_id}")
def get_video_progress(job_id: str):
    info = video_jobs.get(job_id)
    if info is None:
        return {"job_id": job_id, "status": "unknown", "frames_done": 0, "total_frames": 0, "percent": 0.0}
    return info


@router.post("/analyze/video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    model_name: str = Form("yolov8m"),
    tracker_type: str = Form("bytetrack"),
    auto_enhance: bool = Form(True),
    containment_dedup: bool = Form(True),
    proximity_threshold_px: float = Form(50.0),
    max_lost: int = Form(30),
    max_distance: Optional[float] = Form(None),
    jump_threshold_px: Optional[float] = Form(None),
    inactivity_window_s: float = Form(10.0),
    inactivity_thresh_px: Optional[float] = Form(None),
    inactivity_metric: str = Form("extent"),
    density_radius_px: Optional[float] = Form(None),
    density_min_neighbors: int = Form(DEFAULT_MIN_NEIGHBORS),
    max_frames: Optional[int] = Form(None),
    job_id: Optional[str] = Form(None),
):
    filename = Path(file.filename or "upload.mp4").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video file extension '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}",
        )

    try:
        _validate_video_params(
            conf=conf,
            iou=iou,
            tracker_type=tracker_type,
            max_lost=max_lost,
            max_distance=max_distance,
            jump_threshold_px=jump_threshold_px,
            proximity_threshold_px=proximity_threshold_px,
            inactivity_window_s=inactivity_window_s,
            inactivity_thresh_px=inactivity_thresh_px,
            inactivity_metric=inactivity_metric,
            density_radius_px=density_radius_px,
            density_min_neighbors=density_min_neighbors,
            max_frames=max_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        detector = get_detector(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detector unavailable: {exc}")

    uid = uuid.uuid4().hex[:12]
    jid = job_id or uid
    temp_in = TEMP_UPLOADS_DIR / f"in_{uid}{ext}"
    out_name = f"vid_{uid}.mp4"
    out_path = OUTPUTS_DIR / out_name

    # Stream upload to disk in chunks
    total_bytes = 0
    try:
        with temp_in.open("wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds 500 MB limit")
                out_f.write(chunk)
    except Exception:
        temp_in.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        temp_in.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty")

    video_jobs[jid] = {
        "job_id": jid,
        "status": "processing",
        "frames_done": 0,
        "total_frames": 0,
        "percent": 0.0,
    }

    def _progress(done: int, total: int):
        pct = round((done / total) * 100.0, 1) if total > 0 else 0.0
        video_jobs[jid] = {
            "job_id": jid,
            "status": "processing",
            "frames_done": done,
            "total_frames": total,
            "percent": min(99.9, pct),
        }

    try:
        result = await run_in_threadpool(
            process_video,
            video_path=temp_in,
            output_path=out_path,
            detector=detector,
            conf=conf,
            iou=iou,
            tracker_type=tracker_type,
            auto_enhance=auto_enhance,
            containment_dedup=containment_dedup,
            proximity_threshold_px=proximity_threshold_px,
            max_lost=max_lost,
            max_distance=max_distance,
            jump_threshold_px=jump_threshold_px,
            inactivity_window_s=inactivity_window_s,
            inactivity_thresh_px=inactivity_thresh_px,
            inactivity_metric=inactivity_metric,
            density_radius_px=density_radius_px,
            density_min_neighbors=density_min_neighbors,
            max_frames=max_frames,
            progress_cb=_progress,
        )
    except ValueError as exc:
        video_jobs[jid] = {"job_id": jid, "status": "failed", "error": str(exc)}
        out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        video_jobs[jid] = {"job_id": jid, "status": "failed", "error": str(exc)}
        out_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")
    finally:
        temp_in.unlink(missing_ok=True)

    video_jobs[jid] = {
        "job_id": jid,
        "status": "completed",
        "frames_done": result["frames_processed"],
        "total_frames": result["frames_processed"],
        "percent": 100.0,
    }

    payload = {
        "job_id": jid,
        "source": filename,
        "model_used": detector.weights_path.name,
        "processed_video_url": f"/api/outputs/{out_name}",
        **result,
    }

    latest_analyses["last_video"] = {
        "source": filename,
        "model_used": detector.weights_path.name,
        "tracker_used": result["tracker_used"],
        "processed_video_url": payload["processed_video_url"],
        "frames_processed": result["frames_processed"],
        "duration_s": result["duration_s"],
        "total_unique_tracks": result["total_unique_tracks"],
        "mean_chickens_per_frame": result["counts_summary"]["mean_chickens_per_frame"],
        "max_dense_chickens": result["density_summary"]["max_dense_chickens"],
        "mean_flock_nearest_neighbor_px": result["interbird_distance_summary"]["mean_flock_nearest_neighbor_px"],
        "tracks_with_low_mobility": result["inactivity_summary"]["tracks_with_low_mobility"],
        "processing_time_s": result["processing_time_s"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


@router.get("/samples/videos")
def list_sample_videos():
    samples = []
    if REPO_VIDEOS_DIR.is_dir():
        for p in sorted(REPO_VIDEOS_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() == ".mp4":
                samples.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "note": "Repository demo video (contains original experiment overlay)",
                })
    return {"samples": samples}


@router.post("/analyze/video/sample/{sample_name}")
async def analyze_sample_video(
    sample_name: str,
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    model_name: str = Form("yolov8m"),
    tracker_type: str = Form("bytetrack"),
    auto_enhance: bool = Form(True),
    containment_dedup: bool = Form(True),
    proximity_threshold_px: float = Form(50.0),
    max_lost: int = Form(30),
    max_distance: Optional[float] = Form(None),
    jump_threshold_px: Optional[float] = Form(None),
    inactivity_window_s: float = Form(10.0),
    inactivity_thresh_px: Optional[float] = Form(None),
    inactivity_metric: str = Form("extent"),
    density_radius_px: Optional[float] = Form(None),
    density_min_neighbors: int = Form(DEFAULT_MIN_NEIGHBORS),
    max_frames: Optional[int] = Form(120),
    job_id: Optional[str] = Form(None),
):
    clean = Path(sample_name).name
    if clean != sample_name or ".." in sample_name or not SAFE_FILENAME_RE.match(clean):
        raise HTTPException(status_code=400, detail="Invalid sample video name")
    target = (REPO_VIDEOS_DIR / clean).resolve()
    if not target.is_file() or target.parent != REPO_VIDEOS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Sample video not found")

    try:
        _validate_video_params(
            conf=conf,
            iou=iou,
            tracker_type=tracker_type,
            max_lost=max_lost,
            max_distance=max_distance,
            jump_threshold_px=jump_threshold_px,
            proximity_threshold_px=proximity_threshold_px,
            inactivity_window_s=inactivity_window_s,
            inactivity_thresh_px=inactivity_thresh_px,
            inactivity_metric=inactivity_metric,
            density_radius_px=density_radius_px,
            density_min_neighbors=density_min_neighbors,
            max_frames=max_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    detector = get_detector(model_name)
    uid = uuid.uuid4().hex[:12]
    jid = job_id or uid
    out_name = f"vid_{uid}.mp4"
    out_path = OUTPUTS_DIR / out_name

    def _progress(done: int, total: int):
        pct = round((done / total) * 100.0, 1) if total > 0 else 0.0
        video_jobs[jid] = {
            "job_id": jid,
            "status": "processing",
            "frames_done": done,
            "total_frames": total,
            "percent": min(99.9, pct),
        }

    result = await run_in_threadpool(
        process_video,
        video_path=target,
        output_path=out_path,
        detector=detector,
        conf=conf,
        iou=iou,
        tracker_type=tracker_type,
        auto_enhance=auto_enhance,
        containment_dedup=containment_dedup,
        proximity_threshold_px=proximity_threshold_px,
        max_lost=max_lost,
        max_distance=max_distance,
        jump_threshold_px=jump_threshold_px,
        inactivity_window_s=inactivity_window_s,
        inactivity_thresh_px=inactivity_thresh_px,
        inactivity_metric=inactivity_metric,
        density_radius_px=density_radius_px,
        density_min_neighbors=density_min_neighbors,
        max_frames=max_frames,
        progress_cb=_progress,
    )
    video_jobs[jid] = {
        "job_id": jid,
        "status": "completed",
        "frames_done": result["frames_processed"],
        "total_frames": result["frames_processed"],
        "percent": 100.0,
    }
    payload = {
        "job_id": jid,
        "source": clean,
        "model_used": detector.weights_path.name,
        "processed_video_url": f"/api/outputs/{out_name}",
        **result,
    }
    latest_analyses["last_video"] = {
        "source": clean,
        "model_used": detector.weights_path.name,
        "tracker_used": result["tracker_used"],
        "processed_video_url": payload["processed_video_url"],
        "frames_processed": result["frames_processed"],
        "duration_s": result["duration_s"],
        "total_unique_tracks": result["total_unique_tracks"],
        "mean_chickens_per_frame": result["counts_summary"]["mean_chickens_per_frame"],
        "max_dense_chickens": result["density_summary"]["max_dense_chickens"],
        "mean_flock_nearest_neighbor_px": result["interbird_distance_summary"]["mean_flock_nearest_neighbor_px"],
        "tracks_with_low_mobility": result["inactivity_summary"]["tracks_with_low_mobility"],
        "processing_time_s": result["processing_time_s"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


@router.get("/outputs/{filename}")
def serve_output_file(filename: str):
    clean = Path(filename).name
    if clean != filename or ".." in filename or not SAFE_FILENAME_RE.match(clean):
        raise HTTPException(status_code=400, detail="Invalid output filename")
    target = (OUTPUTS_DIR / clean).resolve()
    if not target.is_file() or target.parent != OUTPUTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Output file not found")
    ext = target.suffix.lower()
    media_type = "video/mp4" if ext == ".mp4" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/png")
    return FileResponse(str(target), media_type=media_type)
