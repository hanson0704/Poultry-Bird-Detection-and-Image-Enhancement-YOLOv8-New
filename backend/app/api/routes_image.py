"""Image detection & spatial density API routes."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.density import DEFAULT_MIN_NEIGHBORS, analyze_density
from app.core.detector import annotate_detections, validate_threshold
from app.core.interbird_distance import analyze_interbird_distance
from app.state import IMG_TEST_DIR, OUTPUTS_DIR, get_detector, latest_analyses

router = APIRouter(prefix="/api", tags=["image"])

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 35 * 1024 * 1024  # 35 MB


def _run_image_pipeline(
    img_bgr: np.ndarray,
    source_label: str,
    conf: float,
    iou: float,
    density_radius_px: Optional[float],
    density_min_neighbors: int,
    model_name: str = "yolov8m",
    auto_enhance: bool = True,
    force_clahe: bool = False,
    containment_dedup: bool = True,
    proximity_threshold_px: float = 50.0,
) -> Dict:
    t0 = time.perf_counter()
    try:
        conf = validate_threshold("conf", conf)
        iou = validate_threshold("iou", iou)
        if density_min_neighbors < 2:
            raise ValueError("density_min_neighbors must be >= 2")
        if density_radius_px is not None and density_radius_px <= 0:
            raise ValueError("density_radius_px must be > 0")
        if proximity_threshold_px <= 0:
            raise ValueError("proximity_threshold_px must be > 0")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        detector = get_detector(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detector unavailable: {exc}")

    det = detector.detect(
        img_bgr,
        conf=conf,
        iou=iou,
        auto_enhance=auto_enhance,
        force_clahe=force_clahe,
        containment_dedup=containment_dedup,
    )
    density = analyze_density(
        det.centroids,
        det.bboxes_xywh,
        det.image_width,
        det.image_height,
        radius_px=density_radius_px,
        min_neighbors=density_min_neighbors,
    )
    interbird = analyze_interbird_distance(
        det.centroids,
        proximity_threshold_px=proximity_threshold_px,
    )

    annotated = annotate_detections(
        img_bgr, det, dense_flags=density["dense_flags"], clusters=density["dense_clusters"]
    )
    out_name = f"img_{uuid.uuid4().hex[:12]}.jpg"
    out_path = OUTPUTS_DIR / out_name
    cv2.imwrite(str(out_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    total_ms = (time.perf_counter() - t0) * 1000.0
    detections = det.to_list()
    for i, item in enumerate(detections):
        item["local_neighbors"] = density["neighbor_counts"][i]
        item["in_dense_zone"] = density["dense_flags"][i]
        if i < len(interbird["nearest_neighbor_distances"]):
            item["nearest_neighbor_px"] = interbird["nearest_neighbor_distances"][i]

    scores = det.scores
    conf_summary = {
        "mean": round(float(scores.mean()), 4) if len(scores) else None,
        "min": round(float(scores.min()), 4) if len(scores) else None,
        "max": round(float(scores.max()), 4) if len(scores) else None,
    }

    payload = {
        "source": source_label,
        "image_dimensions": {"width": det.image_width, "height": det.image_height},
        "parameters": {
            "model_name": model_name,
            "conf": conf,
            "iou": iou,
            "auto_enhance": auto_enhance,
            "force_clahe": force_clahe,
            "containment_dedup": containment_dedup,
            "proximity_threshold_px": proximity_threshold_px,
            "density_radius_px": density["radius_px"],
            "density_min_neighbors": density_min_neighbors,
        },
        "chicken_count": det.count,
        "clahe_applied": det.clahe_applied,
        "dedup_removed": det.dedup_removed,
        "brightness": round(det.brightness, 2),
        "model_used": detector.weights_path.name,
        "confidence_summary": conf_summary,
        "detections": detections,
        "density": density,
        "interbird_distance": interbird,
        "annotated_image_url": f"/api/outputs/{out_name}",
        "processing_time_ms": round(total_ms, 2),
        "inference_time_ms": round(det.inference_ms, 2),
    }

    latest_analyses["last_image"] = {
        "source": source_label,
        "model_used": detector.weights_path.name,
        "chicken_count": det.count,
        "dense_chickens": density["dense_chickens"],
        "dense_cluster_count": density["dense_cluster_count"],
        "clahe_applied": det.clahe_applied,
        "confidence_mean": conf_summary["mean"],
        "annotated_image_url": payload["annotated_image_url"],
        "processing_time_ms": payload["processing_time_ms"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return payload


@router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    model_name: str = Form("yolov8m"),
    auto_enhance: bool = Form(True),
    force_clahe: bool = Form(False),
    containment_dedup: bool = Form(True),
    proximity_threshold_px: float = Form(50.0),
    density_radius_px: Optional[float] = Form(None),
    density_min_neighbors: int = Form(DEFAULT_MIN_NEIGHBORS),
):
    filename = Path(file.filename or "upload.png").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTS)}",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty")
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds 35 MB limit")

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None or img_bgr.ndim != 3:
        raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as a valid image")

    return _run_image_pipeline(
        img_bgr=img_bgr,
        source_label=filename,
        conf=conf,
        iou=iou,
        density_radius_px=density_radius_px,
        density_min_neighbors=density_min_neighbors,
        model_name=model_name,
        auto_enhance=auto_enhance,
        force_clahe=force_clahe,
        containment_dedup=containment_dedup,
        proximity_threshold_px=proximity_threshold_px,
    )


def _resolve_safe_sample(sample_name: str) -> Path:
    clean_name = Path(sample_name).name
    if clean_name != sample_name or ".." in sample_name:
        raise HTTPException(status_code=400, detail="Invalid sample image name")
    target = (IMG_TEST_DIR / clean_name).resolve()
    if not target.is_file() or target.parent != IMG_TEST_DIR.resolve():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return target


@router.get("/samples/image/{sample_name}")
def get_sample_image(sample_name: str):
    target = _resolve_safe_sample(sample_name)
    return FileResponse(str(target))


@router.post("/detect/sample/{sample_name}")
def detect_sample_image(
    sample_name: str,
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    model_name: str = Form("yolov8m"),
    auto_enhance: bool = Form(True),
    force_clahe: bool = Form(False),
    containment_dedup: bool = Form(True),
    proximity_threshold_px: float = Form(50.0),
    density_radius_px: Optional[float] = Form(None),
    density_min_neighbors: int = Form(DEFAULT_MIN_NEIGHBORS),
):
    target = _resolve_safe_sample(sample_name)
    img_bgr = cv2.imread(str(target))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Could not read sample image")
    return _run_image_pipeline(
        img_bgr=img_bgr,
        source_label=target.name,
        conf=conf,
        iou=iou,
        density_radius_px=density_radius_px,
        density_min_neighbors=density_min_neighbors,
        model_name=model_name,
        auto_enhance=auto_enhance,
        force_clahe=force_clahe,
        containment_dedup=containment_dedup,
        proximity_threshold_px=proximity_threshold_px,
    )
