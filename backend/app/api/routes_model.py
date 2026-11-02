"""Health, model diagnostics, and overview routes."""
from __future__ import annotations

import csv
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.detector import AVAILABLE_MODELS
from app.state import (
    IMG_TEST_DIR,
    WEIGHTS_DIR,
    get_detector,
    get_health_payload,
    latest_analyses,
)

router = APIRouter(prefix="/api", tags=["system"])


def _load_optional_training_metrics() -> Dict:
    results_csv = WEIGHTS_DIR / "results.csv"
    if not results_csv.is_file():
        return {
            "available": False,
            "note": "Optional training log (results.csv) is not present in backend/weights/. Inference requires only weights checkpoints.",
            "history": [],
        }
    rows: List[Dict] = []
    with results_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                if k is None:
                    continue
                key = k.strip()
                try:
                    cleaned[key] = float(v.strip()) if v is not None else None
                except ValueError:
                    cleaned[key] = v.strip() if v is not None else None
            if cleaned:
                rows.append(cleaned)
    return {
        "available": len(rows) > 0,
        "epochs_recorded": len(rows),
        "latest_epoch": rows[-1] if rows else None,
        "history": rows,
    }


@router.get("/health")
def health_check():
    return get_health_payload()


@router.get("/model/info")
def model_info(model_name: Optional[str] = Query("yolov8m")):
    try:
        det = get_detector(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model initialization error: {exc}")

    available_list = []
    for k, cfg in AVAILABLE_MODELS.items():
        w_path = WEIGHTS_DIR / cfg["filename"]
        available_list.append({
            "key": k,
            "filename": cfg["filename"],
            "description": cfg["description"],
            "default": cfg.get("default", False),
            "installed": w_path.is_file(),
            "file_size_mb": round(w_path.stat().st_size / 1e6, 2) if w_path.is_file() else None,
        })

    return {
        "model": det.model_info,
        "available_models": available_list,
        "training_metrics": _load_optional_training_metrics(),
    }


@router.get("/overview")
def overview():
    samples = []
    if IMG_TEST_DIR.is_dir():
        samples = sorted(
            [p.name for p in IMG_TEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )[:12]
    return {
        "health": get_health_payload(),
        "latest_image_analysis": latest_analyses.get("last_image"),
        "latest_video_analysis": latest_analyses.get("last_video"),
        "sample_images": samples,
    }
