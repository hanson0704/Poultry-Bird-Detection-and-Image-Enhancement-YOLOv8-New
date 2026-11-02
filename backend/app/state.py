"""Application state: multi-model detector registry, directories, job progress, and telemetry."""
from __future__ import annotations

import platform
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

from app.core.detector import AVAILABLE_MODELS, ChickenDetector

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
WEIGHTS_DIR = BACKEND_DIR / "weights"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
TEMP_UPLOADS_DIR = BACKEND_DIR / "temp_uploads"
IMG_TEST_DIR = (
    REPO_ROOT / "data" / "test_images"
    if (REPO_ROOT / "data" / "test_images").is_dir()
    else REPO_ROOT / "img_test"
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_detector_lock = threading.Lock()
_detector_registry: Dict[str, ChickenDetector] = {}
_detector_error: Optional[str] = None

# Track active video jobs and latest completed analyses (in-memory, real runs only)
video_jobs: Dict[str, Dict] = {}
latest_analyses: Dict[str, Optional[Dict]] = {
    "last_image": None,
    "last_video": None,
}


def get_detector(model_name: Optional[str] = None) -> ChickenDetector:
    """
    Returns a cached ChickenDetector instance for the requested model ('yolov8m' or 'yolov8n').
    Defaults to 'yolov8m' (best.pt).
    """
    global _detector_registry, _detector_error
    key = str(model_name).lower().strip() if model_name else "yolov8m"
    if key not in AVAILABLE_MODELS:
        key = "yolov8m"

    with _detector_lock:
        if key in _detector_registry:
            return _detector_registry[key]
        try:
            det = ChickenDetector(model_name=key)
            _detector_registry[key] = det
            _detector_error = None
            return det
        except Exception as exc:
            _detector_error = f"{type(exc).__name__}: {exc}"
            raise


def get_health_payload() -> Dict:
    import torch
    import ultralytics
    import cv2

    det = None
    try:
        det = get_detector("yolov8m")
    except Exception:
        det = None

    return {
        "status": "ok" if det is not None else "degraded",
        "model_loaded": det is not None,
        "model_error": _detector_error,
        "inference_device": det.device if det is not None else None,
        "active_model": det.weights_path.name if det is not None else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
    }
