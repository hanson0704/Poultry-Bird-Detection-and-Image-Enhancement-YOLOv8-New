"""
YOLOv8 chicken detector (Ultralytics) — supports dual models (YOLOv8m and YOLOv8n),
automatic low-light CLAHE enhancement, and containment-based deduplication.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .enhancement import compute_brightness, enhance_low_light
from .deduplication import deduplicate_containment_indices

BACKEND_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BACKEND_DIR / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"

AVAILABLE_MODELS = {
    "yolov8m": {
        "filename": "best.pt",
        "description": "YOLOv8m Baseline (Accurate, 25.9M params)",
        "default": True,
    },
    "yolov8n": {
        "filename": "chicken_detector_yolov8n.pt",
        "description": "YOLOv8n Lightweight (Fast, 3.0M params)",
        "default": False,
    },
}


@dataclass
class DetectionResult:
    image_width: int
    image_height: int
    bboxes_xyxy: np.ndarray          # (N, 4) float
    bboxes_xywh: np.ndarray          # (N, 4) float, (xmin, ymin, w, h) — format used by the centroid tracker
    centroids: np.ndarray            # (N, 2) float
    scores: np.ndarray               # (N,) float
    class_ids: np.ndarray            # (N,) int
    class_names: List[str]
    inference_ms: float = 0.0
    clahe_applied: bool = False
    dedup_removed: int = 0
    brightness: float = 0.0

    @property
    def count(self) -> int:
        return int(len(self.scores))

    def to_list(self) -> List[Dict]:
        out = []
        for i in range(self.count):
            x1, y1, x2, y2 = (round(float(v), 1) for v in self.bboxes_xyxy[i])
            out.append({
                "id": i,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [round(float(v), 1) for v in self.bboxes_xywh[i]],
                "centroid": [round(float(v), 1) for v in self.centroids[i]],
                "confidence": round(float(self.scores[i]), 4),
                "class_id": int(self.class_ids[i]),
                "class_name": self.class_names[i],
            })
        return out


def validate_threshold(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive)")
    return value


def resolve_weights_path(weights_path: Optional[Path] = None, model_name: Optional[str] = None) -> Path:
    """
    Resolves model checkpoint path from explicit path, model key ('yolov8m' or 'yolov8n'),
    or environment variable. Defaults to backend/weights/best.pt.
    """
    if weights_path is not None:
        return Path(weights_path).resolve()

    if model_name:
        key = str(model_name).lower().strip()
        if key in AVAILABLE_MODELS:
            target = WEIGHTS_DIR / AVAILABLE_MODELS[key]["filename"]
            if target.is_file():
                return target.resolve()

    env_path = os.environ.get("CHICKEN_WEIGHTS")
    if env_path:
        return Path(env_path).resolve()

    return DEFAULT_WEIGHTS.resolve()


class ChickenDetector:
    def __init__(
        self,
        weights_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.weights_path = resolve_weights_path(weights_path=weights_path, model_name=model_name)
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path.name}")
        import torch
        from ultralytics import YOLO

        self._torch = torch
        self.model = YOLO(str(self.weights_path), task="detect")
        self.names: Dict[int, str] = dict(self.model.names)
        self.cuda_available = bool(torch.cuda.is_available())
        self.cuda_device_name = torch.cuda.get_device_name(0) if self.cuda_available else None
        self.device_warning: Optional[str] = None
        requested = (device or os.environ.get("CHICKEN_DEVICE", "auto")).lower()
        self.device = self._select_device(requested)
        self._lock = threading.Lock()

    def _select_device(self, requested: str) -> str:
        """Pick the device and VERIFY it with a real inference; fall back to CPU if CUDA inference fails."""
        candidates = []
        if requested in ("auto", "cuda", "0", "cuda:0") and self.cuda_available:
            candidates.append("cuda:0")
        candidates.append("cpu")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        last_err = None
        for dev in candidates:
            try:
                self.model.predict(dummy, device=dev, verbose=False, imgsz=320)
                return dev
            except Exception as exc:  # pragma: no cover - hardware dependent
                last_err = exc
                self.device_warning = f"Inference on {dev} failed ({type(exc).__name__}); falling back."
        raise RuntimeError(f"Model could not run on any device: {last_err}")

    @property
    def model_info(self) -> Dict:
        ckpt = getattr(self.model, "ckpt", None) or {}
        train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
        return {
            "weights_file": self.weights_path.name,
            "weights_size_mb": round(self.weights_path.stat().st_size / 1e6, 2),
            "task": getattr(self.model, "task", None),
            "class_names": self.names,
            "num_classes": len(self.names),
            "ultralytics_version_saved": ckpt.get("version") if isinstance(ckpt, dict) else None,
            "checkpoint_date": ckpt.get("date") if isinstance(ckpt, dict) else None,
            "base_model": train_args.get("model"),
            "train_imgsz": train_args.get("imgsz"),
            "train_epochs": train_args.get("epochs"),
            "run_name": train_args.get("name"),
            "device": self.device,
            "cuda_available": self.cuda_available,
            "cuda_device_name": self.cuda_device_name,
            "device_warning": self.device_warning,
        }

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        max_det: int = 1000,
        auto_enhance: bool = True,
        force_clahe: bool = False,
        containment_dedup: bool = True,
        iou_containment: float = 0.82,
        ratio_containment: float = 0.85,
    ) -> DetectionResult:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3)")
        conf = validate_threshold("conf", conf)
        iou = validate_threshold("iou", iou)
        h, w = image_bgr.shape[:2]

        # Automatic low-light enhancement check
        brightness = compute_brightness(image_bgr)
        apply_clahe = bool(force_clahe or (auto_enhance and brightness < 60.0))
        infer_frame = enhance_low_light(image_bgr, clip_limit=3.5) if apply_clahe else image_bgr

        kwargs = dict(conf=conf, iou=iou, device=self.device, verbose=False, max_det=max_det)
        if imgsz:
            kwargs["imgsz"] = int(imgsz)
        t0 = time.perf_counter()
        with self._lock:
            result = self.model.predict(infer_frame, **kwargs)[0]
        ms = (time.perf_counter() - t0) * 1000.0

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            empty4 = np.empty((0, 4), dtype=float)
            return DetectionResult(
                w, h, empty4, empty4.copy(), np.empty((0, 2)), np.empty((0,)),
                np.empty((0,), dtype=int), [], ms, clahe_applied=apply_clahe,
                dedup_removed=0, brightness=brightness,
            )

        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        scores = boxes.conf.cpu().numpy().astype(float)
        cls = boxes.cls.cpu().numpy().astype(int)

        # Bounding-box containment deduplication
        dedup_removed = 0
        if containment_dedup and len(xyxy) > 1:
            keep_indices = deduplicate_containment_indices(
                xyxy, scores,
                iou_threshold=iou_containment,
                containment_threshold=ratio_containment,
            )
            dedup_removed = len(xyxy) - len(keep_indices)
            if dedup_removed > 0:
                xyxy = xyxy[keep_indices]
                scores = scores[keep_indices]
                cls = cls[keep_indices]

        xywh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], axis=1)
        cents = np.stack([xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1)
        names = [self.names.get(int(c), str(int(c))) for c in cls]
        return DetectionResult(
            w, h, xyxy, xywh, cents, scores, cls, names, ms,
            clahe_applied=apply_clahe, dedup_removed=dedup_removed, brightness=brightness,
        )


# --------------------------------------------------------------------------- drawing
GREEN = (60, 200, 60)
ORANGE = (0, 145, 255)   # notebook's "High Dense" colour (BGR)
RED = (0, 0, 255)


def _thickness(img) -> int:
    return max(1, int(round(max(img.shape[:2]) / 900)))


def annotate_detections(
    image_bgr: np.ndarray,
    det: DetectionResult,
    dense_flags=None,
    clusters=None,
    show_conf: bool = True,
) -> np.ndarray:
    img = image_bgr.copy()
    t = _thickness(img)
    fs = 0.4 * t
    for cl in clusters or []:
        x1, y1, x2, y2 = (int(v) for v in cl["region_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, t + 1)
        cv2.putText(img, f"High density ({cl['size']})", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, ORANGE, t + 1, cv2.LINE_AA)
    for i in range(det.count):
        x1, y1, x2, y2 = (int(v) for v in det.bboxes_xyxy[i])
        color = ORANGE if dense_flags is not None and dense_flags[i] else GREEN
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
        if show_conf:
            cv2.putText(img, f"{det.scores[i]:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, color, t, cv2.LINE_AA)

    banner_text = f"Chickens: {det.count}"
    if det.clahe_applied:
        banner_text += " [CLAHE]"
    _banner(img, banner_text)
    return img


def _banner(img, text: str):
    t = _thickness(img)
    fs = 0.6 * t
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t + 1)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (30, 30, 30), -1)
    cv2.putText(img, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), t + 1, cv2.LINE_AA)
