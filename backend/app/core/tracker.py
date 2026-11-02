"""
Multi-object tracking adapter for YOLOv8 detections.

Supports two tracking engines:
1. `bytetrack` (Ultralytics ByteTrack with Kalman filter motion prediction and two-stage
   association; recommended default for crowded poultry videos).
2. `centroid` (Original greedy `CentroidTracker` from the vendored `multi-object-tracker`
   library with optional distance gating).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance

# Ensure SciPy Hungarian LAP shim before importing Ultralytics trackers
from app.core.lap_shim import ensure_scipy_lap_shim
ensure_scipy_lap_shim()

REPO_ROOT = Path(__file__).resolve().parents[3]
MOTRACKERS_DIR = (
    REPO_ROOT / "research" / "legacy" / "multi-object-tracker" / "motrackers"
    if (REPO_ROOT / "research" / "legacy" / "multi-object-tracker" / "motrackers").is_dir()
    else REPO_ROOT / "multi-object-tracker" / "motrackers"
)


def _load_original_motrackers():
    """Load motrackers.tracker.Tracker from the original files without running package __init__s."""
    if "motrackers.tracker" in sys.modules and hasattr(sys.modules["motrackers.tracker"], "Tracker"):
        return sys.modules["motrackers.tracker"].Tracker
    if not MOTRACKERS_DIR.is_dir():
        raise ImportError(f"Original tracker library not found at {MOTRACKERS_DIR}")

    # Lightweight package shells so 'from motrackers.x import y' resolves to the original files.
    for pkg_name, pkg_dir in (("motrackers", MOTRACKERS_DIR), ("motrackers.utils", MOTRACKERS_DIR / "utils")):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            sys.modules[pkg_name] = pkg

    def _load(mod_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    _load("motrackers.utils.misc", MOTRACKERS_DIR / "utils" / "misc.py")
    _load("motrackers.kalman_tracker", MOTRACKERS_DIR / "kalman_tracker.py")
    _load("motrackers.track", MOTRACKERS_DIR / "track.py")
    tracker_mod = _load("motrackers.tracker", MOTRACKERS_DIR / "tracker.py")
    return tracker_mod.Tracker


OriginalCentroidTracker = _load_original_motrackers()
from motrackers.utils.misc import get_centroid  # noqa: E402  (original helper)


class GatedCentroidTracker(OriginalCentroidTracker):
    """Original greedy centroid tracker + optional distance gate (``max_distance`` in pixels)."""

    def __init__(self, max_lost: int = 5, max_distance: Optional[float] = None,
                 tracker_output_format: str = "mot_challenge"):
        super().__init__(max_lost=max_lost, tracker_output_format=tracker_output_format)
        self.max_distance = max_distance

    def update(self, bboxes, detection_scores, class_ids):
        self.frame_count += 1

        if len(bboxes) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
            return self._get_tracks(self.tracks)

        detections = OriginalCentroidTracker.preprocess_input(bboxes, class_ids, detection_scores)
        track_ids = list(self.tracks.keys())
        updated_tracks, updated_detections = [], []

        if len(track_ids):
            track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
            detection_centroids = get_centroid(np.asarray(bboxes, dtype=float))
            centroid_distances = distance.cdist(track_centroids, detection_centroids)
            track_indices = np.amin(centroid_distances, axis=1).argsort()

            for idx in track_indices:
                track_id = track_ids[idx]
                matched = False
                remaining = [(i, d) for (i, d) in enumerate(centroid_distances[idx, :])
                             if i not in updated_detections]
                if remaining:
                    det_idx, det_dist = min(remaining, key=lambda x: x[1])
                    if self.max_distance is None or det_dist <= self.max_distance:
                        bbox, class_id, confidence = detections[det_idx]
                        self._update_track(track_id, self.frame_count, bbox, confidence, class_id=class_id)
                        updated_detections.append(det_idx)
                        updated_tracks.append(track_id)
                        matched = True
                if not matched:
                    self.tracks[track_id].lost += 1
                    if self.tracks[track_id].lost > self.max_lost:
                        self._remove_track(track_id)

        for i, (bbox, class_id, confidence) in enumerate(detections):
            if i not in updated_detections:
                self._add_track(self.frame_count, bbox, confidence, class_id=class_id)

        return self._get_tracks(self.tracks)


@dataclass
class TrackObservation:
    track_id: int
    bbox_xywh: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float
    class_id: int


@dataclass
class TrackHistory:
    track_id: int
    first_frame: int
    last_frame: int
    # (frame_index, cx, cy) for every frame where the track was observed (bounded length)
    points: Deque[Tuple[int, float, float]] = field(default_factory=lambda: deque(maxlen=5000))
    observations: int = 0


class ByteTrackerAdapter:
    """Adapter wrapping Ultralytics BYTETracker into the common poultry tracking interface."""

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        import torch
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._torch = torch
        self._Boxes = Boxes

        args = SimpleNamespace(
            track_high_thresh=float(track_thresh),
            track_low_thresh=max(0.05, float(track_thresh) * 0.4),
            new_track_thresh=float(track_thresh),
            track_buffer=int(track_buffer),
            match_thresh=float(match_thresh),
            fuse_score=True,
        )
        self.tracker = BYTETracker(args)

    def update(
        self,
        bboxes_xywh: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        img_shape: Tuple[int, int] = (640, 640),
    ) -> List[Tuple[int, Tuple[float, float, float, float], float, int]]:
        """
        Updates ByteTrack with detections.
        Returns list of (track_id, bbox_xywh, confidence, class_id).
        """
        n = len(bboxes_xywh)
        if n == 0:
            empty_boxes = self._Boxes(self._torch.empty((0, 6)), orig_shape=img_shape)
            self.tracker.update(empty_boxes)
            return []

        xywh = np.asarray(bboxes_xywh, dtype=np.float32).reshape(-1, 4)
        x1 = xywh[:, 0]
        y1 = xywh[:, 1]
        x2 = x1 + xywh[:, 2]
        y2 = y1 + xywh[:, 3]
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        c = np.asarray(class_ids, dtype=np.float32).reshape(-1)

        t_data = np.column_stack([x1, y1, x2, y2, s, c])
        tensor_boxes = self._torch.from_numpy(t_data).float()
        boxes = self._Boxes(tensor_boxes, orig_shape=img_shape)

        tracked_arr = self.tracker.update(boxes)
        results = []
        seen_tids = set()

        if len(tracked_arr) > 0:
            # Columns: [x1, y1, x2, y2, track_id, score, cls, idx]
            for row in tracked_arr:
                rx1, ry1, rx2, ry2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                tid = int(row[4])
                score = float(row[5])
                cls_id = int(row[6])
                rw = max(1.0, rx2 - rx1)
                rh = max(1.0, ry2 - ry1)
                results.append((tid, (rx1, ry1, rw, rh), score, cls_id))
                seen_tids.add(tid)

        # Include active/newly-activated stracks so initial frame birds are never dropped
        if hasattr(self.tracker, "tracked_stracks"):
            for st in self.tracker.tracked_stracks:
                tid = int(st.track_id)
                if tid not in seen_tids and (st.is_activated or getattr(st, "state", 0) == 1):
                    tlwh = st.tlwh
                    rx1, ry1, rw, rh = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
                    score = float(st.score)
                    cls_id = int(st.cls) if hasattr(st, "cls") else 0
                    results.append((tid, (rx1, ry1, rw, rh), score, cls_id))
                    seen_tids.add(tid)

        return results


class ChickenTracker:
    """
    Unified multi-object tracker supporting both 'centroid' and 'bytetrack'.
    Maintains persistent track histories, displacement coordinates, and unique track counts.
    """

    def __init__(
        self,
        tracker_type: str = "centroid",
        max_lost: int = 5,
        max_distance: Optional[float] = None,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        history_len: int = 5000,
    ):
        self.tracker_type = tracker_type.lower().strip() if tracker_type else "centroid"
        self.history: Dict[int, TrackHistory] = {}
        self.history_len = history_len

        if self.tracker_type == "centroid":
            self._centroid_tracker = GatedCentroidTracker(max_lost=max_lost, max_distance=max_distance)
            self._byte_tracker = None
        else:
            try:
                self._byte_tracker = ByteTrackerAdapter(
                    track_thresh=track_thresh,
                    track_buffer=max_lost,
                    match_thresh=match_thresh,
                )
                self._centroid_tracker = None
                self.tracker_type = "bytetrack"
            except Exception as exc:  # pragma: no cover
                # Graceful fallback to GatedCentroidTracker if ByteTrack fails to initialize
                print(f"[WARNING] ByteTrackerAdapter failed to initialize ({exc}); falling back to centroid.")
                self._centroid_tracker = GatedCentroidTracker(max_lost=max_lost, max_distance=max_distance)
                self._byte_tracker = None
                self.tracker_type = "centroid"

    def update(
        self,
        frame_index: int,
        bboxes_xywh: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        img_shape: Tuple[int, int] = (640, 640),
    ) -> List[TrackObservation]:
        observations: List[TrackObservation] = []
        bboxes_xywh = np.asarray(bboxes_xywh, dtype=float).reshape(-1, 4)

        if self.tracker_type == "centroid" and self._centroid_tracker is not None:
            raw_tracks = self._centroid_tracker.update(
                bboxes_xywh, np.asarray(scores, dtype=float), np.asarray(class_ids, dtype=int)
            )
            for trk in raw_tracks:
                tid = int(trk[1])
                x, y, w, h = float(trk[2]), float(trk[3]), float(trk[4]), float(trk[5])
                cx, cy = x + 0.5 * w, y + 0.5 * h
                cls_id = int(self._centroid_tracker.tracks[tid].class_id)
                obs = TrackObservation(tid, (x, y, w, h), (cx, cy), float(trk[6]), cls_id)
                observations.append(obs)
                self._record_history(tid, frame_index, cx, cy)

        elif self._byte_tracker is not None:
            raw_byte_tracks = self._byte_tracker.update(
                bboxes_xywh, scores, class_ids, img_shape=img_shape
            )
            for tid, (x, y, w, h), conf, cls_id in raw_byte_tracks:
                cx, cy = x + 0.5 * w, y + 0.5 * h
                obs = TrackObservation(tid, (x, y, w, h), (cx, cy), conf, cls_id)
                observations.append(obs)
                self._record_history(tid, frame_index, cx, cy)

        return observations

    def _record_history(self, tid: int, frame_index: int, cx: float, cy: float):
        hist = self.history.get(tid)
        if hist is None:
            hist = TrackHistory(tid, frame_index, frame_index, deque(maxlen=self.history_len))
            self.history[tid] = hist
        hist.points.append((frame_index, cx, cy))
        hist.last_frame = frame_index
        hist.observations += 1

    @property
    def total_tracks(self) -> int:
        return len(self.history)
