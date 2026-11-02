"""
Movement / distance analysis based on tracked centroids.

Adapted from ``Broiler_Distance_Measure.ipynb``:
  * per-step displacement = Euclidean distance between consecutive centroids of the same track ID
  * steps larger than a jump threshold are treated as tracker re-association noise and
    counted as 0 (the notebook used ``if Distance > 30: Distance = 0`` on 480x480 frames)
  * cumulative displacement per track = "Total Distance in Pixels" (TDP)

Fixes relative to the notebook:
  * a track that is missed for a few frames keeps its accumulated distance (the notebook
    restarted it from -1 because the MOT tuple column 7 is -1 for every new output row);
  * the jump threshold is scaled by the number of frames elapsed since the last observation;
  * the jump threshold default is scaled to the analysed frame resolution.

All values are in image PIXELS. No camera calibration is available, so no metric units.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import math

# The notebook used a 30 px jump filter on frames resized to 480x480.
REFERENCE_FRAME_SIZE = 480.0
REFERENCE_JUMP_PX = 30.0


def default_jump_threshold(frame_w: int, frame_h: int) -> float:
    return REFERENCE_JUMP_PX * max(frame_w, frame_h) / REFERENCE_FRAME_SIZE


@dataclass
class TrackMovement:
    track_id: int
    first_frame: int
    last_frame: int
    last_centroid: Tuple[float, float]
    total_distance_px: float = 0.0
    accepted_steps: int = 0
    rejected_jumps: int = 0
    observations: int = 1
    # (frame_index, accepted_step_px, cx, cy) — bounded history used by the inactivity analyser
    steps: Deque[Tuple[int, float, float, float]] = field(default_factory=lambda: deque(maxlen=20000))
    trajectory: List[Tuple[int, float, float]] = field(default_factory=list)


class MovementAnalyzer:
    def __init__(self, jump_threshold_px: float, max_gap_frames: int = 6,
                 trajectory_stride: int = 1, max_trajectory_points: int = 2000):
        if jump_threshold_px <= 0:
            raise ValueError("jump_threshold_px must be > 0")
        self.jump_threshold_px = float(jump_threshold_px)
        self.max_gap_frames = max(1, int(max_gap_frames))
        self.trajectory_stride = max(1, int(trajectory_stride))
        self.max_trajectory_points = max_trajectory_points
        self.tracks: Dict[int, TrackMovement] = {}

    def update(self, frame_index: int, observations) -> Dict[int, float]:
        """observations: iterable with ``track_id`` and ``centroid``. Returns {track_id: step_px} for this frame."""
        step_by_track: Dict[int, float] = {}
        for obs in observations:
            tid = int(obs.track_id)
            cx, cy = float(obs.centroid[0]), float(obs.centroid[1])
            tm = self.tracks.get(tid)
            if tm is None:
                tm = TrackMovement(tid, frame_index, frame_index, (cx, cy))
                tm.steps.append((frame_index, 0.0, cx, cy))
                tm.trajectory.append((frame_index, round(cx, 1), round(cy, 1)))
                self.tracks[tid] = tm
                step_by_track[tid] = 0.0
                continue

            gap = max(1, frame_index - tm.last_frame)
            step = math.hypot(cx - tm.last_centroid[0], cy - tm.last_centroid[1])
            allowed = self.jump_threshold_px * min(gap, self.max_gap_frames)
            if step > allowed:
                tm.rejected_jumps += 1
                step = 0.0
            else:
                tm.accepted_steps += 1
            tm.total_distance_px += step
            tm.last_centroid = (cx, cy)
            tm.last_frame = frame_index
            tm.observations += 1
            tm.steps.append((frame_index, step, cx, cy))
            if (frame_index - tm.first_frame) % self.trajectory_stride == 0 and \
                    len(tm.trajectory) < self.max_trajectory_points:
                tm.trajectory.append((frame_index, round(cx, 1), round(cy, 1)))
            step_by_track[tid] = step
        return step_by_track

    def summary(self, fps: float) -> Dict:
        distances = [t.total_distance_px for t in self.tracks.values()]
        n = len(distances)
        durations = [(t.last_frame - t.first_frame + 1) / fps for t in self.tracks.values()] if fps > 0 else []
        speeds = [t.total_distance_px / d for t, d in zip(self.tracks.values(), durations) if d > 0]
        return {
            "unit": "pixels",
            "tracks": n,
            "total_distance_px": round(sum(distances), 2),
            "mean_distance_px": round(sum(distances) / n, 2) if n else 0.0,
            "max_distance_px": round(max(distances), 2) if n else 0.0,
            "mean_speed_px_per_s": round(sum(speeds) / len(speeds), 2) if speeds else 0.0,
            "rejected_jumps": int(sum(t.rejected_jumps for t in self.tracks.values())),
            "jump_threshold_px": round(self.jump_threshold_px, 2),
        }

