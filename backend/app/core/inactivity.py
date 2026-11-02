"""
Inactivity / Low-Mobility Indicator.

Adapted from ``Broiler_Sick_Detection.ipynb``. The notebook accumulated each track's
centroid displacement and, every 250 frames (10 s at 25 fps), highlighted tracks whose
accumulated displacement was ``0 < TDP <= 5`` px (480x480 frames), then reset the accumulator.

Problems in the notebook logic that are fixed here:
  * ``my_tracks[index, 7] = 0`` used ``index`` from a previous loop iteration when a track
    was new in the evaluation frame (wrong row reset, or NameError on the first track);
  * windows were fixed at multiples of 250 frames, so a track that appeared mid-window was
    judged on a partial window, i.e. it could be flagged without enough observations;
  * the accumulator restarted whenever a track was missed for a frame;
  * exactly 0 px movement was never flagged (``0 < TDP``).

This implementation evaluates a trailing time window (default 10 s) at a regular stride.
A track is only evaluated when it already existed at the window start AND it was observed
in at least ``min_coverage`` of the window's frames. Otherwise it is reported as
"insufficient history" and never flagged.

This is a motion-based heuristic. It is NOT a disease or health classifier: resting,
sleeping or feeding birds also move very little.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

INDICATOR_LABEL = "Inactivity / Low-Mobility Indicator"
METRICS = ("path", "extent")


@dataclass
class TrackInactivityState:
    evaluations: int = 0
    flagged_evaluations: int = 0
    currently_flagged: bool = False
    last_value_px: Optional[float] = None
    last_coverage: Optional[float] = None
    status: str = "insufficient_history"
    episodes: List[List[float]] = field(default_factory=list)  # [start_s, end_s] windows merged


class InactivityAnalyzer:
    def __init__(self, fps: float, window_seconds: float, threshold_px: float,
                 metric: str = "extent", min_coverage: float = 0.8, stride_seconds: float = 1.0):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if threshold_px < 0:
            raise ValueError("threshold_px must be >= 0")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        self.fps = float(fps)
        self.window_seconds = float(window_seconds)
        self.window_frames = max(2, int(round(window_seconds * fps)))
        self.stride_frames = max(1, int(round(stride_seconds * fps)))
        self.threshold_px = float(threshold_px)
        self.metric = metric
        self.min_coverage = float(min_coverage)
        self.states: Dict[int, TrackInactivityState] = {}

    # ------------------------------------------------------------------ core
    def window_stats(self, steps, current_frame: int) -> Tuple[float, float, float, int]:
        """Return (path_px, extent_px, coverage, n_obs) for the window (current_frame - W, current_frame]."""
        start = current_frame - self.window_frames
        pts = []
        for item in reversed(steps):  # steps are in frame order; stop once we leave the window
            if item[0] <= start:
                break
            if item[0] <= current_frame:
                pts.append(item)
        pts.reverse()
        n = len(pts)
        if n == 0:
            return 0.0, 0.0, 0.0, 0
        path = sum(s for (f, s, _, _) in pts[1:])  # first step's displacement starts before the window
        mx = sum(p[2] for p in pts) / n
        my = sum(p[3] for p in pts) / n
        extent = max(math.hypot(p[2] - mx, p[3] - my) for p in pts)
        return path, extent, n / self.window_frames, n

    def evaluate(self, movement_tracks: Dict, current_frame: int, force: bool = False) -> None:
        """Evaluate all tracks at ``current_frame`` (only every ``stride_frames`` unless forced)."""
        if not force and current_frame % self.stride_frames != 0:
            return
        window_start = current_frame - self.window_frames + 1
        for tid, tm in movement_tracks.items():
            st = self.states.setdefault(tid, TrackInactivityState())
            if tm.last_frame < window_start:
                # Track not seen during this window (left the scene or lost) — keep last verdict, stop flagging.
                st.currently_flagged = False
                st.status = "not_visible"
                continue
            if tm.first_frame > window_start:
                st.currently_flagged = False
                st.status = "insufficient_history"
                continue
            path, extent, coverage, _ = self.window_stats(tm.steps, current_frame)
            st.last_coverage = round(coverage, 3)
            if coverage < self.min_coverage:
                st.currently_flagged = False
                st.status = "insufficient_coverage"
                continue
            value = path if self.metric == "path" else extent
            st.last_value_px = round(value, 2)
            st.evaluations += 1
            flagged = value <= self.threshold_px
            st.currently_flagged = flagged
            st.status = "low_mobility" if flagged else "active"
            if flagged:
                st.flagged_evaluations += 1
                w_start_s = round(max(0, window_start) / self.fps, 2)
                w_end_s = round((current_frame + 1) / self.fps, 2)
                if st.episodes and w_start_s <= st.episodes[-1][1]:
                    st.episodes[-1][1] = w_end_s
                else:
                    st.episodes.append([w_start_s, w_end_s])

    def is_flagged(self, track_id: int) -> bool:
        st = self.states.get(track_id)
        return bool(st and st.currently_flagged)

    # --------------------------------------------------------------- reports
    def track_report(self, track_id: int) -> Dict:
        st = self.states.get(track_id, TrackInactivityState())
        ever = st.flagged_evaluations > 0
        return {
            "evaluated_windows": st.evaluations,
            "low_mobility_windows": st.flagged_evaluations,
            "low_mobility_indicator": ever,
            "final_status": st.status,
            "last_window_value_px": st.last_value_px,
            "low_mobility_periods_s": st.episodes,
        }

    def summary(self) -> Dict:
        evaluated = [s for s in self.states.values() if s.evaluations > 0]
        flagged = [tid for tid, s in self.states.items() if s.flagged_evaluations > 0]
        return {
            "label": INDICATOR_LABEL,
            "disclaimer": ("Motion-based heuristic only. Low movement can be normal resting, sleeping or "
                           "feeding behaviour. This is not a veterinary diagnosis."),
            "metric": self.metric,
            "metric_description": ("total centroid path length within the window" if self.metric == "path"
                                   else "max distance of the centroid from its mean position within the window"),
            "window_seconds": self.window_seconds,
            "window_frames": self.window_frames,
            "threshold_px": round(self.threshold_px, 2),
            "min_coverage": self.min_coverage,
            "tracks_evaluated": len(evaluated),
            "tracks_with_low_mobility": len(flagged),
            "low_mobility_track_ids": sorted(flagged),
        }
