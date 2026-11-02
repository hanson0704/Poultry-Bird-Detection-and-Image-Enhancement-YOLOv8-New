"""
Frame-by-frame video behaviour analysis pipeline.

Streams frames from OpenCV VideoCapture, runs YOLOv8 detection, updates the
selected tracker (ByteTrack or CentroidTracker), updates movement/distance (TDP),
evaluates the sliding-window Inactivity / Low-Mobility Indicator, computes KDTree
spatial density, calculates inter-bird spatial proximity (NND), and writes an H.264
MP4 (yuv420p + faststart) incrementally so the entire video is never held in RAM.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.density import DEFAULT_MIN_NEIGHBORS, analyze_density
from app.core.detector import ChickenDetector, GREEN, ORANGE, RED, _banner, _thickness
from app.core.distance import MovementAnalyzer, default_jump_threshold
from app.core.inactivity import INDICATOR_LABEL, InactivityAnalyzer
from app.core.interbird_distance import analyze_interbird_distance
from app.core.tracker import ChickenTracker


class _FFmpegVideoWriter:
    """Incremental H.264 MP4 writer using imageio-ffmpeg's bundled binary, with OpenCV fallback."""

    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.output_path = Path(output_path)
        # libx264 with yuv420p requires even dimensions
        self.width = width if width % 2 == 0 else width - 1
        self.height = height if height % 2 == 0 else height - 1
        self.fps = max(1.0, float(fps))
        self._proc: Optional[subprocess.Popen] = None
        self._cv_writer: Optional[cv2.VideoWriter] = None
        self.codec_used = "unknown"
        self._open()

    def _open(self):
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", f"{self.fps:.3f}",
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "23",
                "-movflags", "+faststart",
                str(self.output_path),
            ]
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.codec_used = "h264 (libx264/yuv420p)"
            return
        except Exception:
            self._proc = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not self._cv_writer.isOpened():
            raise RuntimeError("Could not initialize video writer (neither ffmpeg nor OpenCV mp4v succeeded)")
        self.codec_used = "mp4v (OpenCV fallback)"

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame_bgr.tobytes())
                return
            except BrokenPipeError:
                self._proc = None
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._cv_writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
                self.codec_used = "mp4v (OpenCV fallback)"
        if self._cv_writer is not None:
            self._cv_writer.write(frame_bgr)

    def close(self):
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=30)
            finally:
                self._proc = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None


def process_video(
    video_path: Path,
    output_path: Path,
    detector: ChickenDetector,
    conf: float = 0.25,
    iou: float = 0.45,
    tracker_type: str = "bytetrack",
    auto_enhance: bool = True,
    containment_dedup: bool = True,
    proximity_threshold_px: float = 50.0,
    max_lost: int = 30,
    max_distance: Optional[float] = None,
    jump_threshold_px: Optional[float] = None,
    inactivity_window_s: float = 10.0,
    inactivity_thresh_px: Optional[float] = None,
    inactivity_metric: str = "extent",
    density_radius_px: Optional[float] = None,
    density_min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
    max_frames: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Uploaded file could not be opened as a readable video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_assumed = False
    if fps <= 0 or fps > 240 or np.isnan(fps):
        fps = 25.0
        fps_assumed = True

    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise ValueError("Video contains no readable frames")

    h, w = first_frame.shape[:2]
    scale = max(w, h) / 480.0
    if jump_threshold_px is None:
        jump_threshold_px = default_jump_threshold(w, h)
    if inactivity_thresh_px is None:
        inactivity_thresh_px = round(5.0 * scale, 2)

    tracker = ChickenTracker(
        tracker_type=tracker_type,
        max_lost=max_lost,
        max_distance=max_distance,
        track_thresh=conf,
    )
    movement = MovementAnalyzer(jump_threshold_px=jump_threshold_px, max_gap_frames=max_lost + 1)
    inactivity = InactivityAnalyzer(
        fps=fps,
        window_seconds=inactivity_window_s,
        threshold_px=inactivity_thresh_px,
        metric=inactivity_metric,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _FFmpegVideoWriter(output_path, w, h, fps)

    counts_per_frame: List[int] = []
    dense_chickens_per_frame: List[int] = []
    dense_clusters_per_frame: List[int] = []
    mean_nnd_per_frame: List[float] = []
    close_contact_pairs_per_frame: List[int] = []
    timeline_samples: List[Dict] = []
    sample_stride = max(1, int(round(fps)))  # ~1 sample per second

    t0 = time.perf_counter()
    frame_idx = 0
    target_total = min(total_reported, max_frames) if (total_reported > 0 and max_frames) else (max_frames or total_reported)

    try:
        current = first_frame
        while current is not None:
            if max_frames is not None and frame_idx >= max_frames:
                break

            det = detector.detect(
                current,
                conf=conf,
                iou=iou,
                auto_enhance=auto_enhance,
                containment_dedup=containment_dedup,
            )
            obs = tracker.update(frame_idx, det.bboxes_xywh, det.scores, det.class_ids, img_shape=(w, h))
            movement.update(frame_idx, obs)

            obs_centroids = np.array([o.centroid for o in obs], dtype=float).reshape(-1, 2)
            obs_bboxes = np.array([o.bbox_xywh for o in obs], dtype=float).reshape(-1, 4)

            dens = analyze_density(
                obs_centroids, obs_bboxes, w, h,
                radius_px=density_radius_px, min_neighbors=density_min_neighbors
            )

            interbird = analyze_interbird_distance(
                obs_centroids, proximity_threshold_px=proximity_threshold_px
            )

            inactivity.evaluate(movement.tracks, frame_idx)

            counts_per_frame.append(len(obs))
            dense_chickens_per_frame.append(dens["dense_chickens"])
            dense_clusters_per_frame.append(dens["dense_cluster_count"])
            if interbird["mean_nearest_neighbor_px"] is not None:
                mean_nnd_per_frame.append(interbird["mean_nearest_neighbor_px"])
            close_contact_pairs_per_frame.append(interbird["close_contact_pair_count"])

            if frame_idx % sample_stride == 0:
                flagged_now = sum(1 for o in obs if inactivity.is_flagged(o.track_id))
                timeline_samples.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 2),
                    "chicken_count": len(obs),
                    "dense_chickens": dens["dense_chickens"],
                    "dense_clusters": dens["dense_cluster_count"],
                    "mean_nearest_neighbor_px": interbird["mean_nearest_neighbor_px"],
                    "close_contact_pairs": interbird["close_contact_pair_count"],
                    "low_mobility_active": flagged_now,
                })

            annotated = _draw_tracked_frame(
                current, frame_idx, fps, obs, tracker, movement, inactivity, dens, interbird
            )
            writer.write(annotated)

            frame_idx += 1
            if progress_cb is not None:
                progress_cb(frame_idx, int(target_total or frame_idx))

            ok, nxt = cap.read()
            current = nxt if ok else None
    finally:
        cap.release()
        writer.close()

    # Final evaluation on the last processed frame
    if frame_idx > 0:
        inactivity.evaluate(movement.tracks, frame_idx - 1, force=True)

    elapsed = time.perf_counter() - t0
    warnings: List[str] = []
    if fps_assumed:
        warnings.append("Video metadata did not report a valid FPS; defaulted to 25.0 FPS.")
    if frame_idx < inactivity.window_frames:
        warnings.append(
            f"Video duration ({frame_idx / fps:.1f}s, {frame_idx} frames) is shorter than the "
            f"configured inactivity window ({inactivity.window_seconds:.1f}s, {inactivity.window_frames} frames); "
            f"no track accumulated enough history for inactivity evaluation."
        )
    if sum(counts_per_frame) == 0:
        warnings.append("No chickens were detected in any frame at the selected confidence threshold.")

    track_summaries = []
    for tid, tm in sorted(movement.tracks.items()):
        dur_s = round((tm.last_frame - tm.first_frame + 1) / fps, 2)
        inact_rep = inactivity.track_report(tid)
        track_summaries.append({
            "track_id": tid,
            "first_frame": tm.first_frame,
            "last_frame": tm.last_frame,
            "duration_s": dur_s,
            "observations": tm.observations,
            "total_distance_px": round(tm.total_distance_px, 1),
            "mean_speed_px_per_s": round(tm.total_distance_px / dur_s, 2) if dur_s > 0 else 0.0,
            "rejected_jumps": tm.rejected_jumps,
            "inactivity": inact_rep,
            "trajectory": tm.trajectory[-150:],
        })

    tracking_note = (
        "ByteTrack association with Kalman filter velocity prediction and SciPy bipartite matching."
        if tracker.tracker_type == "bytetrack" else
        "Centroid-based greedy association (motrackers.CentroidTracker)."
    )

    return {
        "frames_processed": frame_idx,
        "fps": round(fps, 2),
        "duration_s": round(frame_idx / fps, 2),
        "resolution": {"width": writer.width, "height": writer.height},
        "codec": writer.codec_used,
        "tracker_used": tracker.tracker_type,
        "processing_time_s": round(elapsed, 2),
        "processing_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0.0,
        "total_unique_tracks": tracker.total_tracks,
        "counts_summary": {
            "mean_chickens_per_frame": round(float(np.mean(counts_per_frame)), 2) if counts_per_frame else 0.0,
            "max_chickens_in_frame": int(max(counts_per_frame)) if counts_per_frame else 0,
            "min_chickens_in_frame": int(min(counts_per_frame)) if counts_per_frame else 0,
        },
        "movement_summary": movement.summary(fps),
        "inactivity_summary": inactivity.summary(),
        "density_summary": {
            "mean_dense_chickens": round(float(np.mean(dense_chickens_per_frame)), 2) if dense_chickens_per_frame else 0.0,
            "max_dense_chickens": int(max(dense_chickens_per_frame)) if dense_chickens_per_frame else 0,
            "frames_with_dense_clusters": int(sum(1 for c in dense_clusters_per_frame if c > 0)),
            "min_neighbors": density_min_neighbors,
            "unit_note": "Image-space crowding (pixels/counts), not calibrated to physical area.",
        },
        "interbird_distance_summary": {
            "mean_flock_nearest_neighbor_px": round(float(np.mean(mean_nnd_per_frame)), 1) if mean_nnd_per_frame else None,
            "mean_close_contact_pairs": round(float(np.mean(close_contact_pairs_per_frame)), 1) if close_contact_pairs_per_frame else 0.0,
            "max_close_contact_pairs": int(max(close_contact_pairs_per_frame)) if close_contact_pairs_per_frame else 0,
            "proximity_threshold_px": float(proximity_threshold_px),
            "unit": "pixels",
            "unit_note": "Pairwise inter-bird distance between chicken centroids in image pixels.",
        },
        "track_summaries": track_summaries,
        "timeline": timeline_samples,
        "warnings": warnings,
        "tracking_note": tracking_note,
    }


def _draw_tracked_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    fps: float,
    observations,
    tracker: ChickenTracker,
    movement: MovementAnalyzer,
    inactivity: InactivityAnalyzer,
    density: Dict,
    interbird: Optional[Dict] = None,
) -> np.ndarray:
    img = frame_bgr.copy()
    t = _thickness(img)
    fs = 0.38 * t

    for cl in density.get("dense_clusters", []):
        x1, y1, x2, y2 = (int(v) for v in cl["region_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, t + 1)
        cv2.putText(img, f"Dense cluster ({cl['size']})", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, ORANGE, t + 1, cv2.LINE_AA)

    dense_flags = density.get("dense_flags", [])
    flagged_count = 0
    for idx, obs in enumerate(observations):
        tid = obs.track_id
        x, y, w, h = (int(round(v)) for v in obs.bbox_xywh)
        cx, cy = int(round(obs.centroid[0])), int(round(obs.centroid[1]))
        is_inact = inactivity.is_flagged(tid)
        is_dense = idx < len(dense_flags) and dense_flags[idx]
        if is_inact:
            color = RED
            flagged_count += 1
        elif is_dense:
            color = ORANGE
        else:
            color = GREEN

        # Draw recent centroid trail
        hist = tracker.history.get(tid)
        if hist and len(hist.points) > 1:
            pts = np.array([[int(round(px)), int(round(py))] for (_, px, py) in list(hist.points)[-25:]],
                           dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=max(1, t))

        cv2.rectangle(img, (x, y), (x + w, y + h), color, t)
        cv2.circle(img, (cx, cy), max(2, t + 1), color, -1)

        tm = movement.tracks.get(tid)
        tdp = tm.total_distance_px if tm else 0.0
        label = f"ID {tid} | {tdp:.0f}px"
        if is_inact:
            label += " [LOW MOB]"
        cv2.putText(img, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, t, cv2.LINE_AA)

    nnd_str = f"NND: {interbird['mean_nearest_neighbor_px']}px" if (interbird and interbird.get("mean_nearest_neighbor_px") is not None) else "NND: —"
    banner = (f"Frame {frame_idx} ({frame_idx / fps:.1f}s) | Chickens: {len(observations)} | "
              f"Dense: {density.get('dense_chickens', 0)} | {nnd_str} | Low-Mobility: {flagged_count}")
    _banner(img, banner)
    return img
