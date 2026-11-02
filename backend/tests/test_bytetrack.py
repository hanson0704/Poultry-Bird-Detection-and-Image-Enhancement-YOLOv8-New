"""
Unit tests for ByteTrack multi-object tracking and tracker regression.
"""
import numpy as np

from app.core.tracker import ByteTrackerAdapter, ChickenTracker, TrackObservation


def test_bytetrack_adapter_initialization_and_update():
    adapter = ByteTrackerAdapter(track_thresh=0.25)
    assert adapter.tracker is not None

    # Empty frame
    res_empty = adapter.update(
        np.empty((0, 4), dtype=float),
        np.empty((0,), dtype=float),
        np.empty((0,), dtype=int),
    )
    assert res_empty == []

    # Frame 1: 2 chickens
    bboxes_f1 = np.array([
        [50.0, 50.0, 40.0, 40.0],
        [200.0, 200.0, 50.0, 50.0],
    ], dtype=float)
    scores = np.array([0.90, 0.85], dtype=float)
    class_ids = np.array([0, 0], dtype=int)

    tracks_f1 = adapter.update(bboxes_f1, scores, class_ids)
    assert len(tracks_f1) >= 1

    # Frame 2: slightly moved chickens
    bboxes_f2 = np.array([
        [52.0, 51.0, 40.0, 40.0],
        [202.0, 201.0, 50.0, 50.0],
    ], dtype=float)

    tracks_f2 = adapter.update(bboxes_f2, scores, class_ids)
    assert len(tracks_f2) == 2
    track_ids = {t[0] for t in tracks_f2}
    assert len(track_ids) == 2  # Two distinct track IDs


def test_chicken_tracker_unified_interface_bytetrack():
    tracker = ChickenTracker(tracker_type="bytetrack")
    assert tracker.tracker_type == "bytetrack"

    # Feed 3 consecutive frames with small movement
    for frame_idx in range(1, 4):
        offset = float(frame_idx * 2)
        bboxes = np.array([[100.0 + offset, 100.0 + offset, 40.0, 40.0]])
        scores = np.array([0.92])
        classes = np.array([0])

        obs = tracker.update(frame_idx, bboxes, scores, classes)
        assert len(obs) >= 1
        assert isinstance(obs[0], TrackObservation)
        assert obs[0].track_id > 0
        # Kalman filter smooths trajectory; verify centroid is within 2 px of measurement
        assert abs(obs[0].centroid[0] - (100.0 + offset + 20.0)) < 2.0

    # Check that the same bird kept its ID across frames
    assert tracker.total_tracks == 1
    assert 1 in tracker.history
    assert tracker.history[1].observations >= 2


def test_chicken_tracker_centroid_fallback_regression():
    tracker = ChickenTracker(tracker_type="centroid", max_lost=5)
    assert tracker.tracker_type == "centroid"

    bboxes = np.array([[100.0, 100.0, 40.0, 40.0]])
    scores = np.array([0.90])
    classes = np.array([0])

    obs = tracker.update(1, bboxes, scores, classes)
    assert len(obs) == 1
    assert obs[0].track_id == 0

    # Frame 2 with empty detections
    obs_empty = tracker.update(2, np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int))
    assert len(obs_empty) == 0
    assert tracker.total_tracks == 1
