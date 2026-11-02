"""Deterministic synthetic tests for tracking, movement, density and inactivity modules."""
import numpy as np
import pytest

from app.core.density import analyze_density
from app.core.distance import MovementAnalyzer
from app.core.inactivity import InactivityAnalyzer
from app.core.tracker import ChickenTracker, GatedCentroidTracker, OriginalCentroidTracker

FPS = 25.0
BOX = 40.0  # synthetic chicken box size (px)


def run(frames, fps=FPS, window_s=10.0, threshold_px=5.0, metric="extent", max_distance=None,
        jump_px=30.0):
    """frames: list of lists of (cx, cy) per frame. Returns (tracker, movement, inactivity, per-frame obs)."""
    trk = ChickenTracker(max_lost=5, max_distance=max_distance)
    mov = MovementAnalyzer(jump_threshold_px=jump_px, max_gap_frames=6)
    ina = InactivityAnalyzer(fps=fps, window_seconds=window_s, threshold_px=threshold_px, metric=metric)
    all_obs = []
    for f, centers in enumerate(frames):
        boxes = np.array([[cx - BOX / 2, cy - BOX / 2, BOX, BOX] for cx, cy in centers]).reshape(-1, 4)
        obs = trk.update(f, boxes, np.full(len(boxes), 0.9), np.zeros(len(boxes), dtype=int))
        mov.update(f, obs)
        ina.evaluate(mov.tracks, f, force=(f == len(frames) - 1))
        all_obs.append(obs)
    return trk, mov, ina, all_obs


def test_continuous_movement_not_flagged():
    frames = [[(100 + 3 * f, 200)] for f in range(300)]
    trk, mov, ina, _ = run(frames)
    assert trk.total_tracks == 1
    tm = mov.tracks[0]
    assert tm.total_distance_px == pytest.approx(3 * 299, rel=1e-6)
    assert tm.rejected_jumps == 0
    rep = ina.track_report(0)
    assert rep["evaluated_windows"] > 0
    assert rep["low_mobility_indicator"] is False


def test_stationary_track_flagged_only_after_full_window():
    frames = [[(300, 300)] for _ in range(300)]
    _, mov, ina, _ = run(frames)
    assert mov.tracks[0].total_distance_px == 0.0
    rep = ina.track_report(0)
    assert rep["low_mobility_indicator"] is True
    # window = 250 frames => the first eligible evaluation is at frame >= 249
    first_period_start, first_period_end = rep["low_mobility_periods_s"][0]
    assert first_period_end >= 250 / FPS


def test_stationary_before_full_window_is_insufficient_history():
    frames = [[(300, 300)] for _ in range(200)]  # 8 s < 10 s window
    _, _, ina, _ = run(frames)
    rep = ina.track_report(0)
    assert rep["evaluated_windows"] == 0
    assert rep["low_mobility_indicator"] is False
    assert rep["final_status"] == "insufficient_history"


def test_track_appearing_mid_window_is_not_flagged_early():
    # A moving bird for the whole clip and a stationary bird that appears at frame 100.
    frames = []
    for f in range(300):
        c = [(50 + 2 * f, 50)]
        if f >= 100:
            c.append((600, 600))
        frames.append(c)
    _, mov, ina, _ = run(frames)
    stationary_id = [tid for tid, tm in mov.tracks.items() if tm.first_frame == 100][0]
    rep = ina.track_report(stationary_id)
    assert rep["low_mobility_indicator"] is False   # only 200 frames of history at the end
    assert rep["final_status"] == "insufficient_history"

    frames += [[(50 + 2 * f, 50), (600, 600)] for f in range(300, 400)]
    _, mov, ina, _ = run(frames)
    stationary_id = [tid for tid, tm in mov.tracks.items() if tm.first_frame == 100][0]
    assert ina.track_report(stationary_id)["low_mobility_indicator"] is True


def test_missing_observations_keep_id_and_distance():
    frames = []
    for f in range(300):
        frames.append([] if 150 <= f <= 152 else [(200 + 0.5 * f, 200)])
    trk, mov, ina, _ = run(frames)
    assert trk.total_tracks == 1, "short gap (<= max_lost) must not create a new ID"
    tm = mov.tracks[0]
    # displacement across the gap (4 frames * 0.5 px) is still accumulated
    assert tm.total_distance_px == pytest.approx(0.5 * 299, rel=1e-6)
    assert tm.observations == 297


def test_implausible_jump_is_rejected_by_movement_filter():
    frames = [[(100 + f, 100)] for f in range(50)]
    frames += [[(400 + f, 100)] for f in range(50, 100)]  # sudden ~250 px jump at frame 50
    trk, mov, _, _ = run(frames, max_distance=None)
    assert trk.total_tracks == 1  # ungated original behaviour: re-associated to the same ID
    tm = mov.tracks[0]
    assert tm.rejected_jumps == 1
    assert tm.total_distance_px == pytest.approx(98, rel=1e-6)


def test_distance_gate_creates_new_track_on_jump():
    frames = [[(100 + f, 100)] for f in range(50)]
    frames += [[(400 + f, 100)] for f in range(50, 100)]
    trk, _, _, obs = run(frames, max_distance=60)
    assert trk.total_tracks == 2
    assert obs[49][0].track_id != obs[50][0].track_id


def test_empty_detections():
    trk, mov, ina, obs = run([[] for _ in range(10)])
    assert trk.total_tracks == 0
    assert all(o == [] for o in obs)
    assert mov.summary(FPS)["tracks"] == 0
    assert ina.summary()["tracks_evaluated"] == 0
    d = analyze_density(np.empty((0, 2)), np.empty((0, 4)), 640, 480)
    assert d["count"] == 0 and d["dense_cluster_count"] == 0


def test_gated_tracker_matches_original_when_ungated():
    rng = np.random.default_rng(0)
    orig = OriginalCentroidTracker(max_lost=5)
    gated = GatedCentroidTracker(max_lost=5, max_distance=None)
    base = rng.uniform(0, 500, size=(8, 2))
    for f in range(60):
        pts = base + rng.normal(0, 2, size=base.shape) + f
        keep = rng.random(len(pts)) > 0.1
        boxes = np.hstack([pts[keep] - 20, np.full((keep.sum(), 2), 40.0)])
        a = orig.update(boxes, np.ones(len(boxes)), np.zeros(len(boxes), dtype=int))
        b = gated.update(boxes, np.ones(len(boxes)), np.zeros(len(boxes), dtype=int))
        assert [(t[1], tuple(t[2:6])) for t in a] == [(t[1], tuple(t[2:6])) for t in b]


def test_density_uses_centroids_and_finds_cluster():
    cluster = [(100 + 15 * (i % 3), 100 + 15 * (i // 3)) for i in range(7)]
    far = [(600, 100), (100, 600), (600, 600)]
    cents = np.array(cluster + far, dtype=float)
    boxes = np.hstack([cents - 20, np.full((len(cents), 2), 40.0)])
    d = analyze_density(cents, boxes, 800, 800, radius_px=60, min_neighbors=6)
    assert d["count"] == 10
    assert d["dense_chickens"] == 7
    assert d["dense_cluster_count"] == 1
    assert d["dense_clusters"][0]["size"] == 7
    assert d["dense_flags"][-3:] == [False, False, False]
    # neighbour counts exclude self
    assert d["neighbor_counts"][-1] == 0


def test_inactivity_parameter_validation():
    with pytest.raises(ValueError):
        InactivityAnalyzer(fps=0, window_seconds=10, threshold_px=5)
    with pytest.raises(ValueError):
        InactivityAnalyzer(fps=25, window_seconds=10, threshold_px=5, metric="bogus")

