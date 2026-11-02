"""
Unit tests for inter-bird spatial distance and dispersion analysis.
"""
import numpy as np

from app.core.interbird_distance import analyze_interbird_distance


def test_empty_and_single_chicken_cases():
    # 0 chickens
    res_zero = analyze_interbird_distance([])
    assert res_zero["chicken_count"] == 0
    assert res_zero["mean_nearest_neighbor_px"] is None
    assert res_zero["close_contact_pair_count"] == 0
    assert res_zero["close_contact_pairs"] == []

    # 1 chicken
    res_one = analyze_interbird_distance([[100.0, 100.0]])
    assert res_one["chicken_count"] == 1
    assert res_one["mean_nearest_neighbor_px"] is None
    assert res_one["close_contact_pair_count"] == 0


def test_two_chickens_pairwise_distance():
    # Bird 0 at (0, 0), Bird 1 at (30, 40) => Euclidean distance = 50.0 px
    centroids = np.array([[0.0, 0.0], [30.0, 40.0]], dtype=float)
    res = analyze_interbird_distance(centroids, proximity_threshold_px=60.0)

    assert res["chicken_count"] == 2
    assert res["mean_nearest_neighbor_px"] == 50.0
    assert res["min_interbird_distance_px"] == 50.0
    assert res["max_interbird_distance_px"] == 50.0
    # Proximity threshold is 60.0, so 50.0 <= 60.0 triggers 1 close contact pair
    assert res["close_contact_pair_count"] == 1
    assert len(res["close_contact_pairs"]) == 1
    assert res["close_contact_pairs"][0]["bird_a_id"] == 0
    assert res["close_contact_pairs"][0]["bird_b_id"] == 1
    assert res["close_contact_pairs"][0]["distance_px"] == 50.0


def test_distinct_pair_counting_and_no_duplicates():
    # 3 chickens in a line:
    # A at (0, 0), B at (20, 0), C at (100, 0)
    # Distances:
    # A-B = 20
    # B-C = 80
    # A-C = 100
    centroids = [[0.0, 0.0], [20.0, 0.0], [100.0, 0.0]]
    res = analyze_interbird_distance(centroids, proximity_threshold_px=50.0)

    assert res["chicken_count"] == 3
    # Only A-B is <= 50 px
    assert res["close_contact_pair_count"] == 1
    assert res["min_interbird_distance_px"] == 20.0
    assert res["max_interbird_distance_px"] == 100.0
    # Nearest neighbor distances:
    # For A: 20 (to B)
    # For B: 20 (to A)
    # For C: 80 (to B)
    # Mean NND = (20 + 20 + 80) / 3 = 40.0 px
    assert np.isclose(res["mean_nearest_neighbor_px"], 40.0)
    assert len(res["spacing_histogram"]["bins"]) > 0

