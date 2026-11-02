"""
Unit tests for bounding-box containment and duplicate suppression.
"""
import numpy as np

from app.core.deduplication import deduplicate_containment_indices


def test_deduplicate_empty_or_single_box():
    empty_boxes = np.empty((0, 4), dtype=float)
    empty_scores = np.empty((0,), dtype=float)
    assert deduplicate_containment_indices(empty_boxes, empty_scores) == []

    single_box = np.array([[10, 10, 50, 50]], dtype=float)
    single_score = np.array([0.9], dtype=float)
    assert deduplicate_containment_indices(single_box, single_score) == [0]


def test_deduplicate_suppresses_nested_box():
    # Outer box: chicken body [10, 10, 100, 100] (area = 8100), conf = 0.90
    # Nested box: head/torso [20, 20, 80, 80] (area = 3600), conf = 0.75
    # The nested box is 100% contained in the outer box (containment = 1.0 > 0.85).
    # Disjoint box: another chicken [150, 150, 220, 220], conf = 0.85
    boxes = np.array([
        [10, 10, 100, 100],
        [20, 20, 80, 80],
        [150, 150, 220, 220],
    ], dtype=float)
    scores = np.array([0.90, 0.75, 0.85], dtype=float)

    keep = deduplicate_containment_indices(boxes, scores, containment_threshold=0.85)

    # Box 0 and Box 2 must be kept; Box 1 must be suppressed as a nested duplicate
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep
    assert len(keep) == 2


def test_deduplicate_keeps_partially_overlapping_distinct_birds():
    # Two adjacent birds in a dense flock with moderate IoU (e.g., IoU ~ 0.35, containment ~ 0.50)
    boxes = np.array([
        [50, 50, 100, 100],   # Bird A: (50x50, area 2500)
        [75, 50, 125, 100],   # Bird B: (50x50, area 2500, overlap 25x50=1250, containment 0.5)
    ], dtype=float)
    scores = np.array([0.88, 0.84], dtype=float)

    keep = deduplicate_containment_indices(
        boxes, scores, iou_threshold=0.82, containment_threshold=0.85
    )

    # Both birds should be preserved
    assert len(keep) == 2
    assert set(keep) == {0, 1}

