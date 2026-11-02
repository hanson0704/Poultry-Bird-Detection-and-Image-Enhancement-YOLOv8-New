"""
Bounding box deduplication and containment suppression utilities.

Filters out spurious nested bounding boxes (where a detector places multiple
concentric or overlapping boxes on the same chicken body) using both standard
Intersection-over-Union (IoU) and containment ratio (Intersection / Min-Area).
"""
from __future__ import annotations

from typing import List
import numpy as np


def deduplicate_containment_indices(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.82,
    containment_threshold: float = 0.85,
) -> List[int]:
    """
    Returns indices of detections to retain after suppressing redundant nested boxes.

    A lower-scoring candidate box is suppressed if:
    1. Standard IoU with a higher-scoring retained box >= iou_threshold (default: 0.82), OR
    2. Containment (intersection area / candidate box area) >= containment_threshold (default: 0.85).

    Args:
        boxes_xyxy: (N, 4) float array of [x1, y1, x2, y2].
        scores: (N,) float array of detection confidence scores.
        iou_threshold: Maximum allowable IoU overlap.
        containment_threshold: Maximum allowable containment ratio of candidate inside kept box.

    Returns:
        List of integer indices to keep, in decreasing order of confidence.
    """
    n = len(boxes_xyxy)
    if n <= 1:
        return list(range(n))

    order = np.argsort(-scores)
    keep_indices: List[int] = []

    for idx in order:
        cand = boxes_xyxy[idx]
        c_area = max(1.0, float((cand[2] - cand[0]) * (cand[3] - cand[1])))
        is_duplicate = False

        for k_idx in keep_indices:
            kept = boxes_xyxy[k_idx]
            k_area = max(1.0, float((kept[2] - kept[0]) * (kept[3] - kept[1])))

            inter_x1 = max(cand[0], kept[0])
            inter_y1 = max(cand[1], kept[1])
            inter_x2 = min(cand[2], kept[2])
            inter_y2 = min(cand[3], kept[3])

            inter_w = max(0.0, float(inter_x2 - inter_x1))
            inter_h = max(0.0, float(inter_y2 - inter_y1))
            inter_area = inter_w * inter_h

            if inter_area <= 0.0:
                continue

            union_area = c_area + k_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            containment = inter_area / min(c_area, k_area)

            if iou >= iou_threshold or containment >= containment_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep_indices.append(int(idx))

    return keep_indices

