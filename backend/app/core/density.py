"""
Local crowding / density analysis.

Adapted from ``Broiler_Density_Detection.ipynb``:
  * build ``scipy.spatial.KDTree`` over detected chickens
  * neighbours of each chicken = detections within ``radius`` pixels (``query_ball_tree``),
    count INCLUDING the chicken itself (as in the notebook)
  * a chicken is in a "high density" area when that count >= ``min_neighbors`` (notebook: 6)

Changes relative to the notebook:
  * uses bounding-box CENTROIDS instead of top-left corners;
  * the radius default is derived from the median chicken box size in the image
    (the notebook's fixed 100 px only made sense on its 480x480 resized frames);
  * dense chickens are grouped into connected clusters with bounding regions.

Values are image-based (pixels / counts). They are NOT chickens per square metre:
no physical calibration of the camera is available.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import KDTree

DEFAULT_MIN_NEIGHBORS = 6        # notebook: Density >= 6 (count includes the bird itself)
DEFAULT_RADIUS_FACTOR = 2.0      # radius = factor x median box diagonal


def auto_radius(bboxes_xywh: np.ndarray, factor: float = DEFAULT_RADIUS_FACTOR) -> Optional[float]:
    if len(bboxes_xywh) == 0:
        return None
    diag = np.hypot(bboxes_xywh[:, 2], bboxes_xywh[:, 3])
    return float(np.median(diag) * factor)


def analyze_density(centroids: np.ndarray, bboxes_xywh: np.ndarray, image_w: int, image_h: int,
                    radius_px: Optional[float] = None, min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
                    radius_factor: float = DEFAULT_RADIUS_FACTOR) -> Dict:
    centroids = np.asarray(centroids, dtype=float).reshape(-1, 2)
    bboxes_xywh = np.asarray(bboxes_xywh, dtype=float).reshape(-1, 4)
    n = len(centroids)
    if min_neighbors < 2:
        raise ValueError("min_neighbors must be >= 2")
    radius_source = "user"
    if radius_px is None:
        radius_px = auto_radius(bboxes_xywh, radius_factor)
        radius_source = f"auto ({radius_factor} x median box diagonal)"
    elif radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    image_area = float(image_w * image_h)
    coverage = float(np.sum(bboxes_xywh[:, 2] * bboxes_xywh[:, 3]) / image_area) if n and image_area else 0.0
    base = {
        "count": n,
        "radius_px": round(radius_px, 1) if radius_px else None,
        "radius_source": radius_source,
        "min_neighbors": min_neighbors,
        "chickens_per_megapixel": round(n / (image_area / 1e6), 2) if image_area else 0.0,
        "box_area_coverage_ratio": round(min(coverage, 1.0), 4),
        "unit_note": "Image-based measures (pixels). Not calibrated to physical area.",
    }
    if n == 0:
        return {**base, "neighbor_counts": [], "dense_flags": [], "dense_chickens": 0,
                "dense_clusters": [], "dense_cluster_count": 0, "max_neighbor_count": 0}

    tree = KDTree(centroids)
    neighbors = tree.query_ball_tree(tree, r=radius_px)
    counts = np.array([len(nb) for nb in neighbors])  # includes self
    dense = counts >= min_neighbors

    # Group dense chickens into connected clusters (neighbour graph restricted to dense points).
    clusters: List[Dict] = []
    visited = np.zeros(n, dtype=bool)
    for start in np.where(dense)[0]:
        if visited[start]:
            continue
        stack, members = [start], []
        visited[start] = True
        while stack:
            i = stack.pop()
            members.append(int(i))
            for j in neighbors[i]:
                if dense[j] and not visited[j]:
                    visited[j] = True
                    stack.append(j)
        b = bboxes_xywh[members]
        x1, y1 = float(b[:, 0].min()), float(b[:, 1].min())
        x2, y2 = float((b[:, 0] + b[:, 2]).max()), float((b[:, 1] + b[:, 3]).max())
        clusters.append({
            "members": sorted(members),
            "size": len(members),
            "region_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "center": [round(float(centroids[members, 0].mean()), 1), round(float(centroids[members, 1].mean()), 1)],
        })
    clusters.sort(key=lambda c: -c["size"])

    return {
        **base,
        "neighbor_counts": [int(c - 1) for c in counts],  # excluding self
        "dense_flags": [bool(d) for d in dense],
        "dense_chickens": int(dense.sum()),
        "dense_clusters": clusters,
        "dense_cluster_count": len(clusters),
        "max_neighbor_count": int(counts.max() - 1),
    }

