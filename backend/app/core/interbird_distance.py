"""
Inter-bird spatial proximity and nearest-neighbor distance (NND) analysis.

Calculates instantaneous spatial dispersion metrics between chickens in a single
frame using pairwise Euclidean centroid distance matrices.

Distances are measured strictly in pixels (px) since farm cameras lack calibrated
extrinsic real-world ground planes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.spatial.distance import cdist


def analyze_interbird_distance(
    centroids: np.ndarray | List[List[float]],
    proximity_threshold_px: float = 50.0,
    max_pair_reports: int = 50,
) -> Dict[str, Any]:
    """
    Computes pairwise spatial distances and nearest-neighbor dispersion across all chickens.

    Args:
        centroids: (N, 2) array of (cx, cy) pixel coordinates.
        proximity_threshold_px: Threshold under which two birds are considered in close contact
                                or potential crowding/pecking proximity (default: 50.0 px).
        max_pair_reports: Maximum number of close-contact pairs to enumerate in detail.

    Returns:
        Structured dictionary with dispersion metrics, NND distribution, and close contact alerts.
    """
    pts = np.asarray(centroids, dtype=np.float64).reshape(-1, 2) if len(centroids) else np.empty((0, 2))
    n = len(pts)

    if n == 0:
        return {
            "chicken_count": 0,
            "unit": "pixels",
            "mean_nearest_neighbor_px": None,
            "min_interbird_distance_px": None,
            "max_interbird_distance_px": None,
            "proximity_threshold_px": float(proximity_threshold_px),
            "close_contact_pair_count": 0,
            "close_contact_pairs": [],
            "nearest_neighbor_distances": [],
            "spacing_histogram": {"bins": [], "counts": []},
            "dispersion_status": "No birds detected",
        }

    if n == 1:
        return {
            "chicken_count": 1,
            "unit": "pixels",
            "mean_nearest_neighbor_px": None,
            "min_interbird_distance_px": None,
            "max_interbird_distance_px": None,
            "proximity_threshold_px": float(proximity_threshold_px),
            "close_contact_pair_count": 0,
            "close_contact_pairs": [],
            "nearest_neighbor_distances": [None],
            "spacing_histogram": {"bins": [], "counts": []},
            "dispersion_status": "Single bird (isolated)",
        }

    # Pairwise Euclidean Distance Matrix (N x N)
    dist_matrix = cdist(pts, pts, metric="euclidean")

    # Upper triangle indices (i < j) to avoid self-distances and double counting
    upper_r, upper_c = np.triu_indices(n, k=1)
    pair_distances = dist_matrix[upper_r, upper_c]

    min_dist = float(np.min(pair_distances))
    max_dist = float(np.max(pair_distances))

    # Close contact pairs below proximity threshold
    close_mask = pair_distances <= proximity_threshold_px
    close_count = int(np.sum(close_mask))

    close_pairs: List[Dict[str, Any]] = []
    if close_count > 0:
        close_indices = np.where(close_mask)[0]
        # Sort by distance (closest first)
        sorted_close = close_indices[np.argsort(pair_distances[close_indices])]
        for idx in sorted_close[:max_pair_reports]:
            close_pairs.append({
                "bird_a_id": int(upper_r[idx]),
                "bird_b_id": int(upper_c[idx]),
                "distance_px": round(float(pair_distances[idx]), 1),
            })

    # Nearest-Neighbor Distance (NND) per chicken
    dist_no_diag = dist_matrix.copy()
    np.fill_diagonal(dist_no_diag, np.inf)
    nnd_array = np.min(dist_no_diag, axis=1)

    mean_nnd = float(np.mean(nnd_array))
    median_nnd = float(np.median(nnd_array))

    # Nearest-neighbor distance histogram for UI visualization
    num_bins = min(8, max(3, n // 3))
    counts, bin_edges = np.histogram(nnd_array, bins=num_bins)
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i+1])} px"
        for i in range(len(counts))
    ]

    # Qualitative flock dispersion status
    if close_count >= (n // 2):
        status = "Clustered / High Proximity"
    elif mean_nnd < proximity_threshold_px:
        status = "Dense Spacing"
    else:
        status = "Evenly Dispersed"

    return {
        "chicken_count": n,
        "unit": "pixels",
        "mean_nearest_neighbor_px": round(mean_nnd, 1),
        "median_nearest_neighbor_px": round(median_nnd, 1),
        "min_interbird_distance_px": round(min_dist, 1),
        "max_interbird_distance_px": round(max_dist, 1),
        "proximity_threshold_px": float(proximity_threshold_px),
        "close_contact_pair_count": close_count,
        "close_contact_pairs": close_pairs,
        "nearest_neighbor_distances": [round(float(v), 1) for v in nnd_array],
        "spacing_histogram": {
            "bins": bin_labels,
            "counts": [int(c) for c in counts],
        },
        "dispersion_status": status,
    }

