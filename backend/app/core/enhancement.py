"""
Image illumination and low-light enhancement utilities.

Provides adaptive histogram equalization (CLAHE) on the Lightness (L) channel
in LAB color space, preserving natural chicken feather coloration while lifting
deep shadows typical in dim broiler facilities.
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_brightness(image_bgr: np.ndarray) -> float:
    """
    Computes the scalar mean pixel intensity across all channels.
    A value < 60.0 indicates dim/dark farm lighting conditions.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    return float(np.mean(image_bgr))


def is_low_light(image_bgr: np.ndarray, threshold: float = 60.0) -> bool:
    """
    Returns True if the image's mean brightness is strictly below the given threshold.
    """
    return compute_brightness(image_bgr) < threshold


def enhance_low_light(
    image_bgr: np.ndarray,
    clip_limit: float = 3.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the
    Lightness (L) channel of the LAB color space.

    Args:
        image_bgr: Input BGR image (H, W, 3).
        clip_limit: Contrast limit threshold for CLAHE (default: 3.5).
        tile_grid_size: Grid size for local histogram equalization (default: 8x8).

    Returns:
        Enhanced BGR image (H, W, 3) with enhanced shadow detail.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a valid BGR image with shape (H, W, 3)")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def adjust_gamma(image_bgr: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """
    Applies non-linear gamma adjustment using an 8-bit lookup table.
    gamma > 1.0 lifts dark shadows.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a valid BGR image with shape (H, W, 3)")
    if gamma <= 0:
        raise ValueError("Gamma must be strictly positive")

    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image_bgr, table)

