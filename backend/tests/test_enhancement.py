"""
Unit tests for low-light CLAHE enhancement and illumination utilities.
"""
import numpy as np
import pytest

from app.core.enhancement import adjust_gamma, compute_brightness, enhance_low_light, is_low_light


def test_compute_brightness_and_low_light_detection():
    # Pitch black image
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert compute_brightness(dark_img) == 0.0
    assert is_low_light(dark_img, threshold=60.0) is True

    # Dim farm image (mean 45.0)
    dim_img = np.full((100, 100, 3), 45, dtype=np.uint8)
    assert compute_brightness(dim_img) == 45.0
    assert is_low_light(dim_img, threshold=60.0) is True

    # Well-lit farm image (mean 120.0)
    lit_img = np.full((100, 100, 3), 120, dtype=np.uint8)
    assert compute_brightness(lit_img) == 120.0
    assert is_low_light(lit_img, threshold=60.0) is False


def test_enhance_low_light_increases_contrast_and_preserves_shape():
    # Synthetic gradient dim image
    np.random.seed(42)
    dim_gradient = np.random.randint(10, 50, size=(128, 128, 3), dtype=np.uint8)
    enhanced = enhance_low_light(dim_gradient, clip_limit=3.5)

    assert enhanced.shape == dim_gradient.shape
    assert enhanced.dtype == np.uint8
    # CLAHE lifts shadow regions, increasing the mean intensity of dim inputs
    assert float(np.mean(enhanced)) >= float(np.mean(dim_gradient))


def test_enhance_low_light_invalid_input_handling():
    with pytest.raises(ValueError):
        enhance_low_light(None)  # type: ignore

    with pytest.raises(ValueError):
        enhance_low_light(np.zeros((64, 64), dtype=np.uint8))  # 2D instead of 3D


def test_adjust_gamma():
    img = np.full((50, 50, 3), 50, dtype=np.uint8)
    brightened = adjust_gamma(img, gamma=1.5)
    assert float(np.mean(brightened)) > float(np.mean(img))

    with pytest.raises(ValueError):
        adjust_gamma(img, gamma=-1.0)

