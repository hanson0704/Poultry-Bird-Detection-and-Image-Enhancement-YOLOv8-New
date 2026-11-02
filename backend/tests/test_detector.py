"""Tests for YOLOv8 ChickenDetector against backend/weights/best.pt and img_test/frame00097.png."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from app.core.detector import ChickenDetector, annotate_detections
from app.core.density import analyze_density

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGE = (
    (REPO_ROOT / "data" / "test_images" / "frame00097.png")
    if (REPO_ROOT / "data" / "test_images").is_dir()
    else (REPO_ROOT / "img_test" / "frame00097.png")
)
OUTPUTS_DIR = BACKEND_DIR / "outputs"


@pytest.fixture(scope="module")
def detector():
    return ChickenDetector()


def test_model_loads_and_reports_metadata(detector: ChickenDetector):
    info = detector.model_info
    assert info["weights_file"] == "best.pt"
    assert info["weights_size_mb"] > 40
    assert info["num_classes"] == 1
    assert 0 in info["class_names"]
    assert info["device"] in ("cuda:0", "cpu")
    print("Model info:", info)


def test_real_image_detection_and_annotation(detector: ChickenDetector):
    assert TEST_IMAGE.is_file(), f"Expected test image at {TEST_IMAGE}"
    img = cv2.imread(str(TEST_IMAGE))
    assert img is not None and img.ndim == 3

    det = detector.detect(img, conf=0.25)
    assert det.image_width == img.shape[1]
    assert det.image_height == img.shape[0]
    assert det.count > 0, "Expected chickens to be detected in frame00097.png"
    assert det.bboxes_xyxy.shape == (det.count, 4)
    assert det.bboxes_xywh.shape == (det.count, 4)
    assert det.centroids.shape == (det.count, 2)
    assert det.scores.shape == (det.count,)
    assert np.all((det.scores >= 0.25) & (det.scores <= 1.0))
    assert all(name.lower() == "chicken" for name in det.class_names)

    density = analyze_density(det.centroids, det.bboxes_xywh, det.image_width, det.image_height)
    assert density["count"] == det.count
    assert len(density["neighbor_counts"]) == det.count

    annotated = annotate_detections(img, det, dense_flags=density["dense_flags"],
                                    clusters=density["dense_clusters"])
    assert annotated.shape == img.shape
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "test_frame00097_annotated.jpg"
    ok = cv2.imwrite(str(out_path), annotated)
    assert ok and out_path.is_file() and out_path.stat().st_size > 10000
    print(f"Detected {det.count} chickens in frame00097.png (mean conf={det.scores.mean():.3f}, "
          f"dense chickens={density['dense_chickens']}, clusters={density['dense_cluster_count']})")


def test_empty_detections_on_blank_image(detector: ChickenDetector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    det = detector.detect(blank, conf=0.25)
    assert det.count == 0
    assert det.bboxes_xywh.shape == (0, 4)
    assert det.centroids.shape == (0, 2)
    assert det.to_list() == []
    annotated = annotate_detections(blank, det)
    assert annotated.shape == blank.shape


def test_threshold_validation(detector: ChickenDetector):
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.detect(blank, conf=0.0)
    with pytest.raises(ValueError):
        detector.detect(blank, conf=1.5)
    with pytest.raises(ValueError):
        detector.detect(None, conf=0.25)


def test_cpu_fallback_works():
    cpu_det = ChickenDetector(device="cpu")
    assert cpu_det.device == "cpu"
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    res = cpu_det.detect(blank, conf=0.25)
    assert res.count == 0


def test_yolov8n_lightweight_model_loads():
    n_det = ChickenDetector(model_name="yolov8n")
    info = n_det.model_info
    assert info["weights_file"] == "chicken_detector_yolov8n.pt"
    assert info["weights_size_mb"] < 15.0
    assert info["num_classes"] == 1


def test_auto_clahe_enhancement_triggers_on_dark_image(detector: ChickenDetector):
    # Dark synthetic image (mean brightness = 30 < 60)
    dark_img = np.full((320, 320, 3), 30, dtype=np.uint8)
    res = detector.detect(dark_img, auto_enhance=True)
    assert res.clahe_applied is True
    assert res.brightness < 60.0

    # Normal lit image (mean brightness = 120 > 60)
    lit_img = np.full((320, 320, 3), 120, dtype=np.uint8)
    res_lit = detector.detect(lit_img, auto_enhance=True)
    assert res_lit.clahe_applied is False
    assert res_lit.brightness >= 60.0


def test_force_clahe_flag(detector: ChickenDetector):
    lit_img = np.full((320, 320, 3), 120, dtype=np.uint8)
    res = detector.detect(lit_img, force_clahe=True)
    assert res.clahe_applied is True


