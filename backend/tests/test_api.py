"""End-to-end automated API tests for FastAPI endpoints."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGE = (
    (REPO_ROOT / "data" / "test_images" / "frame00097.png")
    if (REPO_ROOT / "data" / "test_images").is_dir()
    else (REPO_ROOT / "img_test" / "frame00097.png")
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["inference_device"] in ("cuda:0", "cpu")


def test_model_info_endpoint(client: TestClient):
    r = client.get("/api/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model"]["weights_file"] == "best.pt"
    assert data["model"]["num_classes"] == 1
    assert "0" in data["model"]["class_names"] or 0 in data["model"]["class_names"]
    # Optional training metrics should be reported honestly
    assert "available" in data["training_metrics"]


def test_detect_real_image_and_serve_annotated(client: TestClient):
    with TEST_IMAGE.open("rb") as f:
        r = client.post(
            "/api/detect/image",
            files={"file": ("frame00097.png", f, "image/png")},
            data={"conf": "0.25", "density_min_neighbors": "6"},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["chicken_count"] == len(data["detections"])
    assert data["chicken_count"] > 100
    assert data["density"]["count"] == data["chicken_count"]
    assert data["annotated_image_url"].startswith("/api/outputs/img_")

    # Fetch the annotated image via the output endpoint
    out_r = client.get(data["annotated_image_url"])
    assert out_r.status_code == 200
    assert out_r.headers["content-type"] == "image/jpeg"
    assert len(out_r.content) > 10000


def test_detect_blank_image_zero_detections(client: TestClient):
    blank = np.zeros((320, 480, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", blank)
    assert ok
    r = client.post(
        "/api/detect/image",
        files={"file": ("blank.png", enc.tobytes(), "image/png")},
        data={"conf": "0.25"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["chicken_count"] == 0
    assert data["detections"] == []
    assert data["density"]["dense_chickens"] == 0


def test_detect_image_error_handling(client: TestClient):
    # Unsupported file extension
    r = client.post(
        "/api/detect/image",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415

    # Corrupt image bytes
    r = client.post(
        "/api/detect/image",
        files={"file": ("broken.png", b"not-an-image", "image/png")},
    )
    assert r.status_code == 400

    # Invalid confidence threshold
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    _, enc = cv2.imencode(".png", blank)
    r = client.post(
        "/api/detect/image",
        files={"file": ("blank.png", enc.tobytes(), "image/png")},
        data={"conf": "1.5"},
    )
    assert r.status_code == 422


def test_path_traversal_protection(client: TestClient):
    r = client.get("/api/outputs/..%5Cweights%5Cbest.pt")
    assert r.status_code in (400, 404)
    r2 = client.get("/api/samples/image/..%5CREADME.md")
    assert r2.status_code in (400, 404)


def test_video_analysis_workflow_and_playback(client: TestClient, tmp_path: Path):
    # Build a 12-frame video from real test image frames so stationary birds are tracked
    img = cv2.imread(str(TEST_IMAGE))
    img_small = cv2.resize(img, (640, 360))
    clip_path = tmp_path / "poultry_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (640, 360))
    for _ in range(12):
        vw.write(img_small)
    vw.release()

    with clip_path.open("rb") as f:
        r = client.post(
            "/api/analyze/video",
            files={"file": ("poultry_clip.mp4", f, "video/mp4")},
            data={
                "conf": "0.30",
                "inactivity_window_s": "1.0",
                "inactivity_thresh_px": "5.0",
                "job_id": "test_job_1",
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frames_processed"] == 12
    assert data["total_unique_tracks"] > 50
    assert "h264" in data["codec"] or "mp4v" in data["codec"]
    assert data["tracker_used"] == "bytetrack"
    assert "interbird_distance_summary" in data
    assert data["interbird_distance_summary"]["mean_flock_nearest_neighbor_px"] is not None
    assert data["inactivity_summary"]["label"] == "Inactivity / Low-Mobility Indicator"
    # Since all 12 frames are identical (1.2s >= 1.0s window), stationary tracks must be flagged
    assert data["inactivity_summary"]["tracks_with_low_mobility"] > 50

    # Verify progress endpoint reports completed
    pr = client.get("/api/analyze/video/progress/test_job_1")
    assert pr.status_code == 200
    assert pr.json()["status"] == "completed"

    # Verify processed video is served and playable
    vid_r = client.get(data["processed_video_url"])
    assert vid_r.status_code == 200
    assert vid_r.headers["content-type"] == "video/mp4"
    assert len(vid_r.content) > 10000


def test_video_empty_and_invalid_handling(client: TestClient, tmp_path: Path):
    # Unsupported extension
    r = client.post(
        "/api/analyze/video",
        files={"file": ("bad.txt", b"not-a-video", "text/plain")},
    )
    assert r.status_code == 415

    # Unreadable video bytes
    r = client.post(
        "/api/analyze/video",
        files={"file": ("corrupt.mp4", b"corrupt-mp4-bytes", "video/mp4")},
    )
    assert r.status_code == 400

    # Blank video with 0 detections
    blank_clip = tmp_path / "blank.mp4"
    vw = cv2.VideoWriter(str(blank_clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(5):
        vw.write(np.zeros((240, 320, 3), dtype=np.uint8))
    vw.release()

    with blank_clip.open("rb") as f:
        r = client.post(
            "/api/analyze/video",
            files={"file": ("blank.mp4", f, "video/mp4")},
            data={"conf": "0.25"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["total_unique_tracks"] == 0
    assert any("No chickens were detected" in w for w in data["warnings"])


def test_detect_image_with_dual_model_and_clahe(client: TestClient):
    with TEST_IMAGE.open("rb") as f:
        r = client.post(
            "/api/detect/image",
            files={"file": ("frame00097.png", f, "image/png")},
            data={
                "conf": "0.25",
                "model_name": "yolov8n",
                "auto_enhance": "true",
                "containment_dedup": "true",
            },
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["model_used"] == "chicken_detector_yolov8n.pt"
    assert "clahe_applied" in data
    assert "dedup_removed" in data
    assert data["chicken_count"] > 0


def test_model_info_returns_available_models(client: TestClient):
    r = client.get("/api/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "available_models" in data
    models = {m["key"]: m for m in data["available_models"]}
    assert "yolov8m" in models
    assert "yolov8n" in models
    assert models["yolov8m"]["installed"] is True
    assert models["yolov8n"]["installed"] is True


