# Poultry-Bird-Detection-and-Image-Enhancement-YOLOv8

An integrated computer vision and behavioral analytics system for commercial poultry flock monitoring. This application combines high-accuracy deep learning chicken detection using YOLOv8 with automated low-light enhancement, ByteTrack multi-object tracking, KDTree spatial crowding/density analysis, cumulative movement measurement, inter-bird spatial distance metrics, and sliding-window inactivity detection.

---

## 1. System Overview and Capabilities

### Detection Engine
- **Primary Detector (YOLOv8m):** Trained for 100 epochs with AdamW optimization on broiler surveillance data, achieving **99.25% mAP@0.5** and **93.00% mAP@0.5:0.95**. Capable of resolving overlapping and partially occluded birds in dense flock environments.
- **Edge Detector (YOLOv8n):** A lightweight 3.0M parameter model (5.96 MB) designed for high-throughput edge deployment (50+ FPS on CUDA, 15+ FPS on CPU).
- **Automated Low-Light Enhancement:** Adaptive CLAHE enhancement operating on the LAB color space L-channel. Automatically activates on dim broiler shed frames (mean brightness < 80) to recover low-contrast detections without washing out well-lit frames.
- **Containment Deduplication:** Geometric suppression that eliminates false duplicate detections caused by nested or fragmented bounding boxes (containment ratio > 0.80).

### Multi-Object Tracking
- **ByteTrack (Recommended):** High-performance tracking utilizing Kalman filter velocity estimation and two-stage association. Includes a custom SciPy-based Linear Assignment Problem (`lap_shim.py`) compatibility layer that operates natively on Windows without requiring C++ build tools.
- **CentroidTracker (Fallback):** Greedy Euclidean distance association adapted from the original research pipeline.

### Behavioral Analytics
- **Spatial Density & Crowding:** Uses `scipy.spatial.KDTree` ball-tree queries over chicken centroids with an auto-scaling search radius (`2 x median box diagonal`) to detect dense clusters.
- **Movement & Cumulative Distance (TDP):** Measures cumulative pixel displacement per track with an adaptive jump rejection filter (`30 px` baseline at 480p) to filter out tracking jitter.
- **Inactivity / Low-Mobility Indicator:** A temporal sliding-window evaluator (default 10.0 seconds) that identifies birds exhibiting abnormally low spatial displacement or path extent after verifying at least 80% observation continuity.
- **Inter-Bird Spatial Distance:** Computes pairwise Euclidean centroid distance matrices, flock Nearest-Neighbor Distance (NND) distributions, and close-contact pair counts under user-configurable proximity thresholds (reported in pixel units `px`).

### Web Interface & Media Streaming
- **FastAPI Backend:** Asynchronous REST API providing video processing with live progress reporting and hardware acceleration management (CUDA with automatic CPU fallback).
- **React + Vite Frontend:** Interactive web dashboard featuring four dedicated views: **Overview Telemetry**, **Image Detection**, **Video Behavior Analysis**, and **Model Diagnostics**.
- **Browser-Compatible H.264 Video Streaming:** Employs `imageio-ffmpeg` to transcode processed video streams directly into web-standard H.264 `yuv420p` MP4 files with live in-browser playback.

---

## 2. Project Directory Structure

```text
Poultry-Bird-Detection-and-Image-Enhancement-YOLOv8/
├── .venv/                         # Project Python virtual environment (root)
├── requirements.txt               # Top-level dependencies for pip install
├── backend/                       # FastAPI Backend Application
│   ├── app/                       # Application source code
│   │   ├── api/                   # REST API routes (image, video, model, overview)
│   │   └── core/                  # Vision & analytics pipeline (detector, tracker, distance, etc.)
│   ├── tests/                     # 45 PyTest automated test suites
│   ├── weights/                   # YOLOv8 model weights (best.pt and yolov8n.pt)
│   ├── requirements.txt           # Backend-specific package requirements
│   └── run.py                     # Backend server entry point
│
├── frontend/                      # React 18 + Vite 6 Web Application
│   ├── src/                       # UI components, dashboard pages, and API services
│   ├── dist/                      # Production build distribution
│   └── package.json               # Frontend dependencies and scripts
│
├── data/                          # Unified Dataset & Benchmark Directory
│   ├── annotations/               # YOLO (.txt) and Pascal VOC (.xml) annotation files
│   ├── test_images/               # 28 benchmark evaluation frames
│   ├── train_images/              # 20 raw training image frames
│   └── sample_videos/             # Sample tracking videos
│
├── research/                      # Research Code & Reference Implementations
│   ├── experiments/               # Converted pure Python (.py) scripts from original Colab notebooks
│   │   ├── broiler_density_detection.py
│   │   ├── broiler_detection.py
│   │   ├── broiler_distance_measure.py
│   │   ├── broiler_sick_detection.py
│   │   └── broiler_tracking.py
│   └── legacy/                    # Historical research components (YOLOv4, Darknet, TF SavedModel)
│       ├── darknet/
│       ├── final_training/
│       ├── multi-object-tracker/
│       └── yolo-v4-tf.keras/
│
├── docs/                          # Technical Documentation & Source Audits
│   ├── INTEGRATION_AUDIT.md       # Feasibility audit of Project A and Project B
│   └── INTEGRATED_APP_SOURCE_AUDIT.md # Architecture audit and implementation plan
│
├── start_backend.bat              # One-click Windows launcher for FastAPI backend
├── start_frontend.bat             # One-click Windows launcher for Vite frontend
├── .gitignore                     # Git ignore rules for caches, venvs, and builds
└── README.md                      # Project documentation
```

---

## 3. Environment & Prerequisites

- **Operating System:** Windows 10/11 (64-bit) or Linux
- **Python:** Python 3.12 (virtual environment located at `.venv/`)
- **Node.js:** Node.js 18+ and npm (tested with Node v24 / npm 11)
- **Hardware Acceleration (Optional):** NVIDIA GPU with CUDA driver (verified on NVIDIA GeForce RTX 3050 Laptop GPU; automatically falls back to CPU if CUDA is unavailable)

---

## 4. Installation and Setup

### 1. Python Environment Setup
The project virtual environment is managed in `.venv` at the project root:

```powershell
# From the project root:
python -m venv .venv

# Step A: Install PyTorch with CUDA (recommended for GPU acceleration):
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
# (or CPU-only: .\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision)

# Step B: Install dependencies:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Frontend Setup
```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## 5. Running the Application

You can launch both services using the provided launcher scripts or through separate terminals:

### Option A: Launcher Scripts
- Double-click `start_backend.bat` (launches FastAPI at `http://127.0.0.1:8000`)
- Double-click `start_frontend.bat` (launches Vite at `http://127.0.0.1:5173`)

### Option B: Manual Terminal Execution

**Terminal 1 — Backend:**
```powershell
.\.venv\Scripts\python.exe backend\run.py
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

Open **`http://127.0.0.1:5173`** in your browser.

---

## 6. Automated Testing

The backend includes a comprehensive 45-test automated test suite covering detection, ByteTrack tracking, SciPy LAP shimming, inter-bird distance calculations, low-light enhancement, bounding-box deduplication, and REST API endpoints.

To run the test suite from the project root:
```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -v
```

All 45 tests pass:
```text
tests/test_api.py (10 tests)           PASSED
tests/test_behaviour.py (11 tests)     PASSED
tests/test_bytetrack.py (3 tests)      PASSED
tests/test_dedup.py (3 tests)          PASSED
tests/test_detector.py (7 tests)       PASSED
tests/test_enhancement.py (4 tests)    PASSED
tests/test_interbird.py (4 tests)      PASSED
tests/test_lap_shim.py (3 tests)       PASSED
======================= 45 passed in 23.13s =======================
```

---

## 7. Model Training & Evaluation Benchmark

The primary detector was trained from the `yolov8m.pt` baseline using transfer learning on annotated poultry imagery. The training run details extracted from the checkpoint metadata are summarized below:

| Hyperparameter / Metric | Value |
| :--- | :--- |
| **Model** | YOLOv8m (`yolov8m.pt`, 25.9M parameters, 49.6 MB) |
| **Epochs** | 100 complete epochs |
| **Optimizer** | AdamW (`lr0=0.002`, `lrf=0.01`, `weight_decay=0.0005`, `momentum=0.937`) |
| **Input Resolution** | 640 x 640 pixels |
| **Batch Size** | 16 |
| **Augmentations** | Mosaic (1.0), FlipLR (0.5), Random Erasing (0.4), HSV jitter |
| **Training Duration** | 5.42 hours on GPU |
| **Peak Validation Epoch** | Epoch 90 |
| **mAP@0.5** | **0.9925 (99.25%)** |
| **mAP@0.5:0.95** | **0.9300 (93.00%)** |
| **Precision** | **0.9850 (98.50%)** |
| **Recall** | **0.9752 (97.52%)** |

---

## 8. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health status, CUDA/CPU device status, loaded model names, library versions |
| `GET` | `/api/model/info` | Checkpoint metadata, active classes, and training parameters |
| `GET` | `/api/overview` | Aggregated dashboard telemetry, recent analysis summaries, and available test samples |
| `POST` | `/api/detect/image` | Multipart image upload for detection, density analysis, and inter-bird distance |
| `POST` | `/api/detect/sample/{name}` | Execute detection pipeline on a benchmark image in `data/test_images/` |
| `POST` | `/api/analyze/video` | Multipart video upload for tracking, movement, density, and inactivity analysis |
| `POST` | `/api/analyze/video/sample/{name}` | Execute video analysis pipeline on a sample video in `data/sample_videos/` |
| `GET` | `/api/analyze/video/progress/{id}` | Live progress tracking for long-running video inference jobs |
| `GET` | `/api/outputs/{filename}` | Retrieve generated annotated JPG frames and H.264 MP4 videos |

---

## 9. Scientific Disclaimer & Technical Limitations

1. **Inactivity Indicator Is a Visual Motion Heuristic:** Low spatial displacement over a sliding window indicates low mobility, not a clinical veterinary diagnosis. Normal poultry behavior includes resting, brooding, and sleeping.
2. **Pixel-Space Measurements:** All distance measurements (`px`), speeds (`px/s`), and nearest-neighbor distances are reported strictly in image pixel coordinates. Real-world metric dimensions (centimeters or meters) require camera calibration and perspective homography.
3. **Sample Video Burn-In:** The historical sample video in `data/sample_videos/` was recorded during the original 2022 research challenge and has previous green annotations burned into the raw video. Uploading a clean MP4 clip provides unannotated input validation.

---

## 10. Attribution and License

- **Poultry Detection Model:** Developed and trained using Ultralytics YOLOv8.
- **Original Challenge Research:** Based on *Chicken Behavior Analysis* (Second Place, 12th International Conference on Computer and Knowledge Engineering, 2022) by Tahereh Zarrat Ehsan and Seyed Mehdi Mohtavipour Shafti.
- **Tracking Algorithm:** Ultralytics BYTETracker with SciPy LAP optimization; fallback tracking adapted from `motrackers` by Aditya M. Deshpande.
- **License:** Apache 2.0 License.
