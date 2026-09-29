# VisionGuard AI 👁️🛡️

**Real-Time Scene Understanding & Action Recognition Pipeline**

VisionGuard AI is an advanced, edge-computed computer vision pipeline designed to transform standard video feeds into rich semantic data. It utilizes a standard computer webcam or network RTSP stream to perform real-time multi-object detection, human pose estimation, and heuristic action recognition. 

This project serves as a comprehensive demonstration of deep learning inference optimization, backend orchestration, and real-time data processing, laying the groundwork for commercial smart CCTV analytics.

## 🚀 Key Features

* **Multi-Class Object Detection:** Utilizes Ultralytics YOLO to detect and track 80+ standard COCO classes (people, vehicles, backpacks, etc.) in real-time.
* **Pose Estimation & Action Engine:** Parallel skeletal tracking maps human keypoints. A temporal logic engine interprets postural geometry to categorize actions (e.g., Sitting vs. Standing).
* **Decoupled API Architecture:** A FastAPI backend ingests video frames and serves MJPEG streams via REST endpoints, ensuring the UI remains highly responsive without rendering bottlenecks.
* **Live Telemetry & Event Logging:** A throttled SQLite pipeline logs continuous detection events and confidence scores without locking the database, served to a responsive Tailwind CSS dashboard.
* **Edge-Ready & Hardware Agnostic:** Abstracted video ingestion supports both local USB webcams and wireless RTSP streams from physical IP cameras.
* **Docker Containerization:** Fully containerized with necessary OpenCV, FFmpeg, and Python dependencies for seamless 24/7 deployment on dedicated edge servers or Jetson devices.

## 🛠️ System Technology Stack

* **Computer Vision:** OpenCV (Frame manipulation, RTSP ingestion, overlay telemetry)
* **Machine Learning:** Ultralytics YOLO26 (Detection & Pose Estimation)
* **Backend Orchestration:** FastAPI, Uvicorn, Python 3.10
* **Frontend UI:** HTML5, JavaScript, Tailwind CSS
* **Database & Logging:** SQLite3
* **DevOps:** Docker, Docker Compose

## 📁 Directory Structure

```text
visionguard-ai-dashboard/
├── backend/
│   ├── main.py              # FastAPI server, YOLO inference, & SQLite pipeline
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Tailwind CSS dashboard and async API polling
├── models/                  # Local storage for YOLO weights (.pt)
├── data/                    # Persistent volume for events.db
├── Dockerfile               # Edge deployment OS blueprint
└── docker-compose.yml       # Container orchestration and volume mapping
