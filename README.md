# 🤖 DriveGuardAI — Python AI Monitoring Service

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)](https://opencv.org)
[![YOLOv3](https://img.shields.io/badge/YOLO-v3-red)](https://pjreddie.com/darknet/yolo/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployed-326CE5?logo=kubernetes)](https://kubernetes.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Real-time AI driver safety monitoring service using computer vision and deep learning. Detects dangerous driving behaviours, captures evidence, and instantly alerts fleet managers — all within seconds.

🌐 **Live Demo:** https://driveguard.duckdns.org  
📡 **API Health:** https://driveguard.duckdns.org/ai/api/health  
🎥 **Live Stream:** https://driveguard.duckdns.org/ai/api/stream

> This service is part of the full **DriveGuardAI** platform:  
> 👉 [Main Repository](https://github.com/mariusbayizere/DriveGuardAI-)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Violation Screenshots](#violation-screenshots)
- [Features](#features)
- [Architecture](#architecture)
- [Violations Detected](#violations-detected)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Docker & Kubernetes](#docker--kubernetes)
- [Author](#author)

---

## 🎯 Overview

DriveGuardAI Python Service is a production-grade AI monitoring microservice that continuously analyses live camera feeds to detect unsafe driver behaviours in real time. It is designed to match the capabilities of commercial fleet safety platforms like **Lytx DriveCam**, **Samsara AI**, and **Netradyne Driver•i**.

The moment a violation is detected the system:
- 📸 Captures a **JPEG screenshot** as evidence
- 🔊 Plays an **in-cabin audio warning** via TTS
- 💾 Saves the incident to the **database** via Spring Boot REST API
- 📱 Sends an **SMS alert** to the fleet manager (critical violations)
- 📧 Sends a detailed **email report** via the Java backend
- 📉 Automatically **deducts safety score** from the driver's profile

All of this happens within **2–3 seconds** of violation detection.

---

## 📸 Violation Screenshots

Real violation evidence automatically captured by the AI system during live monitoring sessions.

### 👁️ Eyes Closed — Drowsiness Detected
![Eyes Closed 1](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/eyes_closed_20260418_074137.jpg)

---

![Eyes Closed 2](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/eyes_closed_20260418_074148.jpg)

---

![Eyes Closed 3](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/eyes_closed_20260418_074158.jpg)

---

![Eyes Closed 4](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/eyes_closed_20260424_093326.jpg)

---

### 🔔 No Seatbelt — Safety Violation Detected
![No Seatbelt 1](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/no_seatbelt_20260205_075413.jpg)

---

![No Seatbelt 2](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/no_seatbelt_20260207_094231.jpg)

---

![No Seatbelt 3](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/no_seatbelt_20260305_111727.jpg)

---

### 📱 Phone Use — Distracted Driving Detected
![Phone Use 1](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/phone_use_20260413_072734.jpg)

---

![Phone Use 2](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/phone_use_20260413_075720.jpg)

---

![Phone Use 3](https://raw.githubusercontent.com/mariusbayizere/DriveGuardAI-/main/python-service/docs/screenshots/phone_use_20260424_124337.jpg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Drowsiness Detection** | Detects eye closure, head nodding, and fatigue patterns using Eye Aspect Ratio (EAR) |
| 📱 **Phone Use Detection** | Identifies phone handling while driving via YOLOv3 computer vision |
| 😴 **Distraction Detection** | Flags when driver attention leaves the road using head pose estimation |
| 🚬 **Smoking Detection** | YOLO-based object detection for smoking behaviour |
| 🍔 **Eating & Drinking Detection** | Detects food and drink consumption while driving |
| 🔔 **Seatbelt Detection** | Alerts when seatbelt is not worn |
| 🍺 **Drunk Driving Pattern Analysis** | Behavioural pattern engine — escalates when drowsiness and distraction co-occur repeatedly |
| 🎙️ **Real-Time Audio Alerts** | In-cabin TTS voice warnings triggered instantly |
| 📸 **Automatic Screenshot Evidence** | JPEG evidence captured and stored for every violation |
| 📧 **Email Notifications** | Detailed violation reports sent via Java backend |
| 📱 **SMS Notifications** | Critical alerts sent to fleet manager via Twilio |
| 🎥 **Live MJPEG Stream** | Real-time camera feed accessible via browser or dashboard |
| 🧑 **Driver Face Verification** | Confirms authorised driver identity at session start |
| 🗄️ **Database Persistence** | All violations stored via Spring Boot REST API |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DriveGuardAI Python Service                 │
│                                                             │
│   Camera (OpenCV)                                           │
│        │                                                    │
│        ▼                                                    │
│   monitor.py ──────────────────────────────────────────┐   │
│        │                                               │   │
│        ├── driver_monitoring_system_v3_simple.py       │   │
│        │   (EAR drowsiness / head pose / seatbelt)     │   │
│        │                                               │   │
│        ├── detector.py                                 │   │
│        │   (YOLOv3 → phone/smoking/eating/drinking)    │   │
│        │   (face_recognition → driver verification)    │   │
│        │                                               │   │
│        └── violation_handler.py                        │   │
│            (screenshot + audio + DB + SMS + email)     │   │
│                     │                                  │   │
│                     ▼                                  ▼   │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          api_server.py  (Flask REST API)             │  │
│   │                                                      │  │
│   │  POST /api/monitoring/start                          │  │
│   │  POST /api/monitoring/stop                           │  │
│   │  GET  /api/monitoring/status                         │  │
│   │  GET  /api/stream        (MJPEG live feed)           │  │
│   │  GET  /api/screenshots   (evidence files)            │  │
│   │  GET  /api/health                                    │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────────┐
         │  Spring Boot API │  │   React Dashboard    │
         │  (Java Backend)  │  │   (Frontend UI)      │
         │  Port 8080       │  │   Port 80            │
         └──────────────────┘  └──────────────────────┘
```

### Violation Response Pipeline

```
Violation Detected
      │
      ├──▶ Save screenshot to disk (JPEG evidence)
      ├──▶ Play audio warning (TTS in-cabin alert)
      ├──▶ POST violation to Java backend → PostgreSQL
      ├──▶ Send SMS to fleet manager (CRITICAL only)
      └──▶ Send email via Java notification service
```

### Drunk Driving Pattern Engine

```
60-second sliding window
        │
        ├── Drowsy events ≥ 2  ──┐
        │                        ├──▶ DRUNK_DRIVING (CRITICAL)
        └── Distracted events ≥ 2 ─┘
```

---

## 🚨 Violations Detected

### 🔴 CRITICAL — SMS + Email + Audio + Screenshot

| Violation | Detection Method | Score Deduction |
|-----------|-----------------|-----------------|
| `DROWSINESS` | Eye Aspect Ratio (EAR) via dlib | −10 pts |
| `EYES_CLOSED` | Sustained eye closure | −10 pts |
| `PHONE_USE` | YOLOv3 object detection | −10 pts |
| `UNAUTHORIZED_DRIVER` | Face recognition mismatch | −10 pts |
| `DRUNK_DRIVING` | Behavioural pattern escalation | −10 pts |

### 🟠 HIGH — Email + Audio + Screenshot

| Violation | Detection Method | Score Deduction |
|-----------|-----------------|-----------------|
| `DISTRACTION` | Head pose estimation | −5 pts |
| `NO_SEATBELT` | Computer vision | −5 pts |
| `SMOKING` | YOLOv3 object detection | −5 pts |

### 🟡 MEDIUM — Audio + Screenshot

| Violation | Detection Method | Score Deduction |
|-----------|-----------------|-----------------|
| `FATIGUE` | Yawning detection | −3 pts |
| `EATING` | YOLOv3 food detection | −3 pts |
| `DRINKING` | YOLOv3 bottle/cup detection | −3 pts |

> ⚠️ Driver is **auto-suspended** when safety score drops below **50 points**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Web Framework** | Flask 3.0, Flask-CORS |
| **Computer Vision** | OpenCV 4.8 |
| **Object Detection** | YOLOv3 (COCO 80 classes) |
| **Facial Landmarks** | dlib + shape_predictor_68_face_landmarks |
| **Face Recognition** | face_recognition library |
| **Deep Learning** | PyTorch 2.1 (CPU) |
| **SMS Alerts** | Twilio |
| **Audio Alerts** | pyttsx3 (TTS) |
| **WSGI Server** | Gunicorn |
| **Container** | Docker |
| **Orchestration** | Kubernetes (Kind) |

---

## 📁 Project Structure

```
python-service/
│
├── api_server.py                          # Flask entry point — REST API routes only
├── config.py                              # All environment variables & constants
├── monitor.py                             # Camera loop, MJPEG stream, frame processing
├── detector.py                            # YOLOv3 object detection + face recognition
├── violation_handler.py                   # Screenshot, audio, DB save, drunk-driving detector
├── notifications.py                       # SMS (Twilio) + Email (via Java backend)
├── driver_monitoring_system_v3_simple.py  # DMS core — EAR / head pose / seatbelt
│
├── yolo/
│   ├── yolov3.weights                     # Pre-trained weights (237MB — not in git)
│   ├── yolov3.cfg                         # YOLOv3 architecture config
│   └── coco.names                         # COCO 80-class labels
│
├── shape_predictor_68_face_landmarks.dat  # dlib facial landmark model
│
├── docs/screenshots/                      # Sample violation evidence screenshots
├── driver_faces/                          # Registered driver face images (not in git)
├── violations_screenshots/                # Live violation evidence (not in git)
│
├── Dockerfile                             # Production Docker image
├── requirements.txt                       # Python dependencies
├── .env                                   # Environment variables (never commit)
└── .gitignore
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- Spring Boot backend running on port 8080
- Webcam or dashcam connected
- YOLOv3 weights file

### 1. Clone the repository

```bash
git clone https://github.com/mariusbayizere/DriveGuardAI-.git
cd DriveGuardAI-/python-service
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO weights

```bash
mkdir -p yolo
wget https://pjreddie.com/media/files/yolov3.weights -O yolo/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolo/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O yolo/coco.names
```

### 5. Configure environment variables

```bash
nano .env
```

```env
# Java Backend
JAVA_BACKEND_URL=http://localhost:8080

# Directories
SCREENSHOTS_DIR=/path/to/violations_screenshots
DRIVER_FACES_DIR=/path/to/driver_faces

# Twilio SMS
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1XXXXXXXXXX
MANAGER_PHONE=+250XXXXXXXXX

# YOLO Model Paths
YOLO_WEIGHTS=/path/to/yolo/yolov3.weights
YOLO_CFG=/path/to/yolo/yolov3.cfg
YOLO_NAMES=/path/to/yolo/coco.names

# Detection Settings
VIOLATION_COOLDOWN=10
```

### 6. Start the service

```bash
python api_server.py
```

Service runs at: `http://localhost:5000`

---

## 📖 API Reference

### 🎥 Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/monitoring/start` | Start a monitoring session |
| POST | `/api/monitoring/stop` | Stop the active session |
| GET | `/api/monitoring/status` | Get current session status |
| GET | `/api/stream` | Live MJPEG video stream |

**Start session request body:**
```json
{
  "driver_id": 1,
  "vehicle_id": 2,
  "trip_id": 3,
  "driver_name": "John Doe",
  "vehicle_name": "Toyota Hilux (RAB-001A)",
  "trip_name": "Kigali — Musanze Route"
}
```

### 🏥 Health & Diagnostics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/test-yolo` | Test YOLO detection live |
| POST | `/api/test-sms` | Send a test SMS alert |
| GET | `/api/test-java-connection` | Verify Java backend connection |

**Health response example:**
```json
{
  "status": "healthy",
  "yolo_loaded": true,
  "face_recog": false,
  "tts_available": false,
  "twilio_configured": true,
  "monitoring_active": false,
  "known_drivers": 0,
  "timestamp": "2026-05-15T10:00:00"
}
```

### 📸 Screenshots

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/screenshots` | List all violation screenshots |
| GET | `/api/screenshots/<filename>` | Serve a specific screenshot |

### 🧑 Driver Face Registration

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/drivers/register-face` | Register a driver's face |
| GET | `/api/drivers/known-faces` | List all registered drivers |

---

## 🔍 How It Works

### 1. Session Start
When a trip begins, the React dashboard calls `POST /api/monitoring/start` with driver, vehicle, and trip details. The service opens the camera and begins real-time analysis.

### 2. Frame Processing (30 FPS)
Every frame is processed by the Driver Monitoring System (DMS) which uses **dlib 68 facial landmarks** to monitor:

- **Eye Aspect Ratio (EAR)** → drowsiness and eye closure
- **Head pose estimation** → distraction detection
- **Mouth aspect ratio** → yawning / fatigue
- **Shoulder / torso detection** → seatbelt status

### 3. YOLO Object Detection (~1 FPS)
Every 30 frames, **YOLOv3** scans for physical objects:

```
COCO classes monitored:
  Phone use  → cell phone, remote, mouse
  Eating     → sandwich, pizza, apple, fork, knife, spoon
  Drinking   → bottle, cup, wine glass, bowl
  Smoking    → cigarette, cigar
```

### 4. Drunk Driving Pattern Engine
A **60-second sliding window** monitors co-occurrence of violations. If drowsiness fires 2+ times AND distraction fires 2+ times within the same window, the system escalates to `DRUNK_DRIVING` — the highest severity level.

### 5. Face Verification
At session start, the system captures a frame and compares it against the registered face of the assigned driver using **face_recognition**. A mismatch triggers an `UNAUTHORIZED_DRIVER` violation.

---

## 🐳 Docker & Kubernetes

### Build Docker image

```bash
docker build -t mariusbayizere/driveguard-flask:latest .
docker push mariusbayizere/driveguard-flask:latest
```

### Deploy to Kubernetes

```bash
kubectl set image deployment/driveguard-flask \
  flask=mariusbayizere/driveguard-flask:latest -n prod

kubectl rollout status deployment/driveguard-flask -n prod
```

### Check service health

```bash
curl https://driveguard.duckdns.org/ai/api/health
```

---

## 🌍 Comparison with Commercial Solutions

| Feature | DriveGuardAI | Lytx DriveCam | Samsara AI |
|---------|-------------|---------------|------------|
| Drowsiness detection | ✅ | ✅ | ✅ |
| Phone use detection | ✅ | ✅ | ✅ |
| Distraction detection | ✅ | ✅ | ✅ |
| Seatbelt detection | ✅ | ✅ | ✅ |
| Audio in-cabin alerts | ✅ | ✅ | ✅ |
| Screenshot evidence | ✅ | ✅ | ✅ |
| SMS / Email alerts | ✅ | ✅ | ✅ |
| Driver identification | ✅ | ✅ | ✅ |
| Drunk driving detection | ✅ | ❌ | ❌ |
| Open source | ✅ | ❌ | ❌ |
| Self-hosted | ✅ | ❌ | ❌ |

---

## 🔐 Security Notes

- Never commit your `.env` file — it is listed in `.gitignore`
- Rotate your Twilio Auth Token immediately if ever exposed
- Driver face images are stored locally and never uploaded to GitHub
- All violation screenshots remain on-device and are served via the API

---

## 👨‍💻 Author

**Bayizere Marius**  
Backend Software Engineer | Kigali, Rwanda

- 📧 Email: bayizeremarius119@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/bayizere-marius/
- 🐙 GitHub: https://github.com/mariusbayizere
- 🌐 Live App: https://driveguard.duckdns.org

---

## 📄 License

This project is licensed under the MIT License.
