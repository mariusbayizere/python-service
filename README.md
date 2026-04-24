🚗 DriveGuardAI — AI-Powered Driver Monitoring System

Real-time driver safety monitoring using computer vision, deep learning, and instant alerting — built for fleet operators, logistics companies, and transportation safety enforcement.

/home/marius/python-service/violations_screenshots/distracted_20260411_121958.jpg
/home/marius/python-service/violations_screenshots/distracted_20260411_122014.jpg
/home/marius/python-service/violations_screenshots/distracted_20260413_075740.jpg
/home/marius/python-service/violations_screenshots/distracted_20260424_124336.jpg




📌 Overview
DriveGuardAI is a production-grade AI monitoring service that continuously analyzes live camera feeds to detect unsafe driver behaviors in real time. It is designed to match the capabilities of commercial fleet safety platforms used across North America and Europe — such as Lytx DriveCam, Samsara AI Dash Cams, and Netradyne Driver•i.
The system detects violations the moment they occur, saves screenshot evidence, plays an in-cabin audio warning, persists the event to a database, and immediately notifies the fleet manager via SMS and email — all within seconds.

🎯 Key Features
FeatureDescription👁️ Drowsiness DetectionDetects eye closure, head nodding, and fatigue patterns📱 Phone Use DetectionIdentifies phone handling while driving via computer vision😴 Distraction DetectionFlags when driver attention leaves the road🚬 Smoking DetectionYOLO-based object detection for smoking behaviour🍔 Eating/Drinking DetectionDetects food and drink consumption while driving🔔 Seatbelt DetectionAlerts when seatbelt is not worn🍺 Drunk Driving Pattern AnalysisBehavioural pattern engine — escalates when drowsiness + distraction co-occur repeatedly🎙️ Real-Time Audio AlertsIn-cabin TTS voice warnings triggered instantly📸 Automatic Screenshot EvidenceJPEG evidence captured and stored for every violation📧 Email NotificationsDetailed violation reports sent via Java backend📱 SMS NotificationsCritical alerts sent to fleet manager via Twilio🎥 Live MJPEG StreamReal-time camera feed accessible via browser or dashboard🧑 Driver Face VerificationConfirms authorized driver identity at session start🗄️ Database PersistenceAll violations stored via Spring Boot REST API

🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                    DriveGuardAI Python Service           │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────────┐ │
│  │ Camera   │──▶│ monitor  │──▶│ violation_handler   │ │
│  │ (OpenCV) │   │ .py      │   │ .py                 │ │
│  └──────────┘   └──────────┘   └─────────────────────┘ │
│                      │                  │               │
│              ┌───────┴───────┐   ┌──────┴──────┐       │
│              │  detector.py  │   │notifications│       │
│              │  YOLO + Face  │   │ SMS + Email │       │
│              └───────────────┘   └─────────────┘       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │           api_server.py  (Flask REST API)        │   │
│  │  /api/monitoring/start   /api/stream             │   │
│  │  /api/monitoring/stop    /api/screenshots        │   │
│  │  /api/test-sms           /api/health             │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  Spring Boot API │          │   React Dashboard    │
│  (Java Backend)  │          │   (Frontend UI)      │
│  Port 8080       │          │   Port 3000          │
└──────────────────┘          └──────────────────────┘

📁 Project Structure
python-service/
│
├── api_server.py          # Flask entry point — REST API routes only
├── config.py              # All environment variables & constants
├── monitor.py             # Camera loop, MJPEG stream, frame processing
├── detector.py            # YOLOv3 object detection + face recognition
├── violation_handler.py   # Screenshot, audio, DB save, drunk-driving detector
├── notifications.py       # SMS (Twilio) + Email (via Java backend)
│
├── driver_monitoring_system_v3_simple.py  # DMS core (drowsy/distraction/seatbelt)
│
├── yolo/
│   ├── yolov3.weights     # YOLOv3 pre-trained weights (not in git — too large)
│   ├── yolov3.cfg         # YOLOv3 architecture config
│   └── coco.names         # COCO 80-class labels
│
├── driver_faces/          # Registered driver face images (not in git — privacy)
├── violations_screenshots/ # Saved violation evidence (not in git — too large)
│
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (never commit this)
└── .gitignore
🚨 Violations Detected
Critical (SMS + Email + Audio + Screenshot)

DROWSINESS — eyes closing, head dropping
EYES_CLOSED — sustained eye closure
PHONE_USE — phone detected in hand while driving
UNAUTHORIZED_DRIVER — unrecognized face at the wheel
DRUNK_DRIVING — pattern escalation (drowsy + distracted repeatedly)

High Severity (Email + Audio + Screenshot)

DISTRACTION — driver not looking at road
NO_SEATBELT — seatbelt not fastened
SMOKING — smoking detected via YOLO

Medium Severity (Audio + Screenshot)

FATIGUE — yawning detected
EATING — eating while driving
DRINKING — drinking while driving

⚙️ Installation
Prerequisites

Python 3.10+
Java Spring Boot backend running on port 8080
Webcam or dashcam connected
YOLOv3 weights file (download here)

1. Clone the repository
bashgit clone https://github.com/mariusbayizere/python-service.git
cd python-service
2. Create and activate virtual environment
bashpython3 -m venv venv
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Download YOLO weights
bashmkdir -p yolo
wget https://pjreddie.com/media/files/yolov3.weights -O yolo/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolo/yolov3.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

5. Configure environment variables
bashcp .env.example .env
nano .env
env# Java Backend
JAVA_BACKEND_URL=http://localhost:8080

# Directories
SCREENSHOTS_DIR=/home/marius/python-service/violations_screenshots
DRIVER_FACES_DIR=/home/marius/python-service/driver_faces

# Twilio SMS
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1XXXXXXXXXX
MANAGER_PHONE=+250XXXXXXXXX

# YOLO
YOLO_WEIGHTS=/home/marius/python-service/yolo/yolov3.weights
YOLO_CFG=/home/marius/python-service/yolo/yolov3.cfg
YOLO_NAMES=/home/marius/python-service/yolo/coco.names

# Detection
VIOLATION_COOLDOWN=10
6. Start the service
bashpython api_server.py

🔌 API Reference
Monitoring
MethodEndpointDescriptionPOST/api/monitoring/startStart a monitoring sessionPOST/api/monitoring/stopStop the active sessionGET/api/monitoring/statusGet current session statusGET/api/streamLive MJPEG video stream
Start session request body:
json{
  "driver_id": 1,
  "vehicle_id": 2,
  "trip_id": 3,
  "driver_name": "John Doe",
  "vehicle_name": "Toyota Hilux KAB 001A",
  "trip_name": "Kigali — Musanze Route"
}
Health & Diagnostics
MethodEndpointDescriptionGET/api/healthSystem health checkGET/api/test-yoloTest YOLO detection livePOST/api/test-smsSend a test SMS alertGET/api/test-java-connectionVerify Java backend connection
Screenshots
MethodEndpointDescriptionGET/api/screenshotsList all violation screenshotsGET/api/screenshots/<filename>Serve a specific screenshot image
Drivers
MethodEndpointDescriptionPOST/api/drivers/register-faceRegister a driver's faceGET/api/drivers/known-facesList all registered drivers

🔍 How It Works
1. Session Start
When a trip begins, the React dashboard calls /api/monitoring/start with driver, vehicle, and trip details. The system opens the camera and begins real-time analysis.
2. Frame Processing (30 FPS)
Every frame is processed by the Driver Monitoring System (DMS) which uses facial landmark detection to monitor:

Eye aspect ratio (EAR) → drowsiness/eye closure
Head pose estimation → distraction
Mouth aspect ratio → yawning
Shoulder/torso detection → seatbelt

3. YOLO Object Detection (~1 FPS)
Every 30 frames, YOLOv3 scans for physical objects: phones, food, drinks, and smoking materials.
4. Drunk Driving Pattern Engine
A sliding 60-second window monitors co-occurrence of drowsiness and distraction events. If both fire 2+ times within the window, the system escalates to a DRUNK_DRIVING violation — the highest severity level.
5. Violation Response Pipeline
Violation Detected
      │
      ├─▶ Save screenshot to disk
      ├─▶ Play audio warning (TTS)
      ├─▶ POST to Java backend → database
      ├─▶ Send SMS to fleet manager (critical only)
      └─▶ Send email via Java notification service

flask
flask-cors
opencv-python
numpy
pyttsx3
face-recognition
twilio
python-dotenv
requests
dlib
imutils
scipy
Install all with:
bashpip install -r requirements.txt

🌍 Real-World Comparison
FeatureDriveGuardAILytx DriveCamSamsara AIDrowsiness detection✅✅✅Phone use detection✅✅✅Distraction detection✅✅✅Seatbelt detection✅✅✅Audio in-cabin alerts✅✅✅Screenshot evidence✅✅✅SMS/Email alerts✅✅✅Driver identification✅✅✅Drunk driving detection✅❌❌Open source✅❌❌

🔐 Security Notes

Never commit your .env file — it is listed in .gitignore
Rotate your Twilio Auth Token immediately if it is ever exposed
Driver face images are stored locally and never uploaded to GitHub
All violation screenshots remain on-device and are served via authenticated API

👨‍💻 Author
Marius Bayizere
Backend Software Engineer | Kigali, Rwanda

📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
