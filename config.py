"""
config.py — DriveGuardAI Central Configuration
Loads all settings from the .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/marius/python-service/.env")
print("Loaded config from .env file")

# ── Java Backend ───────────────────────────────────────────
JAVA_BACKEND_URL   = os.getenv("JAVA_BACKEND_URL",  "http://localhost:8080")

# ── Directories ───────────────────────────────────────────
SCREENSHOTS_DIR    = os.getenv("SCREENSHOTS_DIR",   "/home/marius/python-service/violations_screenshots")
DRIVER_FACES_DIR   = os.getenv("DRIVER_FACES_DIR",  "/home/marius/python-service/driver_faces")

# ── Twilio SMS ────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "YOUR_TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")
MANAGER_PHONE      = os.getenv("MANAGER_PHONE",      "+250787667755")

# ── YOLO Model Paths ──────────────────────────────────────
YOLO_WEIGHTS       = os.getenv("YOLO_WEIGHTS", "/home/marius/python-service/yolo/yolov3.weights")
YOLO_CFG           = os.getenv("YOLO_CFG",     "/home/marius/python-service/yolo/yolov3.cfg")
YOLO_NAMES         = os.getenv("YOLO_NAMES",   "/home/marius/python-service/yolo/coco.names")

# ── Detection Settings ────────────────────────────────────
VIOLATION_COOLDOWN = int(os.getenv("VIOLATION_COOLDOWN", "10"))

# ── YOLO Object Classes ───────────────────────────────────
SMOKING_OBJECTS  = {"cigarette", "cigar"}
EATING_OBJECTS   = {
    "sandwich", "hot dog", "pizza", "donut", "cake",
    "apple", "banana", "orange", "broccoli", "carrot",
    "fork", "knife", "spoon",
}
DRINKING_OBJECTS = {"bottle", "wine glass", "cup", "bowl", "vase"}
PHONE_OBJECTS    = {"cell phone", "remote", "mouse"}

# ── Violation Mappings ────────────────────────────────────
INCIDENT_TYPE_MAP = {
    'drowsy':         'DROWSINESS',
    'eyes_closed':    'DROWSINESS',
    'phone_use':      'PHONE_USE',
    'distracted':     'DISTRACTION',
    'no_seatbelt':    'NO_SEATBELT',
    'yawning':        'FATIGUE',
    'smoking':        'SMOKING',
    'eating':         'DISTRACTION',
    'drinking':       'DISTRACTION',
    'unknown_driver': 'UNAUTHORIZED_DRIVER',
}

SEVERITY_MAP = {
    'CRITICAL': {'drowsy', 'eyes_closed', 'phone_use', 'unknown_driver'},
    'HIGH':     {'distracted', 'no_seatbelt', 'smoking'},
}

CRITICAL_VIOLATIONS = {'drowsy', 'eyes_closed', 'phone_use', 'unknown_driver'}

AUDIO_MESSAGES = {
    'drowsy':         "Warning! Drowsiness detected. Please pull over.",
    'eyes_closed':    "Warning! Eyes closed. Stay alert.",
    'phone_use':      "Warning! Phone use detected. Put the phone down.",
    'distracted':     "Warning! Distraction detected. Focus on the road.",
    'no_seatbelt':    "Warning! Seatbelt not detected. Please buckle up.",
    'yawning':        "Fatigue detected. Consider taking a break.",
    'smoking':        "Warning! Smoking detected while driving.",
    'eating':         "Warning! Eating detected while driving.",
    'drinking':       "Warning! Drinking detected while driving.",
    'unknown_driver': "Warning! Unrecognised driver detected.",
}

VIOLATION_LABELS = {
    'drowsy':         'DROWSINESS',
    'eyes_closed':    'EYES CLOSED',
    'phone_use':      'PHONE USE WHILE DRIVING',
    'unknown_driver': 'UNRECOGNISED DRIVER',
}
