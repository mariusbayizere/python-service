"""
violation_handler.py — DriveGuardAI Violation Handler
Handles screenshot saving, DB persistence, audio alerts,
drunk-driving pattern detection, and notification dispatch.
"""
import os
import time
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np
import requests

import config
import notifications

# ── TTS setup ─────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)
    _tts_engine.setProperty('volume', 1.0)
    TTS_AVAILABLE = True
    print("TTS engine ready")
except Exception as e:
    TTS_AVAILABLE = False
    print(f"pyttsx3 not available: {e}")

_audio_lock = threading.Lock()


# ── Drunk Driving Detector ────────────────────────────────

class DrunkDrivingDetector:
    """
    Detects drunk driving from behaviour patterns.
    Fires if drowsiness AND distraction appear 2+ times each
    within a 60-second window — consistent with DUI impairment.
    """
    def __init__(self, window_seconds: int = 60, threshold: int = 2):
        self.window_seconds = window_seconds
        self.threshold      = threshold
        self.drowsy_times   = []
        self.distract_times = []
        self.drunk_fired_at = 0.0

    def record(self, alert_name: str) -> None:
        now    = time.time()
        cutoff = now - self.window_seconds
        if alert_name in ('drowsy', 'eyes_closed'):
            self.drowsy_times.append(now)
        elif alert_name == 'distracted':
            self.distract_times.append(now)
        self.drowsy_times   = [t for t in self.drowsy_times   if t > cutoff]
        self.distract_times = [t for t in self.distract_times if t > cutoff]

    def is_drunk(self) -> bool:
        if time.time() - self.drunk_fired_at < 300:
            return False
        if (len(self.drowsy_times)   >= self.threshold and
                len(self.distract_times) >= self.threshold):
            self.drunk_fired_at = time.time()
            print(f"DRUNK DRIVING PATTERN DETECTED: "
                  f"drowsy={len(self.drowsy_times)}, "
                  f"distracted={len(self.distract_times)} in last 60s")
            return True
        return False

    def reset(self) -> None:
        self.drowsy_times   = []
        self.distract_times = []
        self.drunk_fired_at = 0.0


drunk_detector = DrunkDrivingDetector()


# ── Helpers ───────────────────────────────────────────────

def _get_severity(alert_name: str) -> str:
    if alert_name in config.SEVERITY_MAP['CRITICAL']:
        return 'CRITICAL'
    if alert_name in config.SEVERITY_MAP['HIGH']:
        return 'HIGH'
    return 'MEDIUM'


def _get_incident_type(alert_name: str) -> str:
    return config.INCIDENT_TYPE_MAP.get(alert_name, 'OTHER')


def save_screenshot(frame: np.ndarray, violation_type: str) -> str | None:
    """Save a JPEG screenshot for a violation. Returns filename or None."""
    print(f"DEBUG screenshot: Saving for '{violation_type}' → {config.SCREENSHOTS_DIR}")
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)
    try:
        if frame is None or frame.size == 0:
            print("DEBUG screenshot: frame is None or empty")
            return None
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation_type}_{ts}.jpg"
        filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
        if cv2.imwrite(filepath, frame):
            size = os.path.getsize(filepath)
            print(f"DEBUG screenshot: saved {filename} ({size} bytes)")
            return filename
        print("DEBUG screenshot: cv2.imwrite returned False")
    except Exception as e:
        print(f"DEBUG screenshot: EXCEPTION — {e}")
        traceback.print_exc()
    return None


def _save_to_database(violation_data: dict) -> bool:
    try:
        r = requests.post(
            config.JAVA_BACKEND_URL + "/api/v1/incidents/violation",
            json=violation_data,
            timeout=10,
        )
        if r.status_code in (200, 201):
            print("Violation saved to database successfully")
            return True
        print(f"Java returned {r.status_code}: {r.text}")
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Java — is Spring Boot running on port 8080?")
    except Exception as e:
        print(f"send_violation error: {e}")
    return False


def play_audio(alert_name: str, monitoring_active_flag: list) -> None:
    """Play TTS alert in a background thread."""
    def _run():
        if not monitoring_active_flag[0]:
            return
        with _audio_lock:
            if not monitoring_active_flag[0]:
                return
            msg = config.AUDIO_MESSAGES.get(
                alert_name,
                f"Warning! {alert_name.replace('_', ' ')} detected."
            )
            print(f"AUDIO: {msg}")
            if TTS_AVAILABLE:
                try:
                    _tts_engine.say(msg)
                    _tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
    threading.Thread(target=_run, daemon=True).start()


def _fire_drunk_driving(frame: np.ndarray, current_session: dict,
                        monitoring_active_flag: list) -> None:
    screenshot   = save_screenshot(frame, 'drunk_driving')
    driver_name  = current_session.get('driver_name')  or f"Driver #{current_session['driver_id']}"
    vehicle_name = current_session.get('vehicle_name') or f"Vehicle #{current_session['vehicle_id']}"
    trip_name    = current_session.get('trip_name')    or f"Trip #{current_session['trip_id']}"

    data = {
        'driver_id':     current_session['driver_id'],
        'vehicle_id':    current_session['vehicle_id'],
        'trip_id':       current_session['trip_id'],
        'incident_type': 'DRUNK_DRIVING',
        'severity':      'CRITICAL',
        'description':   (
            f"{driver_name} is showing multiple simultaneous impairment signs "
            f"(repeated drowsiness + distraction) consistent with drunk driving "
            f"while on {trip_name} in {vehicle_name}. Immediate intervention required."
        ),
        'timestamp':  datetime.now().isoformat(),
        'screenshot': screenshot,
    }

    if TTS_AVAILABLE and monitoring_active_flag[0]:
        def _say():
            with _audio_lock:
                if monitoring_active_flag[0]:
                    _tts_engine.say(
                        "URGENT WARNING! Driver appears to be under the influence. "
                        "Contact authorities immediately."
                    )
                    _tts_engine.runAndWait()
        threading.Thread(target=_say, daemon=True).start()

    if _save_to_database(data):
        notifications.send_sms_alert('drowsy', current_session)
        notifications.trigger_email(data, current_session)
        print("DRUNK_DRIVING violation saved and manager notified")


# ── Main violation entry point ────────────────────────────

def handle_violation(frame: np.ndarray, alert_name: str,
                     current_session: dict,
                     last_violation_times: dict,
                     monitoring_active_flag: list) -> None:
    """
    Central violation handler:
      1. Drunk-driving pattern check
      2. Save screenshot
      3. Play audio alert
      4. Persist to database
      5. Send SMS + email for critical violations
    """
    print(f"VIOLATION DETECTED: {alert_name}")

    drunk_detector.record(alert_name)
    if drunk_detector.is_drunk():
        print("ESCALATING TO DRUNK_DRIVING")
        _fire_drunk_driving(frame, current_session, monitoring_active_flag)

    screenshot   = save_screenshot(frame, alert_name)
    driver_name  = current_session.get('driver_name')  or f"Driver #{current_session['driver_id']}"
    vehicle_name = current_session.get('vehicle_name') or f"Vehicle #{current_session['vehicle_id']}"
    trip_name    = current_session.get('trip_name')    or f"Trip #{current_session['trip_id']}"

    data = {
        'driver_id':     current_session['driver_id'],
        'vehicle_id':    current_session['vehicle_id'],
        'trip_id':       current_session['trip_id'],
        'incident_type': _get_incident_type(alert_name),
        'severity':      _get_severity(alert_name),
        'description':   (
            f"{driver_name} was detected doing "
            f"{alert_name.replace('_', ' ')} "
            f"while on {trip_name} in {vehicle_name}"
        ),
        'timestamp':  datetime.now().isoformat(),
        'screenshot': screenshot,
    }

    play_audio(alert_name, monitoring_active_flag)

    if _save_to_database(data):
        last_violation_times[alert_name] = time.time()
        notifications.send_sms_alert(alert_name, current_session)
        notifications.trigger_email(data, current_session)
    else:
        print("Failed to save violation — skipping SMS/email")
