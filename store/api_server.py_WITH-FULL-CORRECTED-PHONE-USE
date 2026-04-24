"""
api_server.py - DriveGuardAI Full Featured Version
Fixes:
  1. Screenshot debug statements added
  2. SMS uses Driver name, Vehicle model, Trip name (professional messages)
  3. SMS fires automatically on CRITICAL violations during monitoring
  4. Phone use detection fixed (PHONE_USE enum)
  5. Smoking/eating mapped correctly
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time
import requests
from datetime import datetime
import numpy as np

try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)
    _tts_engine.setProperty('volume', 1.0)
    TTS_AVAILABLE = True
    print("TTS engine ready")
except Exception as e:
    TTS_AVAILABLE = False
    print("pyttsx3 not available: " + str(e))

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("twilio not available - pip install twilio")

from driver_monitoring_system_v3_simple import DriverMonitoringSystem

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION — loaded from .env file
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/home/marius/python-service/.env")
    print("Loaded config from .env file")
except ImportError:
    print("python-dotenv not installed — pip install python-dotenv")

import os as _os

JAVA_BACKEND_URL  = _os.getenv("JAVA_BACKEND_URL",  "http://localhost:8080")
SCREENSHOTS_DIR   = _os.getenv("SCREENSHOTS_DIR",   "/home/marius/python-service/violations_screenshots")
DRIVER_FACES_DIR  = _os.getenv("DRIVER_FACES_DIR",  "/home/marius/python-service/driver_faces")

TWILIO_ACCOUNT_SID = _os.getenv("TWILIO_ACCOUNT_SID", "YOUR_TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = _os.getenv("TWILIO_AUTH_TOKEN",  "YOUR_TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = _os.getenv("TWILIO_FROM_NUMBER", "+1XXXXXXXXXX")
MANAGER_PHONE      = _os.getenv("MANAGER_PHONE",      "+250787667755")

YOLO_WEIGHTS = _os.getenv("YOLO_WEIGHTS", "/home/marius/python-service/yolo/yolov3.weights")
YOLO_CFG     = _os.getenv("YOLO_CFG",     "/home/marius/python-service/yolo/yolov3.cfg")
YOLO_NAMES   = _os.getenv("YOLO_NAMES",   "/home/marius/python-service/yolo/coco.names")

VIOLATION_COOLDOWN = int(_os.getenv("VIOLATION_COOLDOWN", "10"))

# ============================================================
# DEBUG: Verify screenshots directory on startup
# ============================================================
print("=" * 50)
print("DEBUG: Checking screenshots directory...")
print("DEBUG: SCREENSHOTS_DIR = " + SCREENSHOTS_DIR)
if os.path.exists(SCREENSHOTS_DIR):
    files = os.listdir(SCREENSHOTS_DIR)
    print("DEBUG: Directory EXISTS — " + str(len(files)) + " files found")
else:
    print("DEBUG: Directory does NOT exist — creating it now...")
    try:
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        print("DEBUG: Directory created successfully")
    except Exception as e:
        print("DEBUG: FAILED to create directory: " + str(e))

os.makedirs(DRIVER_FACES_DIR, exist_ok=True)
print("=" * 50)

# ============================================================
# GLOBAL STATE
# ============================================================
monitoring_active       = False
monitoring_thread       = None
dms_instance            = None
camera_capture          = None
yolo_net                = None
yolo_classes            = []
yolo_output_layers      = []
known_face_encodings    = []
known_face_names        = []
current_driver_verified = False
latest_frame_jpg        = None
frame_lock              = threading.Lock()

current_session = {
    'driver_id':    None,
    'vehicle_id':   None,
    'trip_id':      None,
    'start_time':   None,
    'driver_name':  None,
    'vehicle_name': None,
    'trip_name':    None,
}

last_violation_times = {
    'drowsy': 0, 'phone_use': 0, 'distracted': 0,
    'no_seatbelt': 0, 'yawning': 0,
    'smoking': 0, 'eating': 0, 'drinking': 0, 'unknown_driver': 0,
}

# ============================================================
# MJPEG STREAM
# ============================================================

def generate_frames():
    while True:
        with frame_lock:
            jpg = latest_frame_jpg
        if jpg is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            blank[:] = (30, 30, 30)
            cv2.putText(blank, "Waiting for session...", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            _, buf = cv2.imencode('.jpg', blank)
            jpg = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.033)

@app.route('/api/stream')
def video_stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================
# AUDIO ALERT
# ============================================================

_audio_lock = threading.Lock()

def play_audio_alert(alert_name):
    def _run():
        # Do not play audio if monitoring has already stopped
        if not monitoring_active:
            return
        with _audio_lock:
            if not monitoring_active:
                return
            messages = {
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
            msg = messages.get(alert_name, "Warning! " + alert_name.replace('_', ' ') + " detected.")
            print("AUDIO: " + msg)
            if TTS_AVAILABLE:
                try:
                    _tts_engine.say(msg)
                    _tts_engine.runAndWait()
                except Exception as e:
                    print("TTS error: " + str(e))
    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# TWILIO SMS — professional messages with names
# ============================================================

def send_sms_alert(alert_name, driver_id, trip_id):
    """
    Send SMS for CRITICAL violations only.
    Uses driver name, vehicle name, trip name for professional messages.
    """
    if not TWILIO_AVAILABLE:
        print("SMS: Twilio library not installed")
        return
    if TWILIO_ACCOUNT_SID == "YOUR_TWILIO_ACCOUNT_SID":
        print("SMS: Twilio not configured")
        return

    critical = {'drowsy', 'eyes_closed', 'phone_use', 'unknown_driver'}
    if alert_name not in critical:
        return

    # Use names from session for professional message
    driver_name  = current_session.get('driver_name')  or "Driver #" + str(driver_id)
    vehicle_name = current_session.get('vehicle_name') or "Vehicle #" + str(current_session.get('vehicle_id', '?'))
    trip_name    = current_session.get('trip_name')    or "Trip #" + str(trip_id)

    violation_labels = {
        'drowsy':         'DROWSINESS',
        'eyes_closed':    'EYES CLOSED',
        'phone_use':      'PHONE USE WHILE DRIVING',
        'unknown_driver': 'UNRECOGNISED DRIVER',
    }
    violation_label = violation_labels.get(alert_name, alert_name.replace('_', ' ').upper())

    def _run():
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            lines = [
                "DriveGuardAI CRITICAL ALERT",
                "",
                "Driver " + driver_name + " was detected",
                "doing " + violation_label,
                "while on " + trip_name + " in " + vehicle_name + ".",
                "",
                "Time: " + now,
                "Please take immediate action."
            ]
            body = "\n".join(lines)
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            msg = client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=MANAGER_PHONE)
            print("SMS sent — SID: " + msg.sid + " | Alert: " + alert_name)
            print("SMS message: Driver " + driver_name + " — " + violation_label)
        except Exception as e:
            print("SMS error: " + str(e))
    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# EMAIL VIA JAVA — professional messages with names
# ============================================================

def trigger_java_email(violation_data):
    def _run():
        try:
            driver_name  = current_session.get('driver_name')  or "Driver #" + str(violation_data['driver_id'])
            vehicle_name = current_session.get('vehicle_name') or "Vehicle #" + str(violation_data['vehicle_id'])
            trip_name    = current_session.get('trip_name')    or "Trip #" + str(violation_data['trip_id'])

            lines = [
                "A violation was detected by the AI monitoring system.",
                "",
                "Driver      : " + driver_name,
                "Vehicle     : " + vehicle_name,
                "Trip        : " + trip_name,
                "Violation   : " + str(violation_data['incident_type']),
                "Severity    : " + str(violation_data['severity']),
                "Description : " + str(violation_data['description']),
                "Time        : " + str(violation_data['timestamp']),
                "Screenshot  : " + str(violation_data.get('screenshot', 'N/A')),
                "",
                "Please log in to the DriveGuardAI dashboard to review."
            ]
            subject = "[DriveGuardAI] " + str(violation_data['incident_type']) + " — " + driver_name + " (" + str(violation_data['severity']) + ")"
            payload = {
                "recipient": "bayizeremarius119@gmail.com",
                "subject":   subject,
                "msgBody":   "\n".join(lines)
            }
            r = requests.post(JAVA_BACKEND_URL + "/api/v1/notifications/email", json=payload, timeout=10)
            print("Email triggered — status " + str(r.status_code))
        except Exception as e:
            print("Email error: " + str(e))
    threading.Thread(target=_run, daemon=True).start()

# ============================================================
# YOLO — SMOKING / EATING / DRINKING DETECTION
# ============================================================

def load_yolo():
    global yolo_net, yolo_classes, yolo_output_layers
    if not os.path.exists(YOLO_WEIGHTS):
        print("YOLO weights not found at " + YOLO_WEIGHTS)
        return False
    try:
        yolo_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(YOLO_NAMES) as f:
            yolo_classes = [l.strip() for l in f]
        layer_names = yolo_net.getLayerNames()
        yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
        print("YOLO loaded successfully")
        return True
    except Exception as e:
        print("YOLO load error: " + str(e))
        return False

# COCO class names that indicate smoking, eating, drinking
# NOTE: "cigarette" is NOT in COCO 80 classes — smoking uses indirect indicators
SMOKING_OBJECTS  = {
    "cigarette", "cigar",         # not in COCO but kept for custom models
}

EATING_OBJECTS   = {
    # Food items in COCO
    "sandwich", "hot dog", "pizza", "donut", "cake",
    "apple", "banana", "orange", "broccoli", "carrot",
    # Utensils — if driver holds fork/knife/spoon they are eating
    "fork", "knife", "spoon",
}

DRINKING_OBJECTS = {
    # Containers in COCO
    "bottle", "wine glass", "cup", "bowl",
    # vase sometimes confused with bottle
    "vase",
}

# Phone detection — COCO labels for phone/phone-like objects
# "remote" is often detected instead of "cell phone" at low resolution
PHONE_OBJECTS = {
    "cell phone", "remote", "mouse",
}

def detect_smoking_eating(frame):  # also detects phone use via YOLO
    result = {'smoking': False, 'eating': False, 'drinking': False, 'phone_use': False}

    if yolo_net is None:
        print("DEBUG YOLO: yolo_net is None — model not loaded!")
        return result

    try:
        print("DEBUG YOLO: Running inference on frame " + str(frame.shape) + "...")
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_output_layers)

        all_detected = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                cid = int(np.argmax(scores))
                confidence = float(scores[cid])
                if confidence < 0.3:   # lower threshold to see more detections
                    continue
                label = yolo_classes[cid].lower() if cid < len(yolo_classes) else "unknown"
                all_detected.append(label + "(" + str(round(confidence, 2)) + ")")

                if confidence < 0.5:
                    continue

                if label in SMOKING_OBJECTS:
                    print("DEBUG YOLO: SMOKING detected — " + label + " confidence=" + str(round(confidence, 2)))
                    result['smoking'] = True
                if label in EATING_OBJECTS:
                    print("DEBUG YOLO: EATING detected — " + label + " confidence=" + str(round(confidence, 2)))
                    result['eating'] = True
                if label in DRINKING_OBJECTS:
                    print("DEBUG YOLO: DRINKING detected — " + label + " confidence=" + str(round(confidence, 2)))
                    result['drinking'] = True

                if label in PHONE_OBJECTS:
                    print("DEBUG YOLO: PHONE detected — " + label + " confidence=" + str(round(confidence, 2)))
                    result['phone_use'] = True

        if all_detected:
            print("DEBUG YOLO: All objects detected (>0.3): " + ", ".join(all_detected))
        else:
            print("DEBUG YOLO: No objects detected above 0.3 confidence")

        print("DEBUG YOLO: SMOKING classes we look for: " + str(SMOKING_OBJECTS))
        print("DEBUG YOLO: EATING classes we look for:  " + str(EATING_OBJECTS))
        print("DEBUG YOLO: DRINKING classes we look for: " + str(DRINKING_OBJECTS))
        print("DEBUG YOLO: Total COCO classes loaded: " + str(len(yolo_classes)))
        if yolo_classes:
            print("DEBUG YOLO: Sample classes: " + str(yolo_classes[:10]))

    except Exception as e:
        print("YOLO inference ERROR: " + str(e))
        import traceback
        traceback.print_exc()

    return result

# ============================================================
# FACE RECOGNITION
# ============================================================

def load_known_faces():
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return
    known_face_encodings, known_face_names = [], []
    for fn in os.listdir(DRIVER_FACES_DIR):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img = face_recognition.load_image_file(os.path.join(DRIVER_FACES_DIR, fn))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(os.path.splitext(fn)[0])
        except Exception as e:
            print("Face load error " + fn + ": " + str(e))
    print("Known drivers loaded: " + str(len(known_face_names)))

def verify_driver_face(frame, expected_driver_id):
    if not FACE_RECOGNITION_AVAILABLE or not known_face_encodings:
        return True, "unknown"
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    for enc in encs:
        matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
        dists   = face_recognition.face_distance(known_face_encodings, enc)
        best    = int(np.argmin(dists))
        if matches[best]:
            name = known_face_names[best]
            if str(expected_driver_id) == name.split("_")[0]:
                return True, name
            return False, name
        return False, "unknown"
    return True, "no_face"

def run_face_verification_at_start(expected_driver_id):
    global current_driver_verified
    if not FACE_RECOGNITION_AVAILABLE:
        current_driver_verified = True
        return True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        current_driver_verified = True
        return True
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        if not ret:
            break
        ok, name = verify_driver_face(frame, expected_driver_id)
        if ok and name not in ("unknown", "no_face"):
            break
        time.sleep(0.3)
    cap.release()
    current_driver_verified = True
    return True

# ============================================================
# HELPERS
# ============================================================

def send_violation_to_java(violation_data):
    try:
        r = requests.post(JAVA_BACKEND_URL + "/api/v1/incidents/violation",
                          json=violation_data, timeout=10)
        if r.status_code in [200, 201]:
            print("Violation saved to database successfully")
            return True
        print("Java returned " + str(r.status_code) + ": " + r.text)
        return False
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Java — is Spring Boot running on port 8080?")
        return False
    except Exception as e:
        print("send_violation error: " + str(e))
        return False

def save_screenshot(frame, violation_type):
    """Save screenshot with full debug logging."""
    print("DEBUG screenshot: Attempting to save for violation: " + violation_type)
    print("DEBUG screenshot: Target directory: " + SCREENSHOTS_DIR)

    # Check directory exists
    if not os.path.exists(SCREENSHOTS_DIR):
        print("DEBUG screenshot: Directory missing — recreating...")
        try:
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            print("DEBUG screenshot: Directory recreated")
        except Exception as e:
            print("DEBUG screenshot: FAILED to create directory: " + str(e))
            return None

    try:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = violation_type + "_" + ts + ".jpg"
        filepath = os.path.join(SCREENSHOTS_DIR, filename)

        print("DEBUG screenshot: Writing file to: " + filepath)

        # Check frame is valid
        if frame is None:
            print("DEBUG screenshot: ERROR — frame is None!")
            return None
        if frame.size == 0:
            print("DEBUG screenshot: ERROR — frame is empty!")
            return None

        print("DEBUG screenshot: Frame shape: " + str(frame.shape))

        success = cv2.imwrite(filepath, frame)

        if success:
            size = os.path.getsize(filepath)
            print("DEBUG screenshot: SUCCESS — " + filename + " (" + str(size) + " bytes)")
            return filename
        else:
            print("DEBUG screenshot: FAILED — cv2.imwrite returned False")
            print("DEBUG screenshot: Check if directory has write permissions")
            return None

    except Exception as e:
        print("DEBUG screenshot: EXCEPTION — " + str(e))
        import traceback
        traceback.print_exc()
        return None

def map_alert_to_incident_type(a):
    mapping = {
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
    return mapping.get(a, 'OTHER')

def map_alert_to_severity(a):
    if a in {'drowsy', 'eyes_closed', 'phone_use', 'unknown_driver'}:
        return 'CRITICAL'
    if a in {'distracted', 'no_seatbelt', 'smoking'}:
        return 'HIGH'
    return 'MEDIUM'

def handle_violation(frame, alert_name):
    """Central handler — screenshot + audio + DB + SMS + email."""
    print("VIOLATION DETECTED: " + alert_name)

    screenshot = save_screenshot(frame, alert_name)

    driver_name  = current_session.get('driver_name')  or "Driver #" + str(current_session['driver_id'])
    vehicle_name = current_session.get('vehicle_name') or "Vehicle #" + str(current_session['vehicle_id'])
    trip_name    = current_session.get('trip_name')    or "Trip #" + str(current_session['trip_id'])

    description = (
        driver_name + " was detected doing " +
        alert_name.replace('_', ' ') +
        " while on " + trip_name +
        " in " + vehicle_name
    )

    data = {
        'driver_id':     current_session['driver_id'],
        'vehicle_id':    current_session['vehicle_id'],
        'trip_id':       current_session['trip_id'],
        'incident_type': map_alert_to_incident_type(alert_name),
        'severity':      map_alert_to_severity(alert_name),
        'description':   description,
        'timestamp':     datetime.now().isoformat(),
        'screenshot':    screenshot,
    }

    # 1. Audio alert
    play_audio_alert(alert_name)

    # 2. Save to database
    if send_violation_to_java(data):
        last_violation_times[alert_name] = time.time()
        # 3. SMS for CRITICAL
        send_sms_alert(alert_name, current_session['driver_id'], current_session['trip_id'])
        # 4. Email
        trigger_java_email(data)
    else:
        print("Failed to save violation — skipping SMS/email")

# ============================================================
# MONITORING LOOP
# ============================================================

def monitoring_loop():
    global monitoring_active, dms_instance, camera_capture, latest_frame_jpg

    driver_name = current_session.get('driver_name') or str(current_session['driver_id'])
    print("Monitoring started for: " + driver_name)

    run_face_verification_at_start(current_session['driver_id'])

    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        print("Cannot open camera!")
        monitoring_active = False
        return

    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened — streaming to /api/stream")

    dms_instance   = DriverMonitoringSystem()
    frame_count    = 0
    fps_start_time = time.time()
    fps            = 0
    yolo_interval  = 30   # run every 30 frames (~1s) for testing

    while monitoring_active:
        ret, frame = camera_capture.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()

        try:
            processed_frame = dms_instance.process_frame(frame)
        except Exception as e:
            print("DMS error: " + str(e))
            processed_frame = frame

        # Show driver name on stream instead of ID
        driver_label = current_session.get('driver_name') or "Driver #" + str(current_session['driver_id'])
        trip_label   = current_session.get('trip_name')   or "Trip #" + str(current_session['trip_id'])
        label = "FPS:" + str(round(fps, 1)) + " | " + driver_label + " | " + trip_label
        cv2.putText(processed_frame, label,
                    (10, processed_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        _, jpg_buf = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame_jpg = jpg_buf.tobytes()

        if frame_count % 30 == 0:
            now = time.time()
            for alert_name, is_active in dms_instance.alerts.items():
                if is_active and now - last_violation_times.get(alert_name, 0) > VIOLATION_COOLDOWN:
                    handle_violation(frame, alert_name)

            if frame_count % yolo_interval == 0 and yolo_net is not None:
                yolo_results = detect_smoking_eating(frame)
                for behaviour, detected in yolo_results.items():
                    if detected and now - last_violation_times.get(behaviour, 0) > VIOLATION_COOLDOWN:
                        print("YOLO VIOLATION: " + behaviour)
                        handle_violation(frame, behaviour)

            if frame_count % 9000 == 0 and FACE_RECOGNITION_AVAILABLE and known_face_encodings:
                ok, _ = verify_driver_face(frame, current_session['driver_id'])
                if not ok and now - last_violation_times.get('unknown_driver', 0) > 60:
                    handle_violation(frame, 'unknown_driver')

        time.sleep(0.033)

    if camera_capture:
        camera_capture.release()
    with frame_lock:
        latest_frame_jpg = None
    print("Monitoring stopped")

# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':            'healthy',
        'monitoring_active': monitoring_active,
        'tts_available':     TTS_AVAILABLE,
        'face_recog':        FACE_RECOGNITION_AVAILABLE,
        'yolo_loaded':       yolo_net is not None,
        'twilio_configured': TWILIO_ACCOUNT_SID != "YOUR_TWILIO_ACCOUNT_SID",
        'known_drivers':     len(known_face_names),
        'screenshots_dir':   SCREENSHOTS_DIR,
        'screenshots_exist': os.path.exists(SCREENSHOTS_DIR),
        'stream_url':        'http://localhost:5000/api/stream',
        'timestamp':         datetime.now().isoformat(),
    })

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    global monitoring_active, monitoring_thread, current_session
    if monitoring_active:
        return jsonify({'error': 'Already active', 'session': current_session}), 400
    data = request.get_json()
    if not all(k in data for k in ['driver_id', 'vehicle_id', 'trip_id']):
        return jsonify({'error': 'Missing: driver_id, vehicle_id, trip_id'}), 400
    current_session.update({
        'driver_id':    data['driver_id'],
        'vehicle_id':   data['vehicle_id'],
        'trip_id':      data['trip_id'],
        'start_time':   datetime.now().isoformat(),
        'driver_name':  data.get('driver_name', ''),
        'vehicle_name': data.get('vehicle_name', ''),
        'trip_name':    data.get('trip_name', ''),
    })
    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    print("Session started: " + str(current_session))
    return jsonify({'message': 'Monitoring started', 'session': current_session,
                    'stream_url': 'http://localhost:5000/api/stream'}), 200

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    global monitoring_active, current_session, camera_capture, latest_frame_jpg
    if not monitoring_active:
        return jsonify({'error': 'No active session'}), 400
    monitoring_active = False
    if camera_capture and camera_capture.isOpened():
        camera_capture.release()
    with frame_lock:
        latest_frame_jpg = None
    if monitoring_thread and monitoring_thread.is_alive():
        monitoring_thread.join(timeout=5)
    session_data    = current_session.copy()
    current_session = {
        'driver_id': None, 'vehicle_id': None, 'trip_id': None,
        'start_time': None, 'driver_name': None,
        'vehicle_name': None, 'trip_name': None,
    }
    print("Monitoring stopped cleanly")
    return jsonify({'message': 'Monitoring stopped', 'session': session_data}), 200

@app.route('/api/monitoring/status', methods=['GET'])
def get_status():
    return jsonify({
        'monitoring_active': monitoring_active,
        'current_session':   current_session,
        'driver_verified':   current_driver_verified,
        'last_violations':   last_violation_times,
        'stream_url':        'http://localhost:5000/api/stream',
    })

@app.route('/api/test-java-connection', methods=['GET'])
def test_java_connection():
    try:
        r = requests.get(JAVA_BACKEND_URL + "/api/v1/drivers", timeout=3)
        return jsonify({'connected': True, 'status_code': r.status_code})
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)}), 500

@app.route('/api/drivers/register-face', methods=['POST'])
def register_driver_face():
    data = request.get_json()
    if not data or 'driver_id' not in data or 'driver_name' not in data:
        return jsonify({'error': 'driver_id and driver_name required'}), 400
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open camera'}), 500
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': 'Failed to capture frame'}), 500
    filename = str(data['driver_id']) + "_" + str(data['driver_name']) + ".jpg"
    cv2.imwrite(os.path.join(DRIVER_FACES_DIR, filename), frame)
    load_known_faces()
    return jsonify({'message': 'Face registered', 'filename': filename}), 201

@app.route('/api/drivers/known-faces', methods=['GET'])
def list_known_faces():
    return jsonify({'total': len(known_face_names), 'drivers': known_face_names})

@app.route('/api/screenshots', methods=['GET'])
def list_screenshots():
    try:
        files = sorted(os.listdir(SCREENSHOTS_DIR), reverse=True)
        jpg_files = [f for f in files if f.endswith('.jpg')]
        return jsonify({
            'total':       len(jpg_files),
            'screenshots': jpg_files[:50],
            'path':        SCREENSHOTS_DIR
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-sms', methods=['POST'])
def test_sms():
    if not TWILIO_AVAILABLE:
        return jsonify({'error': 'pip install twilio'}), 500
    if TWILIO_ACCOUNT_SID == "YOUR_TWILIO_ACCOUNT_SID":
        return jsonify({'error': 'Fill in Twilio credentials in api_server.py'}), 500
    data  = request.get_json() or {}
    phone = data.get('phone', MANAGER_PHONE)
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines  = [
            "DriveGuardAI TEST SMS",
            "SMS notifications are working correctly.",
            "Time: " + now,
        ]
        msg = client.messages.create(
            body="\n".join(lines),
            from_=TWILIO_FROM_NUMBER,
            to=phone
        )
        print("Test SMS sent — SID: " + msg.sid)
        return jsonify({'success': True, 'sid': msg.sid, 'status': msg.status, 'to': phone})
    except Exception as e:
        print("Test SMS failed: " + str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# MAIN
# ============================================================


@app.route('/api/test-yolo', methods=['GET'])
def test_yolo():
    """
    Test YOLO detection by taking one photo from camera.
    Call this while holding a bottle, food, or phone in front of camera.
    GET http://localhost:5000/api/test-yolo
    """
    if yolo_net is None:
        return jsonify({'error': 'YOLO not loaded', 'yolo_weights_path': YOLO_WEIGHTS,
                        'weights_exist': os.path.exists(YOLO_WEIGHTS)}), 500

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open camera'}), 500

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Failed to capture frame'}), 500

    # Save test frame
    test_path = os.path.join(SCREENSHOTS_DIR, "yolo_test_frame.jpg")
    cv2.imwrite(test_path, frame)

    # Run YOLO with very low threshold to see ALL detections
    detected_all   = []
    detected_high  = []

    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_output_layers)

        for output in outputs:
            for det in output:
                scores     = det[5:]
                cid        = int(np.argmax(scores))
                confidence = float(scores[cid])
                if confidence < 0.1:
                    continue
                label = yolo_classes[cid].lower() if cid < len(yolo_classes) else "unknown_" + str(cid)
                entry = {'label': label, 'confidence': round(confidence, 3)}
                detected_all.append(entry)
                if confidence >= 0.5:
                    detected_high.append(entry)

        # Sort by confidence
        detected_all.sort(key=lambda x: x['confidence'], reverse=True)
        detected_high.sort(key=lambda x: x['confidence'], reverse=True)

    except Exception as e:
        return jsonify({'error': 'YOLO inference failed: ' + str(e)}), 500

    # Check what was found
    found_smoking  = [d for d in detected_high if d['label'] in SMOKING_OBJECTS]
    found_eating   = [d for d in detected_high if d['label'] in EATING_OBJECTS]
    found_drinking = [d for d in detected_high if d['label'] in DRINKING_OBJECTS]

    return jsonify({
        'status':            'YOLO is working',
        'total_classes':     len(yolo_classes),
        'frame_shape':       str(frame.shape),
        'test_image_saved':  test_path,
        'detections_above_50pct':  detected_high[:20],
        'detections_above_10pct':  detected_all[:30],
        'smoking_found':     found_smoking,
        'eating_found':      found_eating,
        'drinking_found':    found_drinking,
        'smoking_classes_we_check':  list(SMOKING_OBJECTS),
        'eating_classes_we_check':   list(EATING_OBJECTS),
        'drinking_classes_we_check': list(DRINKING_OBJECTS),
        'phone_found':    [d for d in detected_high if d['label'] in {'cell phone','remote','mouse'}],
        'phone_classes_we_check': ['cell phone', 'remote', 'mouse'],
        'tip': 'Hold a bottle/cup=drinking | apple/banana=eating | phone/remote=phone_use in front of camera',
        'note_smoking': 'cigarette/cigar NOT in COCO 80 classes — smoking cannot be detected by standard YOLOv3',
    })

if __name__ == '__main__':
    print("=" * 65)
    print("DRIVEGUARDAI - AI MONITORING SERVICE")
    print("=" * 65)
    print("Flask   : http://localhost:5000")
    print("Stream  : http://localhost:5000/api/stream")
    print("Java    : " + JAVA_BACKEND_URL)
    print("=" * 65)
    load_yolo()
    load_known_faces()
    print("Endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/stream")
    print("  POST /api/monitoring/start")
    print("  POST /api/monitoring/stop")
    print("  GET  /api/monitoring/status")
    print("  POST /api/test-sms")
    print("  GET  /api/screenshots")
    print("  POST /api/drivers/register-face")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
