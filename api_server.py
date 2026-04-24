"""
api_server.py — DriveGuardAI Flask API Server
Entry point. Only REST routes live here.
All logic is delegated to focused modules:
  config.py            — environment & constants
  notifications.py     — SMS (Twilio) + email
  detector.py          — YOLO + face recognition
  violation_handler.py — screenshot, audio, DB, alerts
  monitor.py           — camera loop + MJPEG stream
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import threading
import time
from datetime import datetime

import cv2
import requests
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

import config
import detector
import monitor
import notifications
import violation_handler as vh

app = Flask(__name__)
CORS(app)

# ── Ensure required directories exist ────────────────────
os.makedirs(config.SCREENSHOTS_DIR,  exist_ok=True)
os.makedirs(config.DRIVER_FACES_DIR, exist_ok=True)
print(f"Screenshots dir: {config.SCREENSHOTS_DIR}")

# ── Session & violation state ─────────────────────────────
_monitoring_thread = None

current_session = {
    'driver_id': None, 'vehicle_id': None, 'trip_id':    None,
    'start_time': None, 'driver_name': None,
    'vehicle_name': None, 'trip_name': None,
}

last_violation_times = {
    'drowsy': 0, 'phone_use': 0, 'distracted': 0,
    'no_seatbelt': 0, 'yawning': 0,
    'smoking': 0, 'eating': 0, 'drinking': 0, 'unknown_driver': 0,
}


# ════════════════════════════════════════════════════════════
# STREAM
# ════════════════════════════════════════════════════════════

@app.route('/api/stream')
def video_stream():
    return Response(
        monitor.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


# ════════════════════════════════════════════════════════════
# MONITORING
# ════════════════════════════════════════════════════════════

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    global _monitoring_thread
    if monitor.monitoring_active_flag[0]:
        return jsonify({'error': 'Already active', 'session': current_session}), 400

    data = request.get_json() or {}
    if not all(k in data for k in ('driver_id', 'vehicle_id', 'trip_id')):
        return jsonify({'error': 'Missing: driver_id, vehicle_id, trip_id'}), 400

    current_session.update({
        'driver_id':    data['driver_id'],
        'vehicle_id':   data['vehicle_id'],
        'trip_id':      data['trip_id'],
        'start_time':   datetime.now().isoformat(),
        'driver_name':  data.get('driver_name',  ''),
        'vehicle_name': data.get('vehicle_name', ''),
        'trip_name':    data.get('trip_name',    ''),
    })
    monitor.monitoring_active_flag[0] = True
    _monitoring_thread = threading.Thread(
        target=monitor.monitoring_loop,
        args=(current_session, last_violation_times),
        daemon=True,
    )
    _monitoring_thread.start()
    print(f"Session started: {current_session}")
    return jsonify({
        'message':    'Monitoring started',
        'session':    current_session,
        'stream_url': 'http://localhost:5000/api/stream',
    }), 200


@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    if not monitor.monitoring_active_flag[0]:
        return jsonify({'error': 'No active session'}), 400

    monitor.monitoring_active_flag[0] = False
    with monitor.frame_lock:
        monitor.latest_frame_jpg = None
    if _monitoring_thread and _monitoring_thread.is_alive():
        _monitoring_thread.join(timeout=5)

    session_data = current_session.copy()
    current_session.update({
        'driver_id': None, 'vehicle_id': None, 'trip_id':    None,
        'start_time': None, 'driver_name': None,
        'vehicle_name': None, 'trip_name': None,
    })
    print("Monitoring stopped cleanly")
    return jsonify({'message': 'Monitoring stopped', 'session': session_data}), 200


@app.route('/api/monitoring/status', methods=['GET'])
def get_status():
    return jsonify({
        'monitoring_active': monitor.monitoring_active_flag[0],
        'current_session':   current_session,
        'driver_verified':   monitor.current_driver_verified,
        'last_violations':   last_violation_times,
        'stream_url':        'http://localhost:5000/api/stream',
    })


# ════════════════════════════════════════════════════════════
# HEALTH
# ════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':            'healthy',
        'monitoring_active': monitor.monitoring_active_flag[0],
        'tts_available':     vh.TTS_AVAILABLE,
        'face_recog':        detector.FACE_RECOGNITION_AVAILABLE,
        'yolo_loaded':       detector.yolo_net is not None,
        'twilio_configured': config.TWILIO_ACCOUNT_SID != "YOUR_TWILIO_ACCOUNT_SID",
        'known_drivers':     len(detector.known_face_names),
        'screenshots_dir':   config.SCREENSHOTS_DIR,
        'screenshots_exist': os.path.exists(config.SCREENSHOTS_DIR),
        'stream_url':        'http://localhost:5000/api/stream',
        'timestamp':         datetime.now().isoformat(),
    })


@app.route('/api/test-java-connection', methods=['GET'])
def test_java_connection():
    try:
        r = requests.get(config.JAVA_BACKEND_URL + "/api/v1/drivers", timeout=3)
        return jsonify({'connected': True, 'status_code': r.status_code})
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
# SCREENSHOTS
# ════════════════════════════════════════════════════════════

@app.route('/api/screenshots', methods=['GET'])
def list_screenshots():
    try:
        files     = sorted(os.listdir(config.SCREENSHOTS_DIR), reverse=True)
        jpg_files = [f for f in files if f.endswith('.jpg')]
        return jsonify({'total': len(jpg_files), 'screenshots': jpg_files[:50],
                        'path': config.SCREENSHOTS_DIR})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/screenshots/<path:filename>', methods=['GET'])
def serve_screenshot(filename):
    filepath = os.path.join(config.SCREENSHOTS_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': f'File not found: {filename}'}), 404
    response = send_from_directory(config.SCREENSHOTS_DIR, filename, mimetype='image/jpeg')
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response


# ════════════════════════════════════════════════════════════
# FACE REGISTRATION
# ════════════════════════════════════════════════════════════

@app.route('/api/drivers/register-face', methods=['POST'])
def register_driver_face():
    data = request.get_json() or {}
    if 'driver_id' not in data or 'driver_name' not in data:
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
    filename = f"{data['driver_id']}_{data['driver_name']}.jpg"
    cv2.imwrite(os.path.join(config.DRIVER_FACES_DIR, filename), frame)
    detector.load_known_faces()
    return jsonify({'message': 'Face registered', 'filename': filename}), 201


@app.route('/api/drivers/known-faces', methods=['GET'])
def list_known_faces():
    return jsonify({'total': len(detector.known_face_names),
                    'drivers': detector.known_face_names})


# ════════════════════════════════════════════════════════════
# SMS TEST
# ════════════════════════════════════════════════════════════

@app.route('/api/test-sms', methods=['POST'])
def test_sms():
    from twilio.rest import Client as TwilioClient
    if config.TWILIO_ACCOUNT_SID == "YOUR_TWILIO_ACCOUNT_SID":
        return jsonify({'error': 'Fill in Twilio credentials in .env'}), 500
    data  = request.get_json() or {}
    phone = data.get('phone', config.MANAGER_PHONE)
    try:
        client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg    = client.messages.create(
            body=f"DriveGuardAI TEST SMS\nSMS working correctly.\nTime: {now}",
            from_=config.TWILIO_FROM_NUMBER,
            to=phone,
        )
        print(f"Test SMS sent — SID: {msg.sid}")
        return jsonify({'success': True, 'sid': msg.sid, 'status': msg.status, 'to': phone})
    except Exception as e:
        print(f"Test SMS failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════
# YOLO TEST
# ════════════════════════════════════════════════════════════

@app.route('/api/test-yolo', methods=['GET'])
def test_yolo():
    import numpy as np
    try:
        with open(config.YOLO_NAMES) as f:
            actual_classes = [l.strip() for l in f]
    except Exception as e:
        return jsonify({'error': f'Cannot read coco.names: {e}'}), 500

    if detector.yolo_net is None:
        return jsonify({'error': 'YOLO not loaded',
                        'classes': len(actual_classes)}), 500

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open camera'}), 500
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': 'Camera failed'}), 500

    all_detected = []
    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        detector.yolo_net.setInput(blob)
        outputs = detector.yolo_net.forward(detector.yolo_output_layers)
        for output in outputs:
            for det in output:
                scores     = det[5:]
                cid        = int(np.argmax(scores))
                confidence = float(scores[cid])
                if confidence < 0.1:
                    continue
                label = actual_classes[cid] if cid < len(actual_classes) else f"cls_{cid}"
                all_detected.append({'label': label, 'confidence': round(confidence, 3)})
        all_detected.sort(key=lambda x: x['confidence'], reverse=True)
    except Exception as e:
        return jsonify({'error': f'YOLO failed: {e}'}), 500

    return jsonify({
        'status':      'YOLO running',
        'total_classes': len(actual_classes),
        'detections':  all_detected[:20],
        'drinking':    [d for d in all_detected if d['label'].lower() in config.DRINKING_OBJECTS],
        'eating':      [d for d in all_detected if d['label'].lower() in config.EATING_OBJECTS],
        'phone':       [d for d in all_detected if d['label'].lower() in config.PHONE_OBJECTS],
    })


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("DRIVEGUARDAI - AI MONITORING SERVICE")
    print("=" * 65)
    print(f"Flask  : http://localhost:5000")
    print(f"Stream : http://localhost:5000/api/stream")
    print(f"Java   : {config.JAVA_BACKEND_URL}")
    print("=" * 65)
    detector.load_yolo()
    detector.load_known_faces()
    print("Endpoints:")
    for method, route in [
        ("GET",  "/api/health"),
        ("GET",  "/api/stream"),
        ("POST", "/api/monitoring/start"),
        ("POST", "/api/monitoring/stop"),
        ("GET",  "/api/monitoring/status"),
        ("POST", "/api/test-sms"),
        ("GET",  "/api/test-yolo"),
        ("GET",  "/api/screenshots"),
        ("GET",  "/api/screenshots/<filename>"),
        ("POST", "/api/drivers/register-face"),
        ("GET",  "/api/drivers/known-faces"),
    ]:
        print(f"  {method:<5} {route}")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
