"""
monitor.py — DriveGuardAI Monitoring Loop
Runs the camera feed, DMS processing, YOLO inference,
face verification, and violation dispatching.
"""
import time
import threading

import cv2
import numpy as np

import config
import detector
import violation_handler as vh

# ── Shared state (managed by api_server) ─────────────────
monitoring_active_flag = [False]   # mutable flag shared across threads
latest_frame_jpg       = None
frame_lock             = threading.Lock()
current_driver_verified = False


def _label_frame(frame: np.ndarray, fps: float, session: dict) -> np.ndarray:
    driver = session.get('driver_name') or f"Driver #{session['driver_id']}"
    trip   = session.get('trip_name')   or f"Trip #{session['trip_id']}"
    label  = f"FPS:{round(fps, 1)} | {driver} | {trip}"
    cv2.putText(
        frame, label,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
    )
    return frame


def monitoring_loop(current_session: dict, last_violation_times: dict) -> None:
    global latest_frame_jpg, current_driver_verified

    from driver_monitoring_system_v3_simple import DriverMonitoringSystem

    driver_name = current_session.get('driver_name') or str(current_session['driver_id'])
    print(f"Monitoring started for: {driver_name}")

    # Face verification at session start
    detector.verify_face_at_start(current_session['driver_id'])
    current_driver_verified = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        monitoring_active_flag[0] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera opened — streaming to /api/stream")

    dms            = DriverMonitoringSystem()
    frame_count    = 0
    fps_start_time = time.time()
    fps            = 0.0
    yolo_interval  = 30   # run YOLO every 30 frames (~1 s)

    while monitoring_active_flag[0]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        # FPS calculation
        if frame_count % 30 == 0:
            fps            = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()

        # DMS processing
        try:
            processed = dms.process_frame(frame)
        except Exception as e:
            print(f"DMS error: {e}")
            processed = frame

        processed = _label_frame(processed, fps, current_session)

        _, jpg_buf = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame_jpg = jpg_buf.tobytes()

        # Violation checks every 30 frames
        if frame_count % 30 == 0:
            now = time.time()

            # DMS alerts (drowsy, eyes_closed, phone_use, etc.)
            for alert_name, is_active in dms.alerts.items():
                if is_active and now - last_violation_times.get(alert_name, 0) > config.VIOLATION_COOLDOWN:
                    vh.handle_violation(
                        frame, alert_name,
                        current_session, last_violation_times,
                        monitoring_active_flag,
                    )

            # YOLO alerts (smoking, eating, drinking, phone_use)
            if frame_count % yolo_interval == 0 and detector.yolo_net is not None:
                yolo_results = detector.detect_objects(frame)
                for behaviour, detected in yolo_results.items():
                    if detected and now - last_violation_times.get(behaviour, 0) > config.VIOLATION_COOLDOWN:
                        print(f"YOLO VIOLATION: {behaviour}")
                        vh.handle_violation(
                            frame, behaviour,
                            current_session, last_violation_times,
                            monitoring_active_flag,
                        )

            # Periodic face re-verification (every 5 min)
            if (frame_count % 9000 == 0
                    and detector.FACE_RECOGNITION_AVAILABLE
                    and detector.known_face_encodings):
                ok, _ = detector.verify_driver_face(frame, current_session['driver_id'])
                if not ok and now - last_violation_times.get('unknown_driver', 0) > 60:
                    vh.handle_violation(
                        frame, 'unknown_driver',
                        current_session, last_violation_times,
                        monitoring_active_flag,
                    )

        time.sleep(0.033)

    cap.release()
    with frame_lock:
        latest_frame_jpg = None
    vh.drunk_detector.reset()
    print("Monitoring stopped")


def generate_frames():
    """MJPEG generator for /api/stream."""
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
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        time.sleep(0.033)
