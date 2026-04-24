"""
detector.py — DriveGuardAI Object & Face Detection
Handles YOLO (smoking/eating/drinking/phone) and face recognition.
"""
import os
import time

import cv2
import numpy as np

import config

# ── YOLO globals ──────────────────────────────────────────
yolo_net          = None
yolo_classes      = []
yolo_output_layers = []

# ── Face recognition globals ──────────────────────────────
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

known_face_encodings = []
known_face_names     = []


# ── YOLO ─────────────────────────────────────────────────

def load_yolo() -> bool:
    global yolo_net, yolo_classes, yolo_output_layers
    if not os.path.exists(config.YOLO_WEIGHTS):
        print(f"YOLO weights not found at {config.YOLO_WEIGHTS}")
        return False
    try:
        yolo_net = cv2.dnn.readNet(config.YOLO_WEIGHTS, config.YOLO_CFG)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(config.YOLO_NAMES) as f:
            yolo_classes = [line.strip() for line in f]
        layer_names        = yolo_net.getLayerNames()
        yolo_output_layers = [
            layer_names[i - 1]
            for i in yolo_net.getUnconnectedOutLayers().flatten()
        ]
        print("YOLO loaded successfully")
        return True
    except Exception as e:
        print(f"YOLO load error: {e}")
        return False


def detect_objects(frame: np.ndarray) -> dict:
    """
    Run YOLO on a frame.
    Returns dict: {smoking, eating, drinking, phone_use} → bool
    """
    result = {'smoking': False, 'eating': False, 'drinking': False, 'phone_use': False}
    if yolo_net is None:
        print("DEBUG YOLO: model not loaded")
        return result

    try:
        print(f"DEBUG YOLO: Running inference on frame {frame.shape}...")
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_output_layers)

        all_detected = []
        for output in outputs:
            for det in output:
                scores     = det[5:]
                cid        = int(np.argmax(scores))
                confidence = float(scores[cid])
                if confidence < 0.3:
                    continue
                label = yolo_classes[cid].lower() if cid < len(yolo_classes) else "unknown"
                all_detected.append(f"{label}({round(confidence, 2)})")

                if confidence < 0.5:
                    continue
                if label in config.SMOKING_OBJECTS:
                    print(f"DEBUG YOLO: SMOKING — {label} ({round(confidence,2)})")
                    result['smoking'] = True
                if label in config.EATING_OBJECTS:
                    print(f"DEBUG YOLO: EATING — {label} ({round(confidence,2)})")
                    result['eating'] = True
                if label in config.DRINKING_OBJECTS:
                    print(f"DEBUG YOLO: DRINKING — {label} ({round(confidence,2)})")
                    result['drinking'] = True
                if label in config.PHONE_OBJECTS:
                    print(f"DEBUG YOLO: PHONE — {label} ({round(confidence,2)})")
                    result['phone_use'] = True

        detected_str = ", ".join(all_detected) if all_detected else "none"
        print(f"DEBUG YOLO: Detected (>0.3): {detected_str}")
        print(f"DEBUG YOLO: Total COCO classes: {len(yolo_classes)}")
        if yolo_classes:
            print(f"DEBUG YOLO: Sample: {yolo_classes[:10]}")

    except Exception as e:
        import traceback
        print(f"YOLO inference ERROR: {e}")
        traceback.print_exc()

    return result


# ── Face Recognition ──────────────────────────────────────

def load_known_faces() -> None:
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return
    known_face_encodings, known_face_names = [], []
    for fn in os.listdir(config.DRIVER_FACES_DIR):
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img  = face_recognition.load_image_file(
                os.path.join(config.DRIVER_FACES_DIR, fn)
            )
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(os.path.splitext(fn)[0])
        except Exception as e:
            print(f"Face load error {fn}: {e}")
    print(f"Known drivers loaded: {len(known_face_names)}")


def verify_driver_face(frame: np.ndarray, expected_driver_id) -> tuple:
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


def verify_face_at_start(expected_driver_id) -> bool:
    if not FACE_RECOGNITION_AVAILABLE:
        return True
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
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
    return True
