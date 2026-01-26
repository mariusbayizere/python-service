"""
DriveGuard AI - Complete Driver Monitoring System
File: main_app.py
Save this file as: python-service/main_app.py
"""

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import requests
from datetime import datetime
import time
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
SPRING_BOOT_URL = "http://localhost:8080/api/incidents"
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 48
YAWN_THRESHOLD = 0.6
DISTRACTION_FRAMES = 72
PHONE_CONFIDENCE = 0.5
ALERT_COOLDOWN = 5


# ==================== DROWSINESS DETECTOR ====================
class DrowsinessDetector:
    """Detects drowsiness using Eye Aspect Ratio (EAR)"""
    
    def __init__(self):
        print("🔄 Loading drowsiness detection model...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.ear_frame_counter = 0
        self.yawn_frame_counter = 0
        self.drowsy_alert_active = False
        self.yawn_alert_active = False
        self.last_alert_time = 0
        print("✅ Drowsiness detector ready!")
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio (MAR) for yawning"""
        A = distance.euclidean(mouth[2], mouth[10])
        B = distance.euclidean(mouth[4], mouth[8])
        C = distance.euclidean(mouth[0], mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar
    
    def detect(self, frame, driver_id, vehicle_id):
        """Main drowsiness detection logic"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            self.ear_frame_counter = 0
            self.yawn_frame_counter = 0
            cv2.putText(frame, "NO FACE DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            mouth = shape[48:68]
            
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = self.mouth_aspect_ratio(mouth)
            
            # Draw eye contours
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            mouth_hull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)
            
            # DROWSINESS DETECTION
            if ear < EAR_THRESHOLD:
                self.ear_frame_counter += 1
                
                if self.ear_frame_counter >= EAR_CONSEC_FRAMES:
                    if not self.drowsy_alert_active:
                        self.drowsy_alert_active = True
                        self.send_alert(driver_id, vehicle_id, "DROWSINESS", 
                                      f"EAR: {ear:.3f}, Duration: 2+ seconds")
                    
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                                (0, 0, 255), 3)
            else:
                self.ear_frame_counter = 0
                self.drowsy_alert_active = False
            
            # YAWNING DETECTION
            if mar > YAWN_THRESHOLD:
                self.yawn_frame_counter += 1
                
                if self.yawn_frame_counter >= 15:
                    if not self.yawn_alert_active:
                        self.yawn_alert_active = True
                        self.send_alert(driver_id, vehicle_id, "YAWNING",
                                      f"MAR: {mar:.3f}, Fatigue indicator")
                    
                    cv2.putText(frame, "YAWNING DETECTED", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                self.yawn_frame_counter = 0
                self.yawn_alert_active = False
            
            # Display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (frame.shape[1] - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def send_alert(self, driver_id, vehicle_id, incident_type, details):
        """Send alert to Spring Boot"""
        current_time = time.time()
        
        if current_time - self.last_alert_time < ALERT_COOLDOWN:
            return
        
        self.last_alert_time = current_time
        
        try:
            alert_data = {
                "driverId": driver_id,
                "vehicleId": vehicle_id,
                "incidentType": incident_type,
                "timestamp": datetime.now().isoformat(),
                "severity": "HIGH",
                "details": details,
                "location": "Kigali, Rwanda"
            }
            
            response = requests.post(SPRING_BOOT_URL, json=alert_data, timeout=5)
            print(f"✅ Alert sent: {incident_type} - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending alert: {e}")


# ==================== PHONE DETECTOR ====================
class PhoneDetector:
    """Detects phone usage using YOLOv8"""
    
    def __init__(self):
        print("🔄 Loading phone detection model...")
        self.model = YOLO('yolov8n.pt')
        self.PHONE_CLASS_ID = 67
        self.last_alert_time = 0
        self.phone_detected_frames = 0
        self.PHONE_DETECTION_THRESHOLD = 15
        print("✅ Phone detector ready!")
    
    def detect(self, frame, driver_id, vehicle_id):
        """Detect phone in frame"""
        results = self.model(frame, verbose=False, conf=PHONE_CONFIDENCE)
        phone_detected = False
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == self.PHONE_CLASS_ID:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"PHONE {confidence:.2f}", 
                               (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if phone_detected:
            self.phone_detected_frames += 1
            
            if self.phone_detected_frames >= self.PHONE_DETECTION_THRESHOLD:
                current_time = time.time()
                if current_time - self.last_alert_time > ALERT_COOLDOWN:
                    self.send_alert(driver_id, vehicle_id, confidence)
                    self.last_alert_time = current_time
                
                cv2.putText(frame, "PHONE USAGE DETECTED!", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), 
                            (0, 0, 255), 3)
        else:
            self.phone_detected_frames = 0
        
        return frame
    
    def send_alert(self, driver_id, vehicle_id, confidence):
        """Send phone usage alert"""
        try:
            alert_data = {
                "driverId": driver_id,
                "vehicleId": vehicle_id,
                "incidentType": "PHONE_USAGE",
                "timestamp": datetime.now().isoformat(),
                "severity": "CRITICAL",
                "details": f"Phone detected with {confidence:.1%} confidence",
                "location": "Kigali, Rwanda"
            }
            
            requests.post(SPRING_BOOT_URL, json=alert_data, timeout=5)
            print(f"✅ Phone alert sent - Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")


# ==================== DISTRACTION DETECTOR ====================
class DistractionDetector:
    """Detects when driver is looking away"""
    
    def __init__(self):
        print("🔄 Loading distraction detection...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.distracted_frames = 0
        self.last_alert_time = 0
        print("✅ Distraction detector ready!")
    
    def detect(self, frame, driver_id, vehicle_id):
        """Detect if driver is distracted"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        if len(faces) == 0:
            self.distracted_frames += 1
            
            if self.distracted_frames > DISTRACTION_FRAMES:
                current_time = time.time()
                if current_time - self.last_alert_time > ALERT_COOLDOWN:
                    self.send_alert(driver_id, vehicle_id)
                    self.last_alert_time = current_time
                
                cv2.putText(frame, "DISTRACTION: Eyes Off Road!", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            self.distracted_frames = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.distracted_frames > 0:
            cv2.putText(frame, f"Looking away: {self.distracted_frames/24:.1f}s",
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        return frame
    
    def send_alert(self, driver_id, vehicle_id):
        """Send distraction alert"""
        try:
            alert_data = {
                "driverId": driver_id,
                "vehicleId": vehicle_id,
                "incidentType": "DISTRACTION",
                "timestamp": datetime.now().isoformat(),
                "severity": "HIGH",
                "details": f"Eyes off road for 3+ seconds",
                "location": "Kigali, Rwanda"
            }
            
            requests.post(SPRING_BOOT_URL, json=alert_data, timeout=5)
            print("✅ Distraction alert sent")
        except Exception as e:
            print(f"❌ Error: {e}")


# ==================== GLOBAL VARIABLES ====================
drowsiness_detector = None
phone_detector = None
distraction_detector = None
monitoring_active = False
current_driver_id = None
current_vehicle_id = None
camera = None


# ==================== FLASK ROUTES ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "DriveGuard AI",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    """Start driver monitoring"""
    global monitoring_active, current_driver_id, current_vehicle_id
    global drowsiness_detector, phone_detector, distraction_detector
    
    try:
        data = request.json
        current_driver_id = data.get('driverId')
        current_vehicle_id = data.get('vehicleId')
        
        if drowsiness_detector is None:
            drowsiness_detector = DrowsinessDetector()
            phone_detector = PhoneDetector()
            distraction_detector = DistractionDetector()
        
        monitoring_active = True
        
        print(f"🚀 Monitoring started for Driver {current_driver_id}, Vehicle {current_vehicle_id}")
        
        return jsonify({
            "status": "started",
            "driverId": current_driver_id,
            "vehicleId": current_vehicle_id,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    """Stop driver monitoring"""
    global monitoring_active
    
    monitoring_active = False
    print("🛑 Monitoring stopped")
    
    return jsonify({
        "status": "stopped",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/video-feed')
def video_feed():
    """Video streaming route"""
    def generate():
        global camera
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 24)
        
        print("📹 Camera started")
        
        while monitoring_active:
            ret, frame = camera.read()
            
            if not ret:
                print("❌ Failed to read frame")
                break
            
            try:
                frame = drowsiness_detector.detect(frame, current_driver_id, current_vehicle_id)
                frame = phone_detector.detect(frame, current_driver_id, current_vehicle_id)
                frame = distraction_detector.detect(frame, current_driver_id, current_vehicle_id)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, frame.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, f"Driver ID: {current_driver_id} | Vehicle: {current_vehicle_id}",
                           (10, frame.shape[0] - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            except Exception as e:
                print(f"⚠️ Detection error: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        camera.release()
        print("📹 Camera stopped")
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status', methods=['GET'])
def get_status():
    """Get monitoring status"""
    return jsonify({
        "monitoring": monitoring_active,
        "driverId": current_driver_id,
        "vehicleId": current_vehicle_id,
        "timestamp": datetime.now().isoformat(),
        "detectors": {
            "drowsiness": drowsiness_detector is not None,
            "phone": phone_detector is not None,
            "distraction": distraction_detector is not None
        }
    })


# ==================== MAIN ====================
if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         DriveGuard AI - Real-time Monitoring System       ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  ✅ Drowsiness Detection (EAR Algorithm)                  ║
    ║  ✅ Phone Usage Detection (YOLOv8)                        ║
    ║  ✅ Distraction Detection (Face Tracking)                 ║
    ║  ✅ Yawning Detection (MAR Algorithm)                     ║
    ║                                                           ║
    ║  Running on: http://localhost:5000                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
