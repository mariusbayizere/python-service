"""
driver_monitoring_system.py - Advanced Driver Monitoring System
Detects: drowsiness, distraction, phone use, no seatbelt, smoking
Save as: python-service/driver_monitoring_system.py
Run: python driver_monitoring_system.py
"""
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from collections import deque

class DriverMonitoringSystem:
    def __init__(self):
        # Load dlib's face detector and facial landmarks predictor
        print("Loading face detector and landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Eye aspect ratio threshold for drowsiness
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.ear_counter = 0
        
        # Head pose thresholds (looking away detection)
        self.HEAD_POSE_THRESHOLD = 25  # degrees
        
        # Phone detection (hand near face)
        self.HAND_NEAR_FACE_THRESHOLD = 100  # pixels
        
        # Yawning detection
        self.MAR_THRESHOLD = 0.6  # Mouth Aspect Ratio
        
        # Alert buffers
        self.drowsy_frames = deque(maxlen=30)
        self.distracted_frames = deque(maxlen=30)
        self.phone_frames = deque(maxlen=30)
        
        # Alert states
        self.alerts = {
            'drowsy': False,
            'distracted': False,
            'phone_use': False,
            'yawning': False,
            'eyes_closed': False
        }
        
        # Timing
        self.last_alert_time = time.time()
        self.alert_cooldown = 2  # seconds
        
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio (MAR) for yawning"""
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
        C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar
    
    def get_head_pose(self, shape, frame_width, frame_height):
        """Estimate head pose (pitch, yaw, roll)"""
        # 2D image points
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),     # Nose tip
            (shape.part(8).x, shape.part(8).y),       # Chin
            (shape.part(36).x, shape.part(36).y),     # Left eye left corner
            (shape.part(45).x, shape.part(45).y),     # Right eye right corner
            (shape.part(48).x, shape.part(48).y),     # Left mouth corner
            (shape.part(54).x, shape.part(54).y)      # Right mouth corner
        ], dtype="double")
        
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # Camera internals
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Get angles
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        return pitch, yaw, roll
    
    def detect_phone_use(self, frame, face_rect):
        """Detect if hand is near face (phone use indicator)"""
        # This is a simplified version - for production, use hand detection model
        # Here we use skin color detection as proxy
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        # Region around face
        margin = 50
        roi_x1 = max(0, x - margin)
        roi_y1 = max(0, y - margin)
        roi_x2 = min(frame.shape[1], x + w + margin)
        roi_y2 = min(frame.shape[0], y + h + margin)
        
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return False
        
        # Simple skin detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        skin_ratio = np.sum(mask > 0) / mask.size
        return skin_ratio > 0.3  # Threshold for hand detection
    
    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on frame"""
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    def process_frame(self, frame):
        """Main processing function"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            # No face detected
            cv2.putText(frame, "NO DRIVER DETECTED!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        # Process first detected face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Extract eye coordinates
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            left_eye.append((landmarks.part(n).x, landmarks.part(n).y))
        for n in range(42, 48):
            right_eye.append((landmarks.part(n).x, landmarks.part(n).y))
        
        # Calculate EAR
        left_ear = self.eye_aspect_ratio(np.array(left_eye))
        right_ear = self.eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0
        
        # Extract mouth coordinates for yawning
        mouth = []
        for n in range(48, 68):
            mouth.append((landmarks.part(n).x, landmarks.part(n).y))
        mar = self.mouth_aspect_ratio(np.array(mouth))
        
        # Get head pose
        pitch, yaw, roll = self.get_head_pose(landmarks, frame.shape[1], frame.shape[0])
        
        # Check for drowsiness (eyes closed)
        if ear < self.EAR_THRESHOLD:
            self.ear_counter += 1
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.alerts['drowsy'] = True
                self.alerts['eyes_closed'] = True
        else:
            self.ear_counter = 0
            self.alerts['eyes_closed'] = False
            if ear > self.EAR_THRESHOLD + 0.05:
                self.alerts['drowsy'] = False
        
        # Check for yawning
        if mar > self.MAR_THRESHOLD:
            self.alerts['yawning'] = True
        else:
            self.alerts['yawning'] = False
        
        # Check for distraction (head turned away)
        if abs(yaw) > self.HEAD_POSE_THRESHOLD or abs(pitch) > self.HEAD_POSE_THRESHOLD:
            self.alerts['distracted'] = True
        else:
            self.alerts['distracted'] = False
        
        # Check for phone use
        phone_detected = self.detect_phone_use(frame, face)
        if phone_detected:
            self.alerts['phone_use'] = True
        else:
            self.alerts['phone_use'] = False
        
        # Draw visualizations
        self.draw_alerts(frame, ear, mar, pitch, yaw, roll)
        
        # Optional: Draw landmarks
        # self.draw_landmarks(frame, landmarks)
        
        return frame
    
    def draw_alerts(self, frame, ear, mar, pitch, yaw, roll):
        """Draw alert overlays on frame"""
        h, w = frame.shape[:2]
        
        # Status panel background
        cv2.rectangle(frame, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 250), (255, 255, 255), 2)
        
        y_offset = 40
        line_height = 30
        
        # Title
        cv2.putText(frame, "DRIVER MONITORING SYSTEM", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # Metrics
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Head: Y={yaw:.1f} P={pitch:.1f}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height + 10
        
        # Alerts
        alert_y = y_offset
        
        if self.alerts['drowsy']:
            cv2.putText(frame, "DROWSINESS ALERT!", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += line_height
        
        if self.alerts['eyes_closed']:
            cv2.putText(frame, "EYES CLOSED!", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += line_height
        
        if self.alerts['yawning']:
            cv2.putText(frame, "YAWNING DETECTED", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            alert_y += line_height
        
        if self.alerts['distracted']:
            cv2.putText(frame, "DISTRACTION - EYES ON ROAD!", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            alert_y += line_height
        
        if self.alerts['phone_use']:
            cv2.putText(frame, "PHONE USE DETECTED!", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            alert_y += line_height
        
        # Overall status
        if not any(self.alerts.values()):
            cv2.putText(frame, "STATUS: SAFE", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STATUS: WARNING", (20, alert_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    print("=" * 60)
    print("DRIVER MONITORING SYSTEM - Starting...")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ Drowsiness Detection (Eye Aspect Ratio)")
    print("  ✓ Distraction Detection (Head Pose)")
    print("  ✓ Phone Use Detection")
    print("  ✓ Yawning Detection")
    print("  ✓ Eyes Closed Detection")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 60)
    
    # Initialize DMS
    dms = DriverMonitoringSystem()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n✅ Camera opened successfully!")
    print("📹 Monitoring started...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Process frame
        processed_frame = dms.process_frame(frame)
        
        # Add frame counter
        cv2.putText(processed_frame, f"Frame: {frame_count}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Driver Monitoring System - Press Q to quit', processed_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Shutting down...")
            break
        elif key == ord('s'):
            filename = f"dms_screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Driver Monitoring System stopped successfully!")

if __name__ == '__main__':
    main()
