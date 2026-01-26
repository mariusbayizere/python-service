"""
driver_monitoring_system_v2_simple.py - Advanced Driver Monitoring System
Works without MediaPipe - Uses dlib and OpenCV only
Features: Drowsiness, Distraction, Phone Use, Seatbelt Detection + Audio Alerts
Save as: python-service/driver_monitoring_system_v2_simple.py
Run: python driver_monitoring_system_v2_simple.py
"""
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time
from collections import deque
import pygame
import math

class DriverMonitoringSystem:
    def __init__(self):
        print("Loading AI models...")
        
        # Dlib face detector and landmarks
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Initialize pygame for audio
        pygame.mixer.init()
        self.create_alert_sounds()
        
        # Detection thresholds - IMPROVED FOR BETTER ACCURACY
        self.EAR_THRESHOLD = 0.21  # Lower = more sensitive
        self.EAR_CONSEC_FRAMES = 8  # Reduced from 20 for faster detection
        self.MAR_THRESHOLD = 0.5   # Mouth aspect ratio for yawning
        self.HEAD_YAW_THRESHOLD = 20   # degrees (looking left/right)
        self.HEAD_PITCH_THRESHOLD = 15  # degrees (looking up/down)
        
        # Counters
        self.ear_counter = 0
        self.yawn_counter = 0
        self.distraction_counter = 0
        self.no_seatbelt_counter = 0
        self.phone_counter = 0
        
        # Alert states
        self.alerts = {
            'drowsy': False,
            'eyes_closed': False,
            'yawning': False,
            'distracted': False,
            'phone_use': False,
            'no_seatbelt': False
        }
        
        # Last alert times (to prevent spam)
        self.last_sound_time = {key: 0 for key in self.alerts.keys()}
        self.sound_cooldown = 3  # seconds between same alert
        
        # Previous frame for motion detection
        self.prev_gray = None
        
        # Seatbelt detection using skin color (simplified)
        self.seatbelt_check_counter = 0
        
    def create_alert_sounds(self):
        """Create simple beep sounds for alerts"""
        try:
            sample_rate = 22050
            duration = 0.3
            
            def generate_beep(frequency):
                n_samples = int(sample_rate * duration)
                buf = np.sin(2 * np.pi * frequency * np.linspace(0, duration, n_samples))
                buf = (buf * 32767).astype(np.int16)
                stereo = np.column_stack((buf, buf))
                return pygame.sndarray.make_sound(stereo)
            
            # Different alert sounds
            self.sounds = {
                'drowsy': generate_beep(800),      # High pitch for drowsiness
                'distracted': generate_beep(600),  # Medium pitch
                'phone': generate_beep(1000),      # Very high pitch
                'seatbelt': generate_beep(500),    # Low pitch
                'critical': generate_beep(1200)    # Urgent sound
            }
            print("✓ Audio alerts enabled")
        except Exception as e:
            print(f"Warning: Could not create sounds: {e}")
            self.sounds = {}
    
    def play_alert_sound(self, alert_type):
        """Play alert sound with cooldown"""
        current_time = time.time()
        if current_time - self.last_sound_time.get(alert_type, 0) > self.sound_cooldown:
            sound_map = {
                'drowsy': 'drowsy',
                'eyes_closed': 'critical',
                'yawning': 'drowsy',
                'distracted': 'distracted',
                'phone_use': 'phone',
                'no_seatbelt': 'seatbelt'
            }
            sound_key = sound_map.get(alert_type, 'critical')
            if sound_key in self.sounds:
                self.sounds[sound_key].play()
            self.last_sound_time[alert_type] = current_time
    
    def eye_aspect_ratio(self, eye):
        """Calculate Eye Aspect Ratio (EAR)"""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calculate Mouth Aspect Ratio (MAR)"""
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar
    
    def get_head_pose(self, shape, frame_width, frame_height):
        """Estimate head pose"""
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),    # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye
            (shape.part(45).x, shape.part(45).y),  # Right eye
            (shape.part(48).x, shape.part(48).y),  # Left mouth
            (shape.part(54).x, shape.part(54).y)   # Right mouth
        ], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])
        
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        return pitch, yaw, roll
    
    def detect_phone_use_advanced(self, frame, face_rect):
        """Advanced phone detection using motion and color analysis"""
        x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
        
        # Define region of interest (near ear/face)
        ear_regions = [
            # Left ear region
            (max(0, x - 80), max(0, y), x, y + h // 2),
            # Right ear region
            (x + w, max(0, y), min(frame.shape[1], x + w + 80), y + h // 2)
        ]
        
        phone_detected = False
        
        for (rx1, ry1, rx2, ry2) in ear_regions:
            if rx2 <= rx1 or ry2 <= ry1:
                continue
                
            roi = frame[ry1:ry2, rx1:rx2]
            
            if roi.size == 0:
                continue
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Detect skin color (hand/phone holding)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Also detect dark objects (phones are usually dark)
            lower_dark = np.array([0, 0, 0], dtype=np.uint8)
            upper_dark = np.array([180, 255, 80], dtype=np.uint8)
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(skin_mask, dark_mask)
            
            # Calculate ratio of detected pixels
            detection_ratio = np.sum(combined_mask > 0) / combined_mask.size
            
            if detection_ratio > 0.25:  # 25% of region has object
                phone_detected = True
                # Draw detection box
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                cv2.putText(frame, "HAND/PHONE", (rx1, ry1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return phone_detected
    
    def detect_seatbelt_simple(self, frame, face_rect):
        """Simplified seatbelt detection using edge detection"""
        h, w = frame.shape[:2]
        
        # Check area across chest (below face)
        chest_y1 = min(h - 1, face_rect.bottom() + 20)
        chest_y2 = min(h - 1, chest_y1 + 150)
        chest_x1 = max(0, face_rect.left() - 50)
        chest_x2 = min(w - 1, face_rect.right() + 50)
        
        if chest_y2 <= chest_y1 or chest_x2 <= chest_x1:
            return True  # Can't check, assume wearing
        
        chest_roi = frame[chest_y1:chest_y2, chest_x1:chest_x2]
        
        if chest_roi.size == 0:
            return True
        
        # Convert to grayscale and detect edges
        gray_chest = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_chest, 50, 150)
        
        # Detect diagonal lines (seatbelt strap)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # Check for diagonal lines (seatbelt crosses diagonally)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                
                # Seatbelt is typically at 30-60 degree angle
                if 25 < angle < 65:
                    # Draw detected seatbelt
                    cv2.line(frame, 
                            (chest_x1 + x1, chest_y1 + y1), 
                            (chest_x1 + x2, chest_y1 + y2), 
                            (0, 255, 0), 2)
                    return True  # Seatbelt detected
        
        # Draw chest checking area
        cv2.rectangle(frame, (chest_x1, chest_y1), (chest_x2, chest_y2), (255, 100, 0), 1)
        
        return False  # No seatbelt detected
    
    def process_frame(self, frame):
        """Main processing function"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            cv2.putText(frame, "NO DRIVER DETECTED!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return frame
        
        # Process first face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Draw face rectangle
        cv2.rectangle(frame, 
                     (face.left(), face.top()), 
                     (face.right(), face.bottom()), 
                     (0, 255, 0), 2)
        
        # Extract features
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)]
        
        # Draw eye contours
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        
        # Calculate metrics
        left_ear = self.eye_aspect_ratio(np.array(left_eye))
        right_ear = self.eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0
        mar = self.mouth_aspect_ratio(np.array(mouth))
        pitch, yaw, roll = self.get_head_pose(landmarks, frame.shape[1], frame.shape[0])
        
        # DROWSINESS DETECTION - IMPROVED
        if ear < self.EAR_THRESHOLD:
            self.ear_counter += 1
            if self.ear_counter >= self.EAR_CONSEC_FRAMES:
                self.alerts['eyes_closed'] = True
                self.alerts['drowsy'] = True
                self.play_alert_sound('eyes_closed')
        else:
            if self.ear_counter > 0:
                self.ear_counter = 0
            self.alerts['eyes_closed'] = False
            if ear > self.EAR_THRESHOLD + 0.05:
                self.alerts['drowsy'] = False
        
        # YAWNING DETECTION
        if mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= 3:
                self.alerts['yawning'] = True
                self.play_alert_sound('yawning')
        else:
            if self.yawn_counter > 0:
                self.yawn_counter -= 1
            if mar < self.MAR_THRESHOLD - 0.1:
                self.alerts['yawning'] = False
        
        # DISTRACTION DETECTION - IMPROVED
        if abs(yaw) > self.HEAD_YAW_THRESHOLD or abs(pitch) > self.HEAD_PITCH_THRESHOLD:
            self.distraction_counter += 1
            if self.distraction_counter >= 5:
                self.alerts['distracted'] = True
                self.play_alert_sound('distracted')
        else:
            if self.distraction_counter > 0:
                self.distraction_counter -= 1
            if abs(yaw) < self.HEAD_YAW_THRESHOLD - 5:
                self.alerts['distracted'] = False
        
        # PHONE USE DETECTION - IMPROVED
        phone_detected = self.detect_phone_use_advanced(frame, face)
        if phone_detected:
            self.phone_counter += 1
            if self.phone_counter >= 5:
                self.alerts['phone_use'] = True
                self.play_alert_sound('phone_use')
        else:
            if self.phone_counter > 0:
                self.phone_counter -= 1
            if self.phone_counter == 0:
                self.alerts['phone_use'] = False
        
        # SEATBELT DETECTION
        has_seatbelt = self.detect_seatbelt_simple(frame, face)
        if not has_seatbelt:
            self.no_seatbelt_counter += 1
            if self.no_seatbelt_counter >= 15:
                self.alerts['no_seatbelt'] = True
                self.play_alert_sound('no_seatbelt')
        else:
            if self.no_seatbelt_counter > 0:
                self.no_seatbelt_counter -= 1
            if self.no_seatbelt_counter == 0:
                self.alerts['no_seatbelt'] = False
        
        # Draw visualizations
        self.draw_alerts(frame, ear, mar, pitch, yaw, roll)
        
        return frame
    
    def draw_alerts(self, frame, ear, mar, pitch, yaw, roll):
        """Draw enhanced alert overlays"""
        h, w = frame.shape[:2]
        
        # Status panel
        panel_h = 320
        cv2.rectangle(frame, (10, 10), (450, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, panel_h), (255, 255, 255), 2)
        
        y_pos = 45
        spacing = 35
        
        # Title
        cv2.putText(frame, "DRIVER MONITORING SYSTEM", (20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += spacing + 10
        
        # Metrics with color coding
        ear_color = (0, 0, 255) if ear < self.EAR_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Eye Ratio: {ear:.3f} {'DROWSY!' if ear < self.EAR_THRESHOLD else 'OK'}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 2)
        y_pos += spacing
        
        mar_color = (255, 165, 0) if mar > self.MAR_THRESHOLD else (255, 255, 255)
        cv2.putText(frame, f"Mouth: {mar:.2f} {'YAWN' if mar > self.MAR_THRESHOLD else ''}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 1)
        y_pos += spacing
        
        head_color = (0, 140, 255) if (abs(yaw) > self.HEAD_YAW_THRESHOLD or abs(pitch) > self.HEAD_PITCH_THRESHOLD) else (255, 255, 255)
        cv2.putText(frame, f"Head: Yaw={yaw:.1f} Pitch={pitch:.1f}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 1)
        y_pos += spacing + 10
        
        # ALERTS with clear warnings
        alert_displayed = False
        
        if self.alerts['eyes_closed'] or self.alerts['drowsy']:
            cv2.putText(frame, ">>> DROWSY - WAKE UP! <<<", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += spacing
            alert_displayed = True
        
        if self.alerts['yawning']:
            cv2.putText(frame, "YAWNING - TAKE BREAK", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            y_pos += spacing
            alert_displayed = True
        
        if self.alerts['distracted']:
            cv2.putText(frame, "DISTRACTED - FOCUS AHEAD!", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            y_pos += spacing
            alert_displayed = True
        
        if self.alerts['phone_use']:
            cv2.putText(frame, ">>> PHONE USE - ILLEGAL! <<<", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y_pos += spacing
            alert_displayed = True
        
        if self.alerts['no_seatbelt']:
            cv2.putText(frame, "NO SEATBELT - WEAR IT!", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            y_pos += spacing
            alert_displayed = True
        
        # Overall status
        if not alert_displayed:
            cv2.putText(frame, "STATUS: SAFE DRIVING", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STATUS: VIOLATIONS!", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    print("=" * 70)
    print("ADVANCED DRIVER MONITORING SYSTEM V2 - Starting...")
    print("=" * 70)
    print("\nFeatures:")
    print("  ✓ Drowsiness Detection (Improved - Faster Response)")
    print("  ✓ Distraction Detection (Head Pose Tracking)")
    print("  ✓ Phone Use Detection (Color & Motion Analysis)")
    print("  ✓ Seatbelt Detection (Edge Detection)")
    print("  ✓ Yawning Detection (Fatigue Indicator)")
    print("  ✓ Audio Alerts (Different sounds for violations)")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("=" * 70)
    
    dms = DriverMonitoringSystem()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n✅ Camera opened!")
    print("📹 Monitoring started... (Press 'q' to quit)\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Process frame
        processed_frame = dms.process_frame(frame)
        
        # Add FPS counter
        cv2.putText(processed_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Driver Monitoring System V2 - Press Q to quit', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Shutting down...")
            break
        elif key == ord('s'):
            filename = f"dms_screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("✅ System stopped successfully!")

if __name__ == '__main__':
    main()
