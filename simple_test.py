"""
simple_test.py - Test AI detection without Flask or Spring Boot
Save this file as: python-service/simple_test.py
Run: python simple_test.py

This allows you to test drowsiness and phone detection independently
"""

import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
from ultralytics import YOLO
import numpy as np

# Configuration
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 48


class SimpleDetector:
    def __init__(self):
        print("Loading models...")
        
        # Load face detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Load YOLO for phone detection
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Counters
        self.ear_counter = 0
        
        print("✅ Models loaded!\n")
        print("Instructions:")
        print("  - Close eyes for 2 seconds → Drowsiness alert")
        print("  - Show phone to camera → Phone detection")
        print("  - Press 'q' to quit\n")
    
    def eye_aspect_ratio(self, eye):
        """Calculate EAR"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def detect_drowsiness(self, frame, gray):
        """Detect drowsiness"""
        faces = self.detector(gray, 0)
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eyes
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Check drowsiness
            if ear < EAR_THRESHOLD:
                self.ear_counter += 1
                
                if self.ear_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]),
                                (0, 0, 255), 3)
            else:
                self.ear_counter = 0
            
            # Display EAR
            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def detect_phone(self, frame):
        """Detect phone"""
        results = self.yolo_model(frame, verbose=False, conf=0.5)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                if class_id == 67:  # Phone class
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"PHONE {confidence:.2f}",
                               (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.putText(frame, "PHONE USAGE DETECTED!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print("🎥 Camera started - Detection running...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Run detections
            frame = self.detect_drowsiness(frame, gray)
            frame = self.detect_phone(frame)
            
            # Show instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('DriveGuard AI - Simple Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed!")


if __name__ == '__main__':
    detector = SimpleDetector()
    detector.run()
