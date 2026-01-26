"""
live_camera_test.py - Test camera with live video feed and motion detection
Save this as: python-service/live_camera_test.py
Run: python live_camera_test.py
"""
import cv2
import numpy as np

def test_live_camera():
    """Test webcam with live video feed and motion detection"""
    print("Starting live camera test...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'm' to toggle motion detection")
    print("  - Press 's' to save current frame")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        return False
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("📹 Live feed starting... (press 'q' to quit)\n")
    
    # Motion detection variables
    motion_detection = False
    prev_frame = None
    frame_count = 0
    motion_threshold = 25
    min_contour_area = 500
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        display_frame = frame.copy()
        
        # Motion detection logic
        if motion_detection and prev_frame is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            
            # Compute difference
            frame_diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    continue
                
                motion_detected = True
                # Draw rectangle around motion
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display motion status
            if motion_detected:
                cv2.putText(display_frame, "MOTION DETECTED!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Store current frame for next iteration
        prev_frame = frame.copy()
        
        # Add info overlay
        status_text = "Motion Detection: ON" if motion_detection else "Motion Detection: OFF"
        cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, display_frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Live Camera Feed - Press Q to quit, M for motion detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Quitting...")
            break
        elif key == ord('m'):
            motion_detection = not motion_detection
            status = "ON" if motion_detection else "OFF"
            print(f"Motion detection: {status}")
        elif key == ord('s'):
            filename = f"camera_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed!")
    return True

if __name__ == '__main__':
    test_live_camera()
