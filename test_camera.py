"""
test_camera.py - Test if your webcam is working
Save this file as: python-service/test_camera.py
Run: python test_camera.py
"""

import cv2

def test_camera():
    """Test webcam access"""
    print("Testing camera...")
    
    # Try to open camera (0 is default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        print("Possible solutions:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera index: VideoCapture(1) or VideoCapture(2)")
        print("  3. Check camera permissions in system settings")
        print("  4. Close other apps using the camera")
        return False
    
    print("✅ Camera opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("❌ ERROR: Cannot read from camera!")
        cap.release()
        return False
    
    print("✅ Frame captured successfully!")
    print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]} pixels")
    print(f"   Color channels: {frame.shape[2]}")
    
    # Show the frame
    cv2.imshow('Camera Test - Press any key to close', frame)
    print("\n👁️ Camera window opened - press any key to close")
    cv2.waitKey(0)  # Wait for key press
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Camera test completed successfully!")
    return True


if __name__ == '__main__':
    test_camera()
