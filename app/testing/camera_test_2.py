import cv2
import time
import os
import subprocess

def test_camera_connection():
    """Test different methods to connect to the camera"""
    rtsp_url = "rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/501"
    
    # Test 1: FFMPEG backend (your original approach)
    print("\n[TEST 1] Using FFMPEG backend (original approach)")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    success1 = test_connection(cap, "test1_ffmpeg")
    
    # Test 2: Default backend
    print("\n[TEST 2] Using default backend")
    cap = cv2.VideoCapture(rtsp_url)
    success2 = test_connection(cap, "test2_default")
    
    # Test 3: Direct ffmpeg command
    print("\n[TEST 3] Using direct ffmpeg command")
    test_file = f"/tmp/test3_ffmpeg_{int(time.time())}.jpg"
    success3 = test_ffmpeg_direct(rtsp_url, test_file)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Test 1 (FFMPEG backend): {'✓ SUCCESS' if success1 else '✗ FAILED'}")
    print(f"Test 2 (Default backend): {'✓ SUCCESS' if success2 else '✗ FAILED'}")
    print(f"Test 3 (Direct ffmpeg): {'✓ SUCCESS' if success3 else '✗ FAILED'}")

def test_connection(cap, test_name):
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
        
    print("✓ Camera opened")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        cap.release()
        return False
        
    # Success
    height, width = frame.shape[:2]
    print(f"✓ Successfully read frame: {width}x{height}")
    
    # Save a test image
    test_dir = "/tmp"
    os.makedirs(test_dir, exist_ok=True)
    test_file = f"{test_dir}/{test_name}_{int(time.time())}.jpg"
    cv2.imwrite(test_file, frame)
    print(f"✓ Saved test image to {test_file}")
    
    cap.release()
    return True

def test_ffmpeg_direct(rtsp_url, output_file):
    try:
        # Run ffmpeg to capture a single frame
        cmd = [
            'ffmpeg',
            '-y',
            '-i', rtsp_url,
            '-frames:v', '1',
            '-update', '1',
            output_file
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0 and os.path.exists(output_file):
            print(f"✓ Successfully captured frame with ffmpeg: {output_file}")
            return True
        else:
            print(f"❌ ffmpeg failed with error code {process.returncode}")
            print(stderr.decode())
            return False
    except Exception as e:
        print(f"❌ Error running ffmpeg: {e}")
        return False

if __name__ == "__main__":
    test_camera_connection()