import os
import cv2
import time
from dotenv import load_dotenv
from utils.directories import DEBUG_DIR
from utils.photo_utils import current_timestamp

# get env variables
load_dotenv()

RTSP_URL_BASE = os.getenv("RTSP_URL_BASE")
CAM_NUM = os.getenv("FRONT_DOOR_CAM")
CAM2_NUM = os.getenv("STAIRS_CAM")
RTSP_URL = f"{RTSP_URL_BASE}{CAM2_NUM}"

def setup_camera():
    """Set up the camera with retry logic"""
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Try to open the camera
            print(f"Attempting to connect to camera: {RTSP_URL}")
            
            # Open the RTSP stream with FFMPEG for better performance
            # cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            cap = cv2.VideoCapture("new_video.mp4")

            # Reduce OpenCV buffer to avoid stale frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Keep only 1 frame in buffe
            
            # Check if opened successfully
            if cap.isOpened():
                # Read a test frame
                ret, frame = cap.read()
                if ret:
                    # Print frame dimensions
                    height, width, channels = frame.shape
                    print(f"Frame dimensions: {width}x{height}, {channels} channels")
                    
                    return cap
                else:
                    print(f"❌ Camera opened but couldn't read frame (attempt {attempt+1}/{max_attempts})")
            else:
                print(f"❌ Failed to open camera (attempt {attempt+1}/{max_attempts})")
            
            # Close the connection before retry
            cap.release()
            
        except Exception as e:
            print(f"❌ Error connecting to camera: {e} (attempt {attempt+1}/{max_attempts})")
        
        # Wait before retry
        time.sleep(2)
        attempt += 1
    
    print("⛔ All camera connection attempts failed")
    return None
