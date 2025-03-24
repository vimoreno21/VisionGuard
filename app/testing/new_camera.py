import os
import cv2
import time
from dotenv import load_dotenv
from utils.photo_utils import current_timestamp

# Get env variables
load_dotenv()
RTSP_URL_BASE = os.getenv("RTSP_URL_BASE")
CAM_NUM = os.getenv("FRONT_DOOR_CAM")
CAM2_NUM = os.getenv("STAIRS_CAM")
RTSP_URL = f"{RTSP_URL_BASE}{CAM2_NUM}"

from utils.directories import DEBUG_DIR

def setup_camera():
    """Set up the camera with retry logic using GStreamer pipeline"""
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            print(f"Attempting to connect to camera: {RTSP_URL}")
            
            # Use GStreamer pipeline for RTSP - more reliable for Jetson
            gst_str = (
                f"rtspsrc location={RTSP_URL} latency=0 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! " 
                "videoconvert ! video/x-raw,format=BGR ! appsink"
            )
            
            # Create VideoCapture with GStreamer pipeline
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
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

def create_video_writer(filename, width, height, fps=30):
    """Create a VideoWriter using GStreamer with H.264 encoding"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # GStreamer pipeline for H.264 encoding
    gst_out = (
        f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 "
        f"speed-preset=superfast ! h264parse ! mp4mux ! "
        f"filesink location={filename}"
    )
    
    # Create the writer
    out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Failed to create video writer")
        return None
        
    return out

def record_video(duration=10):
    """Record video for specified duration in seconds"""
    cap = setup_camera()
    if not cap:
        return False
    
    # Get video properties
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
        
    height, width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default if we can't get actual FPS
    
    # Create output filename with timestamp
    timestamp = current_timestamp()
    output_file = os.path.join(DEBUG_DIR, f"recording_{timestamp}.mp4")
    
    # Create video writer
    out = create_video_writer(output_file, width, height, fps)
    if not out:
        cap.release()
        return False
    
    print(f"Recording to {output_file} for {duration} seconds...")
    
    # Calculate end time
    end_time = time.time() + duration
    
    # Record loop
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write the frame
        out.write(frame)
    
    # Clean up
    out.release()
    cap.release()
    
    print(f"✅ Recording saved to {output_file}")
    return True