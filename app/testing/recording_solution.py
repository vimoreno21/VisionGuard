import cv2
import os
import time
from datetime import datetime

def current_timestamp():
    """Generate a timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_camera():
    """Set up the camera connection - using the method we know works"""
    rtsp_url = "rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/501"
    print(f"[INFO] Setting up camera connection...")
    print(f"Attempting to connect to camera: {rtsp_url}")
    
    # Use the FFMPEG backend that we know works
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        # Read a test frame
        ret, frame = cap.read()
        if ret:
            # Get dimensions and other properties
            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default if can't determine actual FPS
                
            print(f"[INFO] Original camera resolution: {width}x{height}")
            print(f"[INFO] Original camera FPS: {fps}")
            
            # Scale down for recording if needed (for better performance)
            scale_factor = 0.3  # Adjust as needed
            rec_width = int(width * scale_factor)
            rec_height = int(height * scale_factor)
            print(f"[INFO] Reduced resolution for recording: {rec_width}x{rec_height}")
            
            return cap, rec_width, rec_height, fps
    
    print("[ERROR] Failed to connect to camera")
    return None, None, None, None

def record_video(duration=10, output_dir="/tmp"):
    """Record video from the camera"""
    # Setup camera
    cap, width, height, fps = setup_camera()
    if not cap:
        return False
    
    # Create output file
    os.makedirs(output_dir, exist_ok=True)
    timestamp = current_timestamp()
    output_file = os.path.join(output_dir, f"recording_{timestamp}.mp4")
    
    # Try different codec options (in order of preference)
    codecs_to_try = [
        ('avc1', "H.264 codec (avc1)"),
        ('H264', "H.264 codec (H264)"),
        ('X264', "H.264 codec (X264)"),
        ('XVID', "XVID codec (fallback)"),  
        ('MJPG', "MJPG codec (highly compatible fallback)")
    ]
    
    out = None
    for codec, codec_name in codecs_to_try:
        print(f"[INFO] Trying {codec_name}...")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        if out.isOpened():
            print(f"[INFO] Successfully initialized {codec_name}")
            break
        else:
            print(f"[INFO] {codec_name} failed to initialize")
            out = None
    
    if not out:
        print("[ERROR] Could not initialize any video codec")
        cap.release()
        return False
    
    print(f"[INFO] Recording to {output_file} for {duration} seconds...")
    
    # Record frames until duration is reached
    start_time = time.time()
    frame_count = 0
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera")
            break
        
        # Resize if needed
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        frame_count += 1
        
        # Show progress every second
        if frame_count % int(fps) == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Recording: {elapsed:.1f}s / {duration}s ({frame_count} frames)")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"[INFO] Recording complete: {output_file}")
    print(f"[INFO] Recorded {frame_count} frames ({frame_count/duration:.1f} FPS average)")
    return True

if __name__ == "__main__":
    record_video(duration=10)