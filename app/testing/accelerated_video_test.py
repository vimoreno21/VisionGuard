import cv2
import os
import time
import signal
import numpy as np
from utils.directories import SAVE_DIR
from camera import setup_camera
from utils.photo_utils import current_timestamp

# Signal handler for cleaner termination
stop_program = False
def signal_handler(sig, frame):
    global stop_program
    print("\n[INFO] Shutdown signal received, finishing gracefully...")
    stop_program = True
    
# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Ensure the videos directory exists
video_save_path = os.path.join(SAVE_DIR, "videos")
os.makedirs(video_save_path, exist_ok=True)

# Use your function to set up the camera
print("[INFO] Setting up camera connection...")
cap = setup_camera()
if cap is None:
    print("⛔ Camera setup failed. Exiting...")
    exit()

# Get camera properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Original camera resolution: {frame_width}x{frame_height}")
print(f"[INFO] Original camera FPS: {original_fps}")

# OPTIMIZATION: Scale down the resolution for better performance
# For high-res cameras (>2MP), reducing more aggressively
scale_factor = 0.5 if frame_width * frame_height <= 2073600 else 0.3  # Adapt to input resolution
new_width = int(frame_width * scale_factor)
new_height = int(frame_height * scale_factor)

# Ensure dimensions are even (required for most codecs)
if new_width % 2 == 1:
    new_width -= 1
if new_height % 2 == 1:
    new_height -= 1

print(f"[INFO] Reduced resolution for recording: {new_width}x{new_height}")

# Output video path
output_video_path = os.path.join(video_save_path, f"camera_stream_{current_timestamp()}.mp4")

# Try different codecs until one works
codecs_to_try = [
    ('avc1', "H.264"),   # Modern H.264
    ('mp4v', "MPEG-4"),  # Fallback, widely compatible
    ('MJPG', "Motion JPEG"), # Most compatible, larger files
]

out = None
codec_name = ""

for codec, codec_desc in codecs_to_try:
    try:
        print(f"[INFO] Trying {codec_desc} codec...")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Use a more conservative frame rate
        target_fps = min(15.0, original_fps)  # Cap at 15fps for stability
        
        # Try to create the writer
        test_out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (new_width, new_height))
        
        # Test if the writer is properly initialized
        if test_out.isOpened():
            out = test_out
            codec_name = codec_desc
            print(f"[INFO] Successfully initialized {codec_desc} codec")
            break
        else:
            test_out.release()
            print(f"[INFO] {codec_desc} codec failed to initialize")
    except Exception as e:
        print(f"[WARNING] Error with {codec_desc} codec: {e}")

if out is None or not out.isOpened():
    print("[ERROR] Failed to initialize any VideoWriter. Exiting...")
    cap.release()
    exit()

print(f"📼 Recording video to: {output_video_path}")
print(f"[INFO] Using codec: {codec_name}")
print(f"[INFO] Target FPS: {target_fps}")
print(f"[INFO] Target resolution: {new_width}x{new_height}")

# Variables for statistics
frame_count = 0
processed_frames = 0
dropped_frames = 0
frame_times = []  # To track performance
start_time = time.time()
max_runtime = 60  # Record for 60 seconds
last_status_time = time.time()

print(f"[INFO] Beginning video recording. Will run for {max_runtime} seconds.")
print("[INFO] Press Ctrl+C once for clean shutdown.")

try:
    while cap.isOpened() and not stop_program:
        # Check if we've reached the maximum runtime
        if time.time() - start_time > max_runtime:
            print(f"[INFO] Maximum runtime of {max_runtime} seconds reached")
            break
            
        # Time the frame processing
        frame_start = time.time()
        
        # Read frame with timeout to handle RTSP stalls
        frame_read_success = False
        retry_count = 0
        max_retries = 3
        
        while not frame_read_success and retry_count < max_retries:
            ret, frame = cap.read()
            
            if ret:
                frame_read_success = True
            else:
                retry_count += 1
                print(f"[WARNING] Frame read retry {retry_count}/{max_retries}")
                time.sleep(0.1)  # Short delay before retry
        
        # If we still couldn't read a frame after retries
        if not frame_read_success:
            print("[WARNING] Failed to read frame after retries, continuing...")
            dropped_frames += 1
            
            # If too many consecutive failures, try to reset connection
            if dropped_frames > 10:
                print("[WARNING] Too many dropped frames, attempting to reset camera...")
                cap.release()
                time.sleep(1)
                cap = setup_camera()
                if cap is None:
                    print("[ERROR] Failed to reconnect to camera. Exiting...")
                    break
                dropped_frames = 0  # Reset counter
                
            continue
        
        # Process frame - resize to target resolution
        try:
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            processed_frames += 1
        except Exception as e:
            print(f"[WARNING] Error processing frame: {e}")
            dropped_frames += 1
            continue
            
        # Write frame to video
        try:
            out.write(resized_frame)
        except Exception as e:
            print(f"[ERROR] Failed to write frame: {e}")
            dropped_frames += 1
            continue
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        frame_count += 1
        
        # Status update every 3 seconds
        current_time = time.time()
        if current_time - last_status_time >= 3:
            elapsed = current_time - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            avg_frame_time = sum(frame_times[-30:]) / min(30, len(frame_times)) if frame_times else 0
            remaining = max_runtime - elapsed
            
            print(f"[INFO] Recorded {processed_frames} frames, dropped {dropped_frames} ({current_fps:.1f} FPS)")
            print(f"[INFO] Average processing time: {avg_frame_time*1000:.1f} ms")
            print(f"[INFO] Time remaining: {remaining:.1f} seconds")
            
            last_status_time = current_time
            
except KeyboardInterrupt:
    print("\n[INFO] Keyboard Interrupt detected, stopping...")
    
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    print("\n[INFO] Finalizing recording...")
    
    # Make sure to properly release resources
    if out is not None:
        out.release()
    
    if cap is not None:
        cap.release()
        
    # Verify the output
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        recording_duration = time.time() - start_time
        
        print(f"[INFO] Video saved to: {output_video_path}")
        print(f"[INFO] File size: {file_size / (1024*1024):.2f} MB")
        print(f"[INFO] Total frames processed: {processed_frames}")
        print(f"[INFO] Total frames dropped: {dropped_frames}")
        
        if recording_duration > 0:
            print(f"[INFO] Average FPS: {processed_frames / recording_duration:.2f}")
            print(f"[INFO] Recording duration: {recording_duration:.2f} seconds")
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            print(f"[INFO] Average processing time per frame: {avg_frame_time*1000:.1f} ms")
    else:
        print("[ERROR] Video file was not saved!")
    
    print("[INFO] Recording completed!")