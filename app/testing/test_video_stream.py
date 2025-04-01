import cv2
import os
import time
import signal
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

# OPTIMIZATION: Scale down the resolution to reduce processing load
scale_factor = 0.5  # Reduce to 50% of original size
new_width = int(frame_width * scale_factor)
new_height = int(frame_height * scale_factor)

# Ensure dimensions are even
if new_width % 2 == 1:
    new_width -= 1
if new_height % 2 == 1:
    new_height -= 1

print(f"[INFO] Reduced resolution for processing: {new_width}x{new_height}")

# Use mp4 format with mp4v codec
output_video_path = os.path.join(video_save_path, f"camera_stream_{current_timestamp()}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Use a more conservative frame rate to ensure stability
target_fps = 15.0  # More achievable than 24fps on Jetson Nano

# Create the video writer with the reduced resolution
out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (new_width, new_height))

if not out.isOpened():
    print("[ERROR] Failed to initialize VideoWriter. Exiting...")
    cap.release()
    exit()

print(f"📼 Recording video to: {output_video_path}")
print(f"[INFO] Target FPS: {target_fps}")

# Variables for statistics
frame_count = 0
frame_times = []  # To track performance
start_time = time.time()
max_runtime = 60  # Record for 60 seconds

try:
    print(f"[INFO] Beginning video recording. Will run for {max_runtime} seconds.")
    print("[INFO] Press Ctrl+C once for clean shutdown.")
    
    while cap.isOpened() and not stop_program:
        # Check if we've reached the maximum runtime
        if time.time() - start_time > max_runtime:
            print(f"[INFO] Maximum runtime of {max_runtime} seconds reached")
            break
            
        # Time the frame processing
        frame_start = time.time()
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("[WARNING] Frame read error")
            # Wait a bit and try again
            time.sleep(0.1)
            continue
            
        # Resize the frame to reduce processing load
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Write frame to video
        out.write(resized_frame)
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        
        frame_count += 1
        
        # Print status occasionally
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            avg_frame_time = sum(frame_times[-30:]) / min(30, len(frame_times))
            remaining = max_runtime - elapsed
            print(f"[INFO] Recorded {frame_count} frames ({current_fps:.1f} FPS)")
            print(f"[INFO] Average processing time per frame: {avg_frame_time*1000:.1f} ms")
            print(f"[INFO] Time remaining: {remaining:.1f} seconds")
            
except KeyboardInterrupt:
    print("\n[INFO] Keyboard Interrupt detected, stopping...")
    
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {e}")
    
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
        print(f"[INFO] Total frames: {frame_count}")
        print(f"[INFO] Average FPS: {frame_count / recording_duration:.2f}")
        print(f"[INFO] Recording duration: {recording_duration:.2f} seconds")
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            print(f"[INFO] Average processing time per frame: {avg_frame_time*1000:.1f} ms")
            print(f"[INFO] Theoretical max FPS based on processing time: {1/avg_frame_time:.1f}")
    else:
        print("[ERROR] Video file was not saved!")
    
    print("[INFO] Recording completed!")