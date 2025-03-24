import cv2
import torch
import os
import numpy as np
import time
import signal
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
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

# Load YOLOv8 model for person detection
# yolo_model = YOLO("yolov8n.pt")  # Lightweight model for real-time tracking
yolo_model = YOLO("yolov8s.pt") # Better performance but slower


# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=1)  # Adjust max_age to control tracking persistence

# Use your function to set up the camera
cap = setup_camera()
if cap is None:
    print("⛔ Camera setup failed. Exiting...")
    exit()

#  Check if its a video or live feed 
is_video_file = isinstance(cap, cv2.VideoCapture) and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0

# Save output video
save_output = True
output_video_path = os.path.join(video_save_path, f"tracked_rtsp_output_{current_timestamp()}.mp4")

if save_output:
    # Use mp4v codec as it was working better for you
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    
    # Ensure dimensions are even
    if frame_width % 2 == 1:
        frame_width -= 1
    if frame_height % 2 == 1:
        frame_height -= 1
        
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"[ERROR] Failed to initialize VideoWriter. Check codec and permissions.")
        exit()
        
    print(f"📼 Saving tracked video to: {output_video_path}")
    
# Frame counter
frame_count = 0

# Define a maximum runtime in seconds
max_runtime = 60  # Adjust as needed (e.g., 60 = 1 minute of recording)
start_time = time.time()

# try:
#     print("[INFO] Beginning video processing. Will run for up to", max_runtime, "seconds.")
#     print("[INFO] Press Ctrl+C once (not repeatedly) for clean shutdown.")
    
#     while cap.isOpened() and not stop_program:
#         # Check if we've reached the maximum runtime
#         if not is_video_file and time.time() - start_time > max_runtime:
#             print(f"[INFO] Maximum runtime of {max_runtime} seconds reached")
#             break

            
#         ret, frame = cap.read()
#         if not ret:
#             print("[INFO] End of video stream reached")
#             break
            
#         frame_count += 1
        
#         if frame_count % 3 == 0:
#             detections = yolo_model(frame, classes=[0])[0] # Only detect people
#             detections_list = []
#             for det in detections.boxes.data.tolist():
#                 x1, y1, x2, y2, conf, class_id = det
#                 if conf > 0.3:  # Confidence filter
#                     detections_list.append([[x1, y1, x2, y2], conf, None])
#             tracked_objects = tracker.update_tracks(detections_list, frame=frame)
            
#         else:
#             tracked_objects = tracker.update_tracks([], frame=frame)

        
#         for track in tracked_objects:
#             print(f"Track {track.track_id} | Confirmed: {track.is_confirmed()} | Age: {track.age}")
#             if not track.is_confirmed():
#                 continue
                
#             track_id = track.track_id
#             x1, y1, x2, y2 = map(int, track.to_tlbr())
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         if save_output:
#             out.write(frame)
            
#             # Print status occasionally
#             if frame_count % 30 == 0:
#                 elapsed = time.time() - start_time
#                 fps_rate = frame_count / elapsed
#                 remaining = max_runtime - elapsed
#                 print(f"[INFO] Processed {frame_count} frames, {fps_rate:.1f} FPS, {remaining:.1f} seconds remaining")
        

last_detection_frame = 0
detection_interval = 5
tracked_objects = []

try:
    print("[INFO] Beginning YOLO + DeepSORT tracking.")
    while cap.isOpened() and not stop_program:
        if not is_video_file and time.time() - start_time > max_runtime:
            print(f"[INFO] Max runtime of {max_runtime}s reached")
            break

        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream reached")
            break

        frame_count += 1

        confirmed_tracks = [t for t in tracked_objects if t.is_confirmed()]
        should_detect = (
            frame_count - last_detection_frame >= detection_interval or
            len(confirmed_tracks) == 0
        )

        detections_list = []

        if should_detect:
            detections = yolo_model(frame, classes=[0])[0]
            for det in detections.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = det
                if conf > 0.4:
                    detections_list.append([[x1, y1, x2, y2], conf, None])
            last_detection_frame = frame_count

        tracked_objects = tracker.update_tracks(detections_list, frame=frame)

        for track in tracked_objects:
            print(f"Track {track.track_id} | Confirmed: {track.is_confirmed()} | Age: {track.age}")
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if save_output:
            out.write(frame)

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_rate = frame_count / elapsed
                remaining = max_runtime - elapsed
                print(f"[INFO] {frame_count} frames | {fps_rate:.1f} FPS | {remaining:.1f}s remaining")


except Exception as e:
    print(f"\n[ERROR] {e}")

    
finally:
    print("\n[INFO] Releasing video resources...")
    
    # Make sure to finalize the VideoWriter properly
    if save_output and out is not None:
        out.release()
    
    # Release the camera
    if cap is not None:
        cap.release()
        
    print(f"[INFO] Verifying saved video: {output_video_path}")
    
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path)
        if file_size > 1000:
            print(f"[INFO] Video saved successfully! Size: {file_size} bytes")
            print(f"[INFO] Total frames processed: {frame_count}")
            print(f"[INFO] Total runtime: {time.time() - start_time:.1f} seconds")
        else:
            print("[WARNING] Video file is very small. It may be corrupted.")
    else:
        print("[ERROR] Video file was not saved!")
    
    print("[INFO] Video processing completed!")