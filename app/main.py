import cv2
import time
import threading
import os
import sys
from datetime import datetime
from mtcnn import MTCNN
from camera import setup_camera
from face_detection import process_frame_for_faces
from tracking import run_tracking
from utils.directories import SAVE_DIR, DEBUG_DIR, DB_PATHS, LOG_FILES_DIR, EMBEDDINGS_DIR, OUTPUT_DIR
from utils.constants import FRAME_SAVE_INTERVAL, HEADLESS

# Import the precompute_embeddings function from wherever you've defined it
from embed import precompute_embeddings

def setup_logging():
    """Set up logging to file and return objects needed for cleanup"""
    # Create log directory if it doesn't exist
    os.makedirs(LOG_FILES_DIR, exist_ok=True)
    
    # Create timestamped log file
    log_file = os.path.join(LOG_FILES_DIR, f"face_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Open log file
    log_fileobj = open(log_file, 'w')
    
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Redirect stdout and stderr to log file
    sys.stdout = log_fileobj
    sys.stderr = log_fileobj
    
    return log_file, log_fileobj, original_stdout, original_stderr
def main():
    print("=" * 60)
    print("STARTING CONTINUOUS TRACKING + FACE RECOGNITION")
    print("=" * 60)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Debug dir: {DEBUG_DIR}")
    print(f"DBs: {', '.join(DB_PATHS)}")
    print(f"Embeddings path: {EMBEDDINGS_DIR}")
    print("=" * 60)

    print("Precomputing face embeddings...")
    precompute_embeddings("Facenet512")
    print("Embeddings done.\n")

    cap = setup_camera()
    if cap is None:
        print("Exiting due to camera failure.")
        return

    seen_ids = set()
    last_detection_frame = 0
    detection_interval = 5
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Camera stream ended.")
                break

            frame_count += 1

            # Get tracking results
            tracked_objects, last_detection_frame, frame = run_tracking(
                frame, frame_count, last_detection_frame, detection_interval
            )

            # Run DeepFace for new IDs
            for track in tracked_objects:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, track.to_tlbr())
                    cropped = frame[y1:y2, x1:x2]
                    process_frame_for_faces(cropped, frame, (x1, y1, x2 - x1, y2 - y1))  # non-blocking thread

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown complete.")

if __name__ == "__main__":
    main()