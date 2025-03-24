import cv2
import time
import threading
import os
import sys
from datetime import datetime
from mtcnn import MTCNN
from camera import setup_camera
from app.face_detection import process_frame_for_faces
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
    # Set up logging to file
    # log_file, log_fileobj, original_stdout, original_stderr = setup_logging()
    
    try:
        print("=" * 60)
        print("STARTING CONTINUOUS FACE MONITORING SYSTEM")
        print("=" * 60)
        print(f"Saving all images to: {OUTPUT_DIR}")
        print(f"Debug images in: {DEBUG_DIR}")
        # print(f"Log file: {log_file}")
        print(f"Embeddings stored in: {EMBEDDINGS_DIR}")
        print(f"Databases to check: {', '.join(DB_PATHS)}")
        print("=" * 60)
        
        # Precompute face embeddings for faster recognition
        print("Precomputing face embeddings...")
        # You can try different models: "VGG-Face", "Facenet", "Facenet512", "ArcFace"
        model_name = "Facenet512"
        precompute_embeddings(model_name)
        print("Embeddings computation completed")

        # Initialize face detector
        detector = MTCNN()
        
        # Keep track of threads we create
        processing_threads = []
        
        # Process exactly 1 frame (as in your original code)
        frames_to_process = 1
        frames_processed = 0

        # Add countdown to give time to get in position
        print("Starting in 10 seconds. Get ready to position yourself...")
        for i in range(10, 0, -1):
            print(f"{i}...", end=" ", flush=True)
            time.sleep(1)
        print("\nStarting capture now!")
        
        # Initialize camera
        cap = setup_camera()
        if cap is None:
            print("Exiting due to camera connection failure")
            return
        
        while frames_processed < frames_to_process:
            print(f"Capturing frame {frames_processed + 1}/{frames_to_process}")
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading from camera stream!")
                time.sleep(1)  # Wait a bit before retrying
                continue
            
            # Process the frame and get the thread if one was created
            success, thread = process_frame_for_faces(frame, detector, model_name,create_thread=True)
            
            if thread:
                processing_threads.append(thread)
                
            # Only count successfully processed frames
            if success:
                frames_processed += 1
                print(f"Frame {frames_processed}/{frames_to_process} captured successfully")
            
            # Add a delay to allow system to stabilize (2 seconds as in your original code)
            time.sleep(2)
            
        print(f"Completed capturing {frames_processed} frames")
        
        # Wait for all processing threads to complete
        print(f"Waiting for {len(processing_threads)} processing threads to complete...")
        for thread in processing_threads:
            thread.join(timeout=50)  # Wait up to 50 seconds per thread
            
        print("All processing completed")
        
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if 'cap' in locals() and cap is not None:
            cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
            
        # Close the log file and restore stdout/stderr
        # sys.stdout = original_stdout
        # sys.stderr = original_stderr
        # log_fileobj.close()
        
        # Print to the actual console where the log file is
        print(f"Face detection system stopped. ")

if __name__ == "__main__":
    main()