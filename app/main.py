import utils.preload 

import cv2
import time
from datetime import datetime
from facenet_pytorch import MTCNN
import torch
from camera import setup_camera
from face_detection import process_frame_for_faces
from tracking import run_tracking
from utils.logger import logger
from utils.directories import SAVE_DIR, DEBUG_DIR, DB_PATHS, LOG_FILES_DIR, EMBEDDINGS_DIR, OUTPUT_DIR
from utils.constants import FRAME_SAVE_INTERVAL, HEADLESS

# Import the precompute_embeddings function from wherever you've defined it
from embed import precompute_embeddings
    

def main():    
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS TRACKING + FACE RECOGNITION")
    logger.info("=" * 60)
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Debug dir: {DEBUG_DIR}")
    logger.info(f"DBs: {', '.join(DB_PATHS)}")
    logger.info(f"Embeddings path: {EMBEDDINGS_DIR}")
    logger.info("=" * 60)

    # Precompute face embeddings for faster recognition
    logger.info("Precomputing face embeddings...")
    # You can try different models: "VGG-Face", "Facenet", "Facenet512", "ArcFace"
    model_name = "Facenet512"
    precompute_embeddings(model_name)
    logger.info("Embeddings computation completed")

    cap = setup_camera()
    if cap is None:
        logger.error("Exiting due to camera failure.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=True, device=device)

    seen_ids = set()

    identified_identities = set()

    last_detection_frame = 0
    detection_interval = 6
    frame_count = 0

    # Keep track of threads we create
    processing_threads = []

    # At the beginning of your main function, initialize tracked_objects
    tracked_objects = []
    
    try:
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera stream ended.")
                break

            frame_count += 1
            
            if frame_count % 3 != 0:
                continue  # skip recognition this frame

            # Get tracking results
            tracked_objects, last_detection_frame, frame = run_tracking(
                frame, frame_count, last_detection_frame, detection_interval, tracked_objects
            )

            logger.debug(f"Frame {frame_count}: {len(tracked_objects)} tracked objects")

            # Run DeepFace for new IDs
            for track in tracked_objects:

                track_id = track.track_id

                # Skip if the track is not confirmed
                if not track.is_confirmed():
                    continue
    
                if track_id in seen_ids:
                    continue  # already identified, skip everything

                logger.info(f"Processing track ID {track_id}")

                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cropped = frame[y1:y2, x1:x2]

                # Process the frame and get the thread if one was created
                success, thread, result_dict = process_frame_for_faces(cropped, detector, model_name, frame_count, create_thread=True)

                if thread:
                    processing_threads.append((thread, result_dict, track_id))

            for thread, result_dict, track_id in processing_threads:
                thread.join(timeout=5)
                person_name = result_dict.get('identity')

                if person_name:
                    if person_name not in identified_identities:
                        identified_identities.add(person_name)
                    seen_ids.add(track_id)
                    logger.info(f"Frame {frame_count} captured and ID {track_id} identified as {person_name}")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    logger.info(f"{person_name} identified in {elapsed_time:.2f} seconds")
                else:
                    logger.info(f"Frame {frame_count} captured but ID {track_id} could not be identified")

            processing_threads.clear()   
                
    except KeyboardInterrupt:
        logger.info("\n[INFO] Interrupted by user.")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        logger.info("[INFO] System shutdown complete.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total runtime: {elapsed_time:.2f} seconds")

        # If you want video length in frames and approx duration:
        logger.info(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()