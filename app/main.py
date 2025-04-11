import utils.preload 
import os
import cv2
import time
from facenet_pytorch import MTCNN
import torch
import requests
from utils.photo_utils import current_timestamp
from camera import setup_camera
from face_detection import process_frame_for_faces
from tracking import run_tracking
from utils.logger import logger
from utils.directories import DEBUG_DIR, EMBEDDINGS_DIR, OUTPUT_DIR, DATABASE_ROOT
from utils.constants import API_URL

# Import the precompute_embeddings function from wherever you've defined it
from embed import update_pkls

def send_people_inside_batch(current_people):
    try:
        logger.info(f"Sending to backend: {current_people}")
        response = requests.post(
            f"{API_URL}/api/update_people_batch",
            json={"people": current_people}
        )
        if response.status_code == 200:
            logger.info("Sent people_inside batch update.")
        else:
            logger.warning(f"Server responded with {response.status_code}: {response.text}")
            raise Exception(f"Failed to update server: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending people_inside batch: {e}")
        raise e

def main():    
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS TRACKING + FACE RECOGNITION")
    logger.info("=" * 60)
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Debug dir: {DEBUG_DIR}")
    logger.info(f"Embeddings path: {EMBEDDINGS_DIR}")
    logger.info(f"Database root: {DATABASE_ROOT}")
    logger.info("=" * 60)

    # Precompute face embeddings for faster recognition
    logger.info("Precomputing face embeddings...")
    # You can try different models: "VGG-Face", "Facenet", "Facenet512", "ArcFace"
    model_name = "Facenet512"
    update_pkls(model_name)
    logger.info("Embeddings computation completed")

    cap = setup_camera()
    if cap is None:
        logger.error("Exiting due to camera failure.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(keep_all=True, device=device)

    identified_identities = {}
    

    last_detection_frame = 0
    detection_interval = 4
    frame_count = 0

    # Keep track of threads we create
    processing_threads = []

    # At the beginning of your main function, initialize tracked_objects
    tracked_objects = []

    # Initialize VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # default if unavailable
    video_log_path = os.path.join(OUTPUT_DIR, f"log_video_{current_timestamp()}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_log_path, fourcc, fps, (frame_width, frame_height))
    logger.info(f"Video log path: {video_log_path}")
    
    try:
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera stream ended.")
                break

            frame_count += 1
            
            if frame_count % detection_interval != 0:
                continue  # skip recognition this frame
            
            currently_visible = []

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

                # saving debug frames
                annotated_frame = frame.copy()
                # write the names of the identified people on the frame
                name = identified_identities.get(track_id, "Unknown")
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                # add some padding for the face
                y1 = max(0, y1 - 50)
                y2 = min(frame.shape[0], y2 - 300)
                x1 = max(0, x1 - 50)
                x2 = min(frame.shape[1], x2 + 50)

                cv2.putText(annotated_frame, f"Name: {name}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Save the annotated frame to video
                video_writer.write(annotated_frame)

                currently_visible.append({
                    "id": track_id,
                    "name": name,
                    "face_image": None
                })

                if track_id in identified_identities:
                    continue  # already identified, skip everything
                
                logger.info(f"Processing track ID {track_id}")

                cropped = frame[y1:y2, x1:x2]

                # Process the frame and get the thread if one was created
                success, thread, result_dict = process_frame_for_faces(frame, cropped, detector, model_name, frame_count, create_thread=True)

                if thread:
                    processing_threads.append((thread, result_dict, track_id))

            
            for thread, result_dict, track_id in processing_threads:
                thread.join(timeout=5)
                person_name = result_dict.get('identity')

                if person_name:
                    identified_identities[track_id] = person_name
                    logger.info(f"Frame {frame_count} captured and ID {track_id} identified as {person_name}")
                else:
                    person_name = "Unknown"
                    logger.info(f"Frame {frame_count} captured but ID {track_id} could not be identified")

            send_people_inside_batch(currently_visible)

            processing_threads.clear()   
                
    except KeyboardInterrupt:
        logger.info("\n[INFO] Interrupted by user.")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        logger.info("[INFO] System shutdown complete.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total runtime: {elapsed_time:.2f} seconds")

        # If you want video length in frames and approx duration:
        logger.info(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()