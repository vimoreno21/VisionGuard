import utils.preload
import os
import cv2
import time
from facenet_pytorch import MTCNN
import torch
import threading

from utils.photo_utils import current_timestamp
from camera import setup_camera
from face_detection import process_frame_for_faces
from tracking import run_tracking
from utils.video_utils import annotate_and_write

from pipeline import check_tracked_objects
from utils.logger import logger
from utils.directories import EMBEDDINGS_DIR, VIDEO_DIR
from utils.helper import send_people_inside_batch
from tracking import run_tracking
from utils.constants import track_interval

from embed import update_pkls


def main():
    # Initialization
    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS TRACKING + FACE RECOGNITION")
    logger.info(f"Video dir: {VIDEO_DIR}")
    logger.info(f"Embeddings path: {EMBEDDINGS_DIR}")
    model_name = "Facenet512"
    update_pkls(model_name)

    cap = setup_camera()
    if not cap:
        logger.error("Camera failure. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # currently using the mtcnn detector from facenet-pytorch. meaning the model is kinda stuck with facenet
    # TODO: test more detector models
    detector = MTCNN(keep_all=True, device=device)

    # State
    identified_identities = {}
    tracked_objects = []
    processing_threads = []
    frame_count = 0
    last_detection_frame = 0

    # Video writer setup
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    path = os.path.join(VIDEO_DIR, f"log_video_{current_timestamp()}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    logger.info(f"Video path: {path}")

    try:
        start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera stream ended.")
                break
            frame_count += 1

            # 1) Tracking (light) every track_interval
            t0 = time.time()
            if frame_count % track_interval == 0:
                tracked_objects, last_detection_frame, frame = run_tracking(
                    frame, frame_count, last_detection_frame, detection_interval, tracked_objects
                )
            logger.debug(f"Tracking took {time.time() - t0:.3f}s")

            # 2) ROI-based detection & UI update every detection_interval
            if frame_count % detection_interval == 0:
                detection_interval = min(max(6, len(tracked_objects) * 2), 30)
                people_visible = []
                unknown_crops = []
                check_tracked_objects(identified_identities, detector, writer, model_name, frame_count, tracked_objects, processing_threads, last_detection_frame, frame)

                # Add known identities without re-detection
                for t in tracked_objects:
                    tid = t.track_id
                    if tid in identified_identities:
                        people_visible.append({"id": tid, "name": identified_identities[tid], "face_image": None})

                # UI update
                threading.Thread(
                    target=send_people_inside_batch,
                    args=(people_visible,),
                    daemon=True
                ).start()

                # Annotate and write video frame
                annotate_and_write(frame, people_visible, tracked_objects, writer)

                # Spawn recognition threads for new faces
                for crop, tid in unknown_crops:
                    _, th, res = process_frame_for_faces(
                        frame, crop, detector, model_name, frame_count, create_thread=True
                    )
                    if th:
                        processing_threads.append((th, res, tid))
                    
            # 3) Collect recognition thread results
            alive_threads = []
            for th, res, tid in processing_threads:
                if not th.is_alive():
                    name = res.get("person_name")
                    if name:
                        identified_identities[tid] = name
                else:
                    alive_threads.append((th, res, tid))
            processing_threads = alive_threads


    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as ex:
        logger.exception(f"Error in main loop: {ex}")
    finally:
        send_people_inside_batch([])
        cap.release()
        writer.release()
        total = time.time() - start
        logger.info(f"Total runtime: {total:.2f}s, frames processed: {frame_count}")

if __name__ == "__main__":
    main()
