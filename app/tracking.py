import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.logger import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize YOLO and DeepSORT
yolo_model = YOLO("yolov8s.pt")
yolo_model.model.to(device)
tracker = DeepSort(
    max_age=60,         # Maximum frames to keep track alive without detection
    n_init=2,           # Increase to require more detections for confirmation
    nn_budget=100,      # Number of samples to retain in appearance descriptors
    max_cosine_distance=0.3,  # Lower threshold for better matching
)

def run_tracking(frame, frame_count, last_detection_frame, detection_interval, previous_tracks=None):
    """
    Runs YOLO + DeepSORT tracking on the given frame.
    Returns: tracked_objects, updated last_detection_frame, and the annotated frame.
    """

    # Validate input frame
    if frame is None or frame.size == 0:
        logger.warning("Empty frame received in run_tracking")
        return previous_tracks or [], last_detection_frame, frame
    
    detections_list = []

    # Check if we have any confirmed tracks from previous call
    confirmed_tracks = []
    if previous_tracks is not None:
        confirmed_tracks = [t for t in previous_tracks if t.is_confirmed()]


    # Force detection if no confirmed tracks or detection interval reached
    should_detect = (
        frame_count - last_detection_frame >= detection_interval or
        len(confirmed_tracks) == 0
    )

    if should_detect:
        try:
            detections = yolo_model(frame, classes=[0])[0]  # Only detect people
            for det in detections.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = det
                if conf > 0.4:
                    detections_list.append([[x1, y1, x2, y2], conf, None])
            last_detection_frame = frame_count
        except Exception as e:
            logger.exception(f"Error during detection: {e}")
    
    try:
        tracked_objects = tracker.update_tracks(detections_list, frame=frame)
    except Exception as e:
        logger.exception(f"Error during tracking: {e}")
        tracked_objects = previous_tracks or []

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        try:
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            # add some padding for the face
            y1 = max(0, y1 - 50)
            y2 = min(frame.shape[0], y2 - 300)
            x1 = max(0, x1 - 50)
            x2 = min(frame.shape[1], x2 + 50)

            # Validate the bounding box coordinates are within frame dimensions
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Only draw if the box has valid dimensions
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            logger.exception(f"Error drawing confirmed track {track.track_id}: {e}")

    return tracked_objects, last_detection_frame, frame