import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLO and DeepSORT
yolo_model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=50, n_init=1)

def run_tracking(frame, frame_count, last_detection_frame, detection_interval):
    """
    Runs YOLO + DeepSORT tracking on the given frame.
    Returns: tracked_objects, updated last_detection_frame, and the annotated frame.
    """
    detections_list = []
    should_detect = (frame_count - last_detection_frame >= detection_interval)

    if should_detect:
        detections = yolo_model(frame, classes=[0])[0]  # Only detect people
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = det
            if conf > 0.4:
                detections_list.append([[x1, y1, x2, y2], conf, None])
        last_detection_frame = frame_count

    tracked_objects = tracker.update_tracks(detections_list, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return tracked_objects, last_detection_frame, frame