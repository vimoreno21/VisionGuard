import os
import cv2
import numpy as np
import threading
import pickle
import time

from embed import find_match_with_embeddings
from utils.photo_utils import current_timestamp, mark_frame_with_face
from utils.directories import SAVE_DIR, DEBUG_DIR, EMBEDDINGS_DIR, OUTPUT_DIR
from utils.constants import MATCH_THRESHOLD
from utils.logger import logger


def process_face(face_img, face_location, original_frame, timestamp, img_frame, model_name, result_dict=None):
    """Process a detected face for recognition (run in separate thread)"""

    try:
        # Create output directories if they don't exist
        detected_faces_dir = os.path.join(OUTPUT_DIR, "detected_faces")
        matches_dir = os.path.join(OUTPUT_DIR, "matches")
        os.makedirs(detected_faces_dir, exist_ok=True)
        os.makedirs(matches_dir, exist_ok=True)
        
        # Find match using precomputed embeddings
        identity, distance, match_type = find_match_with_embeddings(face_img, model_name)

        # Create reson variable 
        # reason = "Unassigned reason var"
        
        # Determine match confidence
        if identity:
            confidence = 1 - distance  # Convert distance to confidence (0-1)
            if distance < MATCH_THRESHOLD:
                # Extract person name from identity
                person_name = os.path.basename(os.path.dirname(identity))
                
                logger.info(f"✅ Frame {img_frame}: match found via {match_type}! "
                            f"Matched with: {person_name} (Confidence: {confidence:.2f}) "
                            f"Full identity: {identity}")
                
                # Annotate the original frame with the person's name
                annotated_frame = original_frame.copy()
                x, y, w, h = face_location
                # Draw a rectangle around the face
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Put text with the person's name above the face
                cv2.putText(annotated_frame, person_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Save the annotated image
                filename = os.path.join(matches_dir, f"{timestamp}_frame{img_frame}_{person_name}.jpg")
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Saved annotated image to {filename}")

                if result_dict is not None:
                    result_dict['identity'] = identity
                return person_name
            else:
                logger.info(f"No strong match found (Low confidence: {confidence:.2f})")
                # reason = "Low confidence (Distance too high)"
        else:
            logger.info("No match found in database")
            # reason = "No identity found"

        if result_dict is not None:
            result_dict['identity'] = None
        
        return None

    except Exception as e:
        logger.exception(f"Error processing face: {e}")

def process_frame_for_faces(full_frame, cropped_frame, detector, model_name, img_frame, create_thread=False):
    """Process a single frame to detect and recognize faces"""
    # Save this frame (with timestamp) for debugging
    timestamp = current_timestamp()
    
    # Detect and crop face
    face_img, face_location = detect_and_crop_face(cropped_frame, detector)
    
    thread = None
    if face_img is not None:
        # Make sure output directory exists
        detected_faces_dir = os.path.join(OUTPUT_DIR, "detected_faces")
        os.makedirs(detected_faces_dir, exist_ok=True)
        
        if create_thread:
            result_dict = {}  # shared dict to hold the result
            # Process face in a separate thread
            thread = threading.Thread(target=process_face, args=(face_img, face_location, full_frame, timestamp, img_frame, model_name, result_dict))
            thread.start()  # Note: not setting daemon=True
            return True, thread, result_dict
        
        else:
            # Process face directly
            identity = process_face(face_img, face_location, full_frame, timestamp, model_name)
            return True, None, identity
    else:
        logger.debug("No face detected in this frame")
        return False, None, None

def detect_and_crop_face(frame, detector):
    if frame is None or frame.size == 0:
        logger.warning("Empty frame received in detect_and_crop_face")
        return None, None

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, probs = detector.detect(rgb_frame)
        except RuntimeError as e:
            if "torch.cat()" in str(e):
                logger.debug("No faces detected (empty tensor list)")
                return None, None
            else:
                raise e

        if boxes is None or len(boxes) == 0:
            return None, None

        # Process detections: find largest face, add margin, crop, etc.
        boxes = boxes.astype(int)
        areas = [(box[2]-box[0]) * (box[3]-box[1]) for box in boxes]
        largest_idx = areas.index(max(areas))
        x1, y1, x2, y2 = boxes[largest_idx]

        margin = 20
        height, width, _ = frame.shape
        x_new = max(0, x1 - margin)
        y_new = max(0, y1 - margin)
        x2_new = min(width - 1, x2 + margin)
        y2_new = min(height - 1, y2 + margin)

        if x2_new <= x_new or y2_new <= y_new:
            logger.warning("Invalid face dimensions after applying margin")
            return None, None

        cropped_face = frame[y_new:y2_new, x_new:x2_new].copy()
        face_location = (x_new, y_new, x2_new - x_new, y2_new - y_new)
        return cropped_face, face_location

    except Exception as e:
        logger.exception(f"Error during face detection: {e}")
        return None, None
