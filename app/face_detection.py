import os
import cv2
import numpy as np
import threading
import pickle
from deepface import DeepFace
import time

from embed import find_match_with_embeddings
from utils.photo_utils import current_timestamp, mark_frame_with_face
from utils.directories import SAVE_DIR, DEBUG_DIR, EMBEDDINGS_DIR, OUTPUT_DIR
from utils.constants import MATCH_THRESHOLD
from utils.logger import logger


def process_face(face_img, face_location, original_frame, timestamp, model_name, result_dict=None):
    """Process a detected face for recognition (run in separate thread)"""

    try:
        # Create output directories if they don't exist
        detected_faces_dir = os.path.join(OUTPUT_DIR, "detected_faces")
        matches_dir = os.path.join(OUTPUT_DIR, "matches")
        os.makedirs(detected_faces_dir, exist_ok=True)
        os.makedirs(matches_dir, exist_ok=True)
        
        # Save the face temporarily
        temp_path = os.path.join(detected_faces_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(temp_path, face_img)
        
        # Find match using precomputed embeddings
        identity, distance, match_type = find_match_with_embeddings(temp_path, model_name)

        # Create reson variable 
        reason = "Unassigned reason var"
        
        # Determine match confidence
        if identity:
            confidence = 1 - distance  # Convert distance to confidence (0-1)
            if distance < MATCH_THRESHOLD:
                # Extract person name from identity
                person_name = os.path.basename(os.path.dirname(identity))
                
                logger.info(f"✅ Match found via {match_type}!")
                logger.info(f"🔹 Matched with: {person_name} (Confidence: {confidence:.2f})")
                logger.info(f"🔹 Full identity: {identity}")
                
                # Mark frame with identity and save it
                marked_frame = mark_frame_with_face(original_frame, face_location, f"{person_name} ({match_type})", confidence)
                marked_frame_path = os.path.join(matches_dir, f"match_{person_name}_{timestamp}.jpg")
                cv2.imwrite(marked_frame_path, marked_frame)

                if result_dict is not None:
                    result_dict['identity'] = identity
                return person_name
            else:
                logger.info(f"❌ No strong match found (Low confidence: {confidence:.2f})")
                reason = "Low confidence (Distance too high)"
        else:
            logger.info("❌ No match found in database")
            reason = "No identity found"

        # Save unknown face with reason
        marked_frame = mark_frame_with_face(original_frame, face_location, f"Unknown ({reason})")
        marked_frame_path = os.path.join(matches_dir, f"unknown_{timestamp}.jpg")
        cv2.imwrite(marked_frame_path, marked_frame)

        if result_dict is not None:
            result_dict['identity'] = None
        
        return None

    except Exception as e:
        logger.exception(f"Error processing face: {e}")

def process_frame_for_faces(frame, detector, model_name, create_thread=False):
    """Process a single frame to detect and recognize faces"""
    # Save this frame (with timestamp) for debugging
    timestamp = current_timestamp()
    
    # Create raw images directory if it doesn't exist
    raw_images_dir = os.path.join(DEBUG_DIR, "raw_images")
    os.makedirs(raw_images_dir, exist_ok=True)
    
    # # Save raw frame
    # raw_path = os.path.join(raw_images_dir, f"raw_frame_{timestamp}.jpg")
    # cv2.imwrite(raw_path, frame)
    
    # Detect and crop face
    face_img, face_location = detect_and_crop_face(frame, detector)
    
    thread = None
    if face_img is not None:
        # Make sure output directory exists
        detected_faces_dir = os.path.join(OUTPUT_DIR, "detected_faces")
        os.makedirs(detected_faces_dir, exist_ok=True)
        
        # Save the cropped face
        face_path = os.path.join(detected_faces_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_img)
        logger.info(f"✅ Face detected and saved to {face_path}")
        
        if create_thread:
            # Process face in a separate thread
            thread = threading.Thread(target=process_face, args=(face_img, face_location, frame, timestamp, model_name))
            thread.start()  # Note: not setting daemon=True
            return True, thread, None
        
        else:
            # Process face directly
            identity = process_face(face_img, face_location, frame, timestamp, model_name)
            return True, None, identity
    else:
        logger.debug("No face detected in this frame")
        return False, None, None

def detect_and_crop_face(frame, detector):
    """
    Detect and crop the largest face in the frame
    Returns: cropped face image or None if no face detected
    """
    # Validate input frame
    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        logger.warning("Warning: Empty frame received in detect_and_crop_face")
        return None, None
    
    try:
        # Convert to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(rgb_frame)
        logger.debug(f"Face detection results: {len(results)} faces found")
        
        if not results:
            return None, None
        
        # Find the largest face based on bounding box area
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add margin to the face crop
        margin = 20
        height, width, _ = frame.shape
        
        # Adjust coordinates with margin, ensuring they stay within the image boundaries
        x_new = max(0, x - margin)
        y_new = max(0, y - margin)
        w_new = min(width - x_new, w + margin * 2)
        h_new = min(height - y_new, h + margin * 2)
        
        # Validate final dimensions
        if w_new <= 0 or h_new <= 0:
            logger.warning("Invalid face dimensions after applying margin")
            return None, None
            
        # Crop face and make a copy to avoid reference issues
        cropped_face = frame[y_new:y_new + h_new, x_new:x_new + w_new].copy()
        
        # Final validation of cropped face
        if cropped_face is None or cropped_face.size == 0:
            logger.warning("Empty face image after cropping")
            return None, None
            
        face_location = (x_new, y_new, w_new, h_new)
        
        return cropped_face, face_location
        
    except Exception as e:
        logger.exception(f"Error during face detection: {e}")
        return None, None