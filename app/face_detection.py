import os
import cv2
import numpy as np
import threading
import pickle
from deepface import DeepFace
import time

from app.embed import find_match_with_embeddings
from utils.photo_utils import current_timestamp, mark_frame_with_face
from utils.directories import SAVE_DIR, DEBUG_DIR, EMBEDDINGS_DIR, OUTPUT_DIR
from utils.constants import MATCH_THRESHOLD

def process_face(face_img, face_location, original_frame, timestamp, model_name):
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
        
        # Determine match confidence
        if identity:
            confidence = 1 - distance  # Convert distance to confidence (0-1)
            if distance < MATCH_THRESHOLD:
                # Extract person name from identity
                person_name = os.path.basename(os.path.dirname(identity)) if os.path.isfile(identity) else identity
                
                print(f"✅ Match found via {match_type}!")
                print(f"🔹 Matched with: {person_name} (Confidence: {confidence:.2f})")
                print(f"🔹 Full identity: {identity}")
                
                # Mark frame with identity and save it
                marked_frame = mark_frame_with_face(original_frame, face_location, f"{person_name} ({match_type})", confidence)
                marked_frame_path = os.path.join(matches_dir, f"match_{person_name}_{timestamp}.jpg")
                cv2.imwrite(marked_frame_path, marked_frame)
            else:
                print(f"❌ No strong match found (Low confidence: {confidence:.2f})")
                reason = "Low confidence (Distance too high)"
        else:
            print("❌ No match found in database")
            reason = "No identity found"

        # Save unknown face with reason
        marked_frame = mark_frame_with_face(original_frame, face_location, f"Unknown ({reason})")
        marked_frame_path = os.path.join(matches_dir, f"unknown_{timestamp}.jpg")
        cv2.imwrite(marked_frame_path, marked_frame)

    except Exception as e:
        print(f"Error processing face: {e}")

def process_frame_for_faces(frame, detector, model_name, create_thread=False):
    """Process a single frame to detect and recognize faces"""
    # Save this frame (with timestamp) for debugging
    timestamp = current_timestamp()
    
    # Create raw images directory if it doesn't exist
    raw_images_dir = os.path.join(DEBUG_DIR, "raw_images")
    os.makedirs(raw_images_dir, exist_ok=True)
    
    # Save raw frame
    raw_path = os.path.join(raw_images_dir, f"raw_frame_{timestamp}.jpg")
    cv2.imwrite(raw_path, frame)
    
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
        print(f"✅ Face detected and saved to {face_path}")
        
        if create_thread:
            # Process face in a separate thread
            thread = threading.Thread(target=process_face, args=(face_img, face_location, frame, timestamp, model_name))
            thread.start()  # Note: not setting daemon=True
            return True, thread
        else:
            # Process face directly
            process_face(face_img, face_location, frame, timestamp, model_name)
            return True, None
    else:
        print("No face detected in this frame")
        return False, None

def detect_and_crop_face(frame, detector):
    """
    Detect and crop the largest face in the frame
    Returns: cropped face image or None if no face detected
    """
    # Convert to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    try:
        results = detector.detect_faces(rgb_frame)
        print(f"Face detection results: {len(results)} faces found")
    except Exception as e:
        print(f"Error during face detection: {e}")
        return None, None
    
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
    
    # Crop face
    cropped_face = frame[y_new:y_new + h_new, x_new:x_new + w_new]
    face_location = (x_new, y_new, w_new, h_new)
    
    return cropped_face, face_location