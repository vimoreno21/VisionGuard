import os
import cv2
import time
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
import warnings
import threading
import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
RTSP_URL = "rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/701"
DB_PATHS = ["victoria", "other"]
SAVE_DIR = "./database/captured_images"
MATCH_THRESHOLD = 0.5  # Lower values = stricter matching
HEADLESS = True  # Set to True if running without display (e.g. in Docker)
SAVE_ALL_FRAMES = True  # Save frames periodically even without faces
FRAME_SAVE_INTERVAL = 5  # Save a frame every 5 seconds

# Ensure directories exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for db in DB_PATHS:
    db_dir = f"./database/{db}"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created missing database directory: {db_dir}")

# Create debug directory
DEBUG_DIR = os.path.join(SAVE_DIR, "debug2")
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

def current_timestamp():
    """Return current timestamp as string"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def detect_and_crop_face(frame):
    """
    Detect and crop the largest face in the frame
    Returns: cropped face image or None if no face detected
    """
    # Save the raw frame for debugging
    debug_path = os.path.join(DEBUG_DIR, f"raw_frame_{current_timestamp()}.jpg")
    cv2.imwrite(debug_path, frame)
    print(f"Saved raw frame for debugging: {debug_path}")
    
    # Initialize face detector
    detector = MTCNN()
    
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
    
    # Save the detected face for debugging
    face_debug_path = os.path.join(DEBUG_DIR, f"face_detected_{current_timestamp()}.jpg")
    cv2.imwrite(face_debug_path, cropped_face)
    print(f"Saved detected face for debugging: {face_debug_path}")
    
    return cropped_face, face_location

def process_frame_for_faces(frame):
    """Process a single frame to detect and recognize faces"""
    # Save this frame (with timestamp) for debugging
    timestamp = current_timestamp()
    frame_path = os.path.join(DEBUG_DIR, f"frame_{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)
    
    # Detect and crop face
    face_img, face_location = detect_and_crop_face(frame)
    
    if face_img is not None:
        # Save the cropped face
        face_path = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_img)
        print(f"✅ Face detected and saved to {face_path}")
        
        # Process face in a separate thread to avoid blocking the main loop
        thread = threading.Thread(target=process_face, args=(face_img, face_location, frame, timestamp))
        thread.daemon = True
        thread.start()
        
        return True
    else:
        print("No face detected in this frame")
        return False

def process_face(face_img, face_location, original_frame, timestamp):
    """Process a detected face for recognition (run in separate thread)"""
    try:
        # Find match in database
        db_name, identity, distance = find_match_in_db(face_img)
        
        # Check if match is good enough
        if db_name and distance < MATCH_THRESHOLD:
            confidence = 1 - distance  # Convert distance to confidence (0-1)
            person_name = os.path.basename(os.path.dirname(identity)) if os.path.sep in identity else "Unknown"
            
            print(f"✅ Match found in {db_name} database!")
            print(f"🔹 Matched with: {person_name} (Confidence: {confidence:.2f})")
            print(f"🔹 Full identity path: {identity}")
            
            # Mark frame with identity and save it
            marked_frame = mark_frame_with_face(original_frame, face_location, person_name, confidence)
            marked_frame_path = os.path.join(SAVE_DIR, f"match_{person_name}_{timestamp}.jpg")
            cv2.imwrite(marked_frame_path, marked_frame)
        else:
            print("❌ No strong match found in database")
            
            # Mark frame as unknown and save it
            marked_frame = mark_frame_with_face(original_frame, face_location, "Unknown")
            marked_frame_path = os.path.join(SAVE_DIR, f"unknown_{timestamp}.jpg")
            cv2.imwrite(marked_frame_path, marked_frame)
    except Exception as e:
        print(f"Error processing face: {e}")

def find_match_in_db(face_img):
    """
    Find matching faces in the databases
    Returns: (db_name, identity, confidence) of the best match, or (None, None, 0) if no match
    """
    # Save the face temporarily for DeepFace
    temp_path = os.path.join(SAVE_DIR, f"temp_face_{current_timestamp()}.jpg")
    cv2.imwrite(temp_path, face_img)
    print(f"Saved temporary face for DeepFace: {temp_path}")
    
    best_match = (None, None, 1.0)  # (db_name, identity, distance) - lower distance is better
    
    for db_name in DB_PATHS:
        db_path = f"./database/{db_name}"
        
        if not os.path.exists(db_path):
            print(f"Warning: Database path {db_path} does not exist. Skipping.")
            continue
            
        try:
            print(f"Searching in {db_name} database...")
            results = DeepFace.find(
                img_path=temp_path,
                db_path=db_path,
                model_name="VGG-Face",
                distance_metric="cosine",
                enforce_detection=False  # Important for robustness
            )
            
            if results and not results[0].empty:
                df = results[0]
                match = df.iloc[0]
                distance = match["distance"]
                print(f"Found match in {db_name} with distance {distance}")
                
                # Update best match if this is better
                if distance < best_match[2]:
                    identity = match["identity"]
                    best_match = (db_name, identity, distance)
            else:
                print(f"No matches found in {db_name} database")
                    
        except Exception as e:
            print(f"Error searching in {db_name} database: {e}")
    
    # Cleanup is optional - we can keep the temp files for debugging
    return best_match

def mark_frame_with_face(frame, location, name=None, confidence=None):
    """
    Draw a box around the detected face with optional name and confidence
    """
    # Create a copy to avoid modifying the original
    marked_frame = frame.copy()
    
    x, y, w, h = location
    
    # Draw rectangle
    cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text if name is provided
    if name:
        label = f"{name}"
        if confidence:
            label += f" ({confidence:.2f})"
        
        # Background for text
        cv2.rectangle(marked_frame, (x, y - 30), (x + len(label) * 12, y), (0, 255, 0), -1)
        cv2.putText(marked_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return marked_frame

def setup_camera():
    """Set up the camera with retry logic"""
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Try to open the camera
            print(f"Attempting to connect to camera: {RTSP_URL}")
            cap = cv2.VideoCapture(RTSP_URL)
            
            # Check if opened successfully
            if cap.isOpened():
                # Read a test frame
                ret, frame = cap.read()
                if ret:
                    # Save the first frame for debugging
                    first_frame_path = os.path.join(DEBUG_DIR, f"first_frame_{current_timestamp()}.jpg")
                    cv2.imwrite(first_frame_path, frame)
                    print(f"✅ Camera connected successfully! First frame saved to {first_frame_path}")
                    
                    # Print frame dimensions
                    height, width, channels = frame.shape
                    print(f"Frame dimensions: {width}x{height}, {channels} channels")
                    
                    return cap
                else:
                    print(f"❌ Camera opened but couldn't read frame (attempt {attempt+1}/{max_attempts})")
            else:
                print(f"❌ Failed to open camera (attempt {attempt+1}/{max_attempts})")
            
            # Close the connection before retry
            cap.release()
            
        except Exception as e:
            print(f"❌ Error connecting to camera: {e} (attempt {attempt+1}/{max_attempts})")
        
        # Wait before retry
        time.sleep(2)
        attempt += 1
    
    print("⛔ All camera connection attempts failed")
    return None

def main():
    print("=" * 60)
    print("STARTING CONTINUOUS FACE MONITORING SYSTEM")
    print("=" * 60)
    print(f"Camera URL: {RTSP_URL}")
    print(f"Saving all images to: {SAVE_DIR}")
    print(f"Debug images in: {DEBUG_DIR}")
    print(f"Databases to check: {', '.join(DB_PATHS)}")
    print("=" * 60)
    print("Press Ctrl+C to quit")
    
    # Initialize camera
    cap = setup_camera()
    if cap is None:
        print("Exiting due to camera connection failure")
        return
    
    # Variables for timing
    last_frame_save_time = 0
    frame_count = 0
    
    # We'll use the periodic frame saving as our main loop timing
    # rather than doing separate face detection checks
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            # Increment frame counter and log periodically
            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                print(f"✅ Still running - processed {frame_count} frames")
            
            if not ret:
                print("Error reading from camera stream! Attempting to reconnect...")
                cap.release()
                cap = setup_camera()
                if cap is None:
                    print("Failed to reconnect to camera. Exiting.")
                    break
                continue
            
            current_time = time.time()
            
            # Periodically save frames even without faces (if enabled)
            if SAVE_ALL_FRAMES and (current_time - last_frame_save_time) >= FRAME_SAVE_INTERVAL:
                last_frame_save_time = current_time
                timestamp = current_timestamp()
                periodic_frame_path = os.path.join(DEBUG_DIR, f"periodic_{timestamp}.jpg")
                cv2.imwrite(periodic_frame_path, frame)
                print(f"SAVING NEW FRAME AT {timestamp}")
                print(f"Saved new frame to {periodic_frame_path}")
                
                # Also process this frame for faces
                process_frame_for_faces(frame)
            
            # We've removed the separate face detection interval
            # and integrated it with the frame saving logic above
            
            # Brief sleep to reduce CPU usage
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nDetection system stopped by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("Face detection system stopped")

if __name__ == "__main__":
    main()