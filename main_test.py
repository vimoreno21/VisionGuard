import cv2
import time
import os
from mtcnn import MTCNN
from deepface import DeepFace
from utils.photo_utils import current_timestamp
from utils.directories import SAVE_DIR, DEBUG_DIR, DB_PATHS
from camera import setup_camera
from utils.constants import HEADLESS, MATCH_THRESHOLD

def main():
    print("=" * 60)
    print("STARTING FACE RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Saving all images to: {SAVE_DIR}")
    print(f"Databases to check: {', '.join(DB_PATHS)}")
    print("=" * 60)
    
    # Initialize camera
    cap = setup_camera()
    if cap is None:
        print("Exiting due to camera connection failure")
        return
    
    # Initialize face detector
    print("Initializing MTCNN detector - this may take a moment...")
    detector = MTCNN()
    print("MTCNN detector initialized")
    
    # Process exactly 2 frames
    frames_to_process = 2
    frames_processed = 0
    
    try:
        while frames_processed < frames_to_process:
            print(f"Capturing frame {frames_processed + 1}/{frames_to_process}")
            
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading from camera stream!")
                time.sleep(1)
                continue
            
            # Save the original frame
            timestamp = current_timestamp()
            frame_path = os.path.join(SAVE_DIR, f"frame_{timestamp}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Try to detect faces
            face_found = False
            try:
                print("Detecting faces...")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(rgb_frame)
                print(f"Face detection results: {len(results)} faces found")
                
                # If faces found, save and process the first one
                if results:
                    face = results[0]
                    x, y, w, h = face['box']
                    margin = 10
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    w = min(frame.shape[1] - x, w + margin * 2)
                    h = min(frame.shape[0] - y, h + margin * 2)
                    
                    face_img = frame[y:y+h, x:x+w]
                    face_path = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
                    cv2.imwrite(face_path, face_img)
                    print(f"Face saved to {face_path}")
                    face_found = True
                    
                    # Now try to recognize the face using DeepFace
                    print("Attempting face recognition...")
                    best_match = (None, None, 1.0)
                    
                    for db_name in DB_PATHS:
                        db_path = f"./database/{db_name}"
                        
                        if not os.path.exists(db_path):
                            print(f"Warning: Database path {db_path} does not exist. Skipping.")
                            continue
                        
                        try:
                            print(f"Searching in {db_name} database...")
                            results = DeepFace.find(
                                img_path=face_path,  # Use the saved image
                                db_path=db_path,
                                model_name="VGG-Face",
                                distance_metric="cosine",
                                enforce_detection=False
                            )
                            
                            if results and not results[0].empty:
                                df = results[0]
                                match = df.iloc[0]
                                distance = match["distance"]
                                print(f"Found match in {db_name} with distance {distance}")
                                
                                if distance < best_match[2]:
                                    identity = match["identity"]
                                    best_match = (db_name, identity, distance)
                            else:
                                print(f"No matches found in {db_name} database")
                                
                        except Exception as e:
                            print(f"Error searching in {db_name} database: {e}")
                    
                    # Check if good match found
                    db_name, identity, distance = best_match
                    if db_name and distance < MATCH_THRESHOLD:
                        confidence = 1 - distance
                        person_name = os.path.basename(os.path.dirname(identity)) if os.path.sep in identity else "Unknown"
                        print(f"✅ Match found! Person: {person_name}, Confidence: {confidence:.2f}")
                    else:
                        print("❌ No strong match found in database")
                
            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()
            
            frames_processed += 1
            print(f"Frame {frames_processed}/{frames_to_process} fully processed")
            
            # Add a longer delay between frames to let system stabilize
            time.sleep(2)
            
        print(f"Completed processing {frames_processed} frames")
        
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()
        print("Recognition system stopped")

if __name__ == "__main__":
    main()