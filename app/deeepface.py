import os
import cv2
import datetime
import threading
from deepface import DeepFace

DB_PATHS = ["victoria", "other"]
SAVE_DIR = "./database/captured_images"
DEBUG_DIR = os.path.join(SAVE_DIR, "debug")
MATCH_THRESHOLD = 0.5

os.makedirs(DEBUG_DIR, exist_ok=True)

def current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def process_frame_for_faces(face_img, frame, face_location):
    timestamp = current_timestamp()
    face_path = os.path.join(SAVE_DIR, f"face_{timestamp}.jpg")
    cv2.imwrite(face_path, face_img)
    print(f"[DeepFace] Saved face for processing: {face_path}")

    thread = threading.Thread(target=process_face, args=(face_img, face_location, frame, timestamp))
    thread.daemon = True
    thread.start()

def process_face(face_img, face_location, original_frame, timestamp):
    db_name, identity, distance = find_match_in_db(face_img)

    if db_name and distance < MATCH_THRESHOLD:
        confidence = 1 - distance
        person_name = os.path.basename(os.path.dirname(identity))
        marked_frame = mark_frame_with_face(original_frame, face_location, person_name, confidence)
        cv2.imwrite(os.path.join(SAVE_DIR, f"match_{person_name}_{timestamp}.jpg"), marked_frame)
        print(f"[DeepFace] ✅ Match: {person_name} ({confidence:.2f})")
    else:
        marked_frame = mark_frame_with_face(original_frame, face_location, "Unknown")
        cv2.imwrite(os.path.join(SAVE_DIR, f"unknown_{timestamp}.jpg"), marked_frame)
        print("[DeepFace] ❌ No strong match")

def find_match_in_db(face_img):
    temp_path = os.path.join(SAVE_DIR, f"temp_face_{current_timestamp()}.jpg")
    cv2.imwrite(temp_path, face_img)

    best_match = (None, None, 1.0)
    for db_name in DB_PATHS:
        db_path = f"./database/{db_name}"
        if not os.path.exists(db_path):
            continue
        try:
            results = DeepFace.find(
                img_path=temp_path,
                db_path=db_path,
                model_name="VGG-Face",
                distance_metric="cosine",
                enforce_detection=False,
                verbose=0
            )
            if results and not results[0].empty:
                df = results[0]
                match = df.iloc[0]
                distance = match["distance"]
                if distance < best_match[2]:
                    best_match = (db_name, match["identity"], distance)
        except Exception as e:
            print(f"[DeepFace] Error in {db_name}: {e}")
    return best_match

def mark_frame_with_face(frame, location, name=None, confidence=None):
    marked_frame = frame.copy()
    x, y, w, h = location
    cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if name:
        label = f"{name}"
        if confidence:
            label += f" ({confidence:.2f})"
        cv2.rectangle(marked_frame, (x, y - 30), (x + len(label) * 12, y), (0, 255, 0), -1)
        cv2.putText(marked_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return marked_frame