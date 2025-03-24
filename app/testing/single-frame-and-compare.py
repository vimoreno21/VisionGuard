import os
import cv2
from deepface import DeepFace
from mtcnn import MTCNN

SAVE_DIR = "./captured_images"
SAVE_PATH = os.path.join(SAVE_DIR, "captured2.jpg")

# Make sure the output directory exists
if not os.path.isdir(SAVE_DIR):
    try:
        os.makedirs(SAVE_DIR)
    except Exception as e:
        print(f"Failed to create directory {SAVE_DIR}: {e}")
        exit()

# 1) Capture a single frame from RTSP
cap = cv2.VideoCapture("rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/701")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error grabbing frame from RTSP stream.")
    exit()

# 2) Save the frame to an image file
saved = cv2.imwrite(SAVE_PATH, frame)
if not saved:
    print(f"Error: Failed to write the image to {SAVE_PATH}.")
    exit()

if not os.path.isfile(SAVE_PATH):
    print(f"Error: File {SAVE_PATH} not found after attempting to save.")
    exit()

print(f"Image saved to {SAVE_PATH} successfully.")

# 3) Run DeepFace.find on multiple face databases
for db_name in ["victoria", "other", "group"]:
    db_path = f"./database/{db_name}"
    try:
        df = DeepFace.find(
            img_path=SAVE_PATH,
            db_path=db_path,
            model_name="VGG-Face",
            distance_metric="cosine"
        )
        print(f"Results from {db_name} database:")
        print(df)
    except Exception as e:
        print(f"Error searching in {db_name} database: {e}")
