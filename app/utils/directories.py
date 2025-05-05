import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

EMBEDDINGS_DIR = "./embeddings"
LOG_FILES_DIR = "./logs/log_files" # if logging to file is enabled
VIDEO_DIR = "./logs/video_files"
DETECTED_FACES_DIR = "./logs/detected_faces"
MATCHES_DIR = "./logs/matches"


os.makedirs(LOG_FILES_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)
os.makedirs(MATCHES_DIR, exist_ok=True)