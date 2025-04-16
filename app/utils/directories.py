import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", ".."))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

LOG_FILES_DIR = "./logs/log_files"
OUTPUT_DIR = "./logs/output_files"
EMBEDDINGS_DIR = "./embeddings"


if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

        
# Create debug directory
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
