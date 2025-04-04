import os

SAVE_DIR = "./logs/captured_images"
LOG_FILES_DIR = "./logs/log_files"
OUTPUT_DIR = "./logs/output_files"
EMBEDDINGS_DIR = "./embeddings"

# Ensure directories exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

        
# Create debug directory
DEBUG_DIR = os.path.join(SAVE_DIR, "debug")
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)
