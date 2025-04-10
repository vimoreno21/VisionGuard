import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LOG_FILES_DIR = "./logs/log_files"
OUTPUT_DIR = "./logs/output_files"
EMBEDDINGS_DIR = "./embeddings"
PEOPLE_FILE = os.path.join(BASE_DIR, "people_inside.json")

DATABASE_ROOT = os.path.join(BASE_DIR, "server", "database")

# Ensure directories exist
if not os.path.exists(DATABASE_ROOT):
    # throw an error if the database root directory does not exist
    raise FileNotFoundError(f"Database root directory does not exist: {DATABASE_ROOT}")

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
