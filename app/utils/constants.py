import os
from dotenv import load_dotenv

MATCH_THRESHOLD = 0.5  # Lower values = stricter matching

load_dotenv()
# to connect to API
API_URL=os.getenv("API_URL")
# API_URL = "http://localhost:8000"

# to connect to camera
RTSP_URL_BASE = os.getenv("RTSP_URL_BASE")
CAM_NUM = os.getenv("FRONT_DOOR_CAM")
CAM2_NUM = os.getenv("STAIRS_CAM")
RTSP_URL = f"{RTSP_URL_BASE}{CAM2_NUM}"


# for database
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

# Parameters for pipeline
downscale = 0.7
conf_thresh = 0.9
pad = 50
detection_interval = 6
track_interval = 3