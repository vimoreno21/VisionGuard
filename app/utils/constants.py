import os
from dotenv import load_dotenv

MATCH_THRESHOLD = 0.5  # Lower values = stricter matching

load_dotenv()
# API_URL=os.getenv("API_URL")
# API_URL = "http://localhost:8000"  # Update this URL
API_URL = "http://127.0.0.1:8000"  # Update this URL