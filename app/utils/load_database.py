from supabase import create_client, Client
import json
import os
from utils.logger import logger

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_image_map_from_metadata():
    """Fetch metadata.json from Supabase and return {person_name: [img1.jpg, img2.jpg, ...]}"""
    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).download("metadata.json")
        metadata = json.loads(res.decode("utf-8"))
        return metadata
    except Exception as e:
        logger.exception(f"Failed to load metadata.json from {SUPABASE_BUCKET}: {e}")
        return {}

