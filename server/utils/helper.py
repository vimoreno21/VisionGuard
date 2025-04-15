import io
import json
from fastapi import HTTPException
from utils.supabase_client import supabase, SUPABASE_BUCKET
import re
import os
import time

def get_unique_filename(base_name, existing):
    name, ext = os.path.splitext(base_name)
    counter = 1
    candidate = base_name
    while candidate in existing:
        candidate = f"{name}_{counter}{ext}"
        counter += 1
    return candidate


def sanitize_filename(name: str) -> str:
    # Replace spaces and forbidden characters
    name = re.sub(r"[^\w\-.]", "_", name)
    return name

def load_metadata():
    try:
        timestamp = int(time.time())  # forces freshness
        raw = supabase.storage.from_(SUPABASE_BUCKET).download(f"metadata.json?t={timestamp}")
        return json.load(io.BytesIO(raw))
    except Exception as e:
        print("metadata.json not found or invalid:", e)
        return {}

def get_public_url(filename):
    return f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
