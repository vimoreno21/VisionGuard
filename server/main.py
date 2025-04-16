from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Form, Depends
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, shutil, uvicorn
import asyncio, json
import traceback
import subprocess
from typing import Optional, Dict
from datetime import timedelta
import cv2
from utils.helper import load_metadata, sanitize_filename, get_public_url, get_unique_filename
from utils.video_stream import add_video_routes, start_video_stream, get_rtsp_url
from utils.supabase_client import supabase, SUPABASE_BUCKET, SUPABASE_URL
import io
# Import authentication modules
from auth import (
    authenticate_user, create_access_token, 
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)


load_dotenv()
app = FastAPI()

# In-memory list 
CURRENT_PEOPLE = []
video_thread = None

# Mount static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.globals["get_public_url"] = get_public_url

class Person(BaseModel):
    id: int = Field(..., alias='id')  
    name: str
    face_image: Optional[str] = None

def ensure_video_stream_running():
    global video_thread
    if video_thread is None:
        print("Starting video stream")
        video_thread = start_video_stream()
    return video_thread

# Add video routes to the app
add_video_routes(app)


# --- Authentication routes ---
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: Optional[str] = None):
    """Render the login page"""
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login(
    request: Request, 
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """Process login form submission"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        return templates.TemplateResponse(
            "login.html", 
            {"request": request, "error": "Invalid username or password"},
            status_code=401
        )
    
    # Create access token with specified expiry time
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, 
        expires_delta=access_token_expires
    )
    
    # Set cookie and redirect to homepage
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=access_token,  # No Bearer prefix needed
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax"
    )
    return response

@app.get("/logout")
async def logout():
    """Log out by clearing the authentication cookie"""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")
    return response

# ---- API Endpoints ----
@app.post("/api/update_people_batch")
def update_people_batch(data: dict = Body(...), current_user: str = Depends(get_current_active_user)):
    global CURRENT_PEOPLE
    try:
        incoming_people = [Person(**p) for p in data.get("people", [])]

        # Build a dictionary keyed by ID to automatically deduplicate.
        # Also skip adding a new person if the name (when not "Unknown") is already present.
        people_dict: Dict[int, Person] = {}
        for p in incoming_people:
            if p.name != "Unknown" and any(existing.name == p.name for existing in people_dict.values()):
                continue  # Skip adding duplicate names.
            people_dict[p.id] = p

        # Update global list with deduplicated people.
        CURRENT_PEOPLE = list(people_dict.values())

        return {"status": "ok", "people_tracked": [p.model_dump() for p in CURRENT_PEOPLE]}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.get("/api/supabase/persons")
async def list_persons_supabase(current_user: str = Depends(get_current_active_user)):
    metadata = load_metadata()

@app.get("/api/supabase/persons")
async def list_persons_supabase():
    metadata = load_metadata()

@app.get("/api/supabase/persons")
async def list_persons_supabase():
    metadata = load_metadata()

    # Construct full public URLs
    people = {}
    for person, images in metadata.items():
        people[person] = [
            f"{os.getenv('SUPABASE_URL')}/storage/v1/object/public/{SUPABASE_BUCKET}/{img}"
            for img in images
        ]
    return {"persons": people}


@app.get("/api/current_people")
def get_current_people(current_user: str = Depends(get_current_active_user)):
    is_secure = all(p.name != "Unknown" for p in CURRENT_PEOPLE)
    system_status = "secure" if is_secure else "not secure"
    return {
        "people": [p.model_dump() for p in CURRENT_PEOPLE],
        "system_status": system_status
    }

# deleting an image
@app.post("/api/person/{person_name}/delete/{filename}")
async def delete_image(person_name: str, filename: str, current_user: str = Depends(get_current_active_user)):
    file_path = filename

    # Delete file from Supabase
    supabase.storage.from_(SUPABASE_BUCKET).remove([file_path])

    # Update metadata
    metadata = load_metadata()
    if person_name in metadata and filename in metadata[person_name]:
        metadata[person_name].remove(filename)

        # Re-upload updated metadata
        updated = json.dumps(metadata).encode("utf-8")
        supabase.storage.from_(SUPABASE_BUCKET).update("metadata.json", updated)

    return RedirectResponse(
        url=f"/allowed_access?person={person_name}&message=Deleted+{filename}&success=true",
        status_code=303
    )

# deleting a person
@app.post("/api/person/{person_name}/delete")
async def delete_person(person_name: str, current_user: str = Depends(get_current_active_user)):
    metadata = load_metadata()
    if person_name not in metadata:
        raise HTTPException(status_code=404, detail="Person not found")

    # Delete all images
    file_list = metadata[person_name]
    supabase.storage.from_(SUPABASE_BUCKET).remove(file_list)

    # Remove from metadata
    del metadata[person_name]
    updated = json.dumps(metadata).encode("utf-8")
    supabase.storage.from_(SUPABASE_BUCKET).update("metadata.json", updated)

    return RedirectResponse(
        url=f"/allowed_access?message=Deleted+{person_name}&success=true",
        status_code=303
    )


# --- UI Endpoints ---

# Dashboard page
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, current_user: str = Depends(get_current_active_user)):
    # Now that user is authenticated, ensure stream is running
    ensure_video_stream_running()

    metadata = load_metadata()  # Same helper you use elsewhere
    current_people = [p.model_dump() for p in CURRENT_PEOPLE]
    persons = []
    print("Metadata loaded:", metadata)

    for person, image_filenames in metadata.items():
        image_list = []
        for filename in image_filenames:
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            image_list.append({"filename": filename, "url": image_url})

        thumbnail = image_list[0]["url"] if image_list else "/static/images/default.jpg"

        persons.append({
            "name": person,
            "thumbnail": thumbnail,
            "images": image_list
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "persons": persons,
        "current_people": current_people,
    })


# Upload page (GET shows the form)
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, current_user: str = Depends(get_current_active_user)):
    return templates.TemplateResponse("upload.html", {"request": request})


# Upload page (POST handles the form submission)
@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(
    request: Request,
    person_name: str = Form(...),
    file: UploadFile = File(...), current_user: str = Depends(get_current_active_user)
):
    metadata = load_metadata()
    existing_files = [img for imgs in metadata.values() for img in imgs]

    filename = sanitize_filename(f"{person_name}_{file.filename}")
    filename = get_unique_filename(filename, existing_files)
    content = await file.read()

    # Upload image
    supabase.storage.from_(SUPABASE_BUCKET).upload(filename, content)

    # Download and update metadata.json
    metadata = load_metadata()

    if person_name not in metadata:
        metadata[person_name] = []
    metadata[person_name].append(filename)

    supabase.storage.from_(SUPABASE_BUCKET).update(
        "metadata.json", json.dumps(metadata).encode()
    )

    return RedirectResponse(url="/", status_code=303)


# Alerts page
@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request, current_user: str = Depends(get_current_active_user)):
    # Replace with actual alert logic; here's dummy data for illustration
    alerts = [{"timestamp": "2023-04-01 12:00", "message": "Unknown person detected", "thumbnail": "/static/images/unknown1.jpg"}]
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# Current Access page
@app.get("/current-access", response_class=HTMLResponse)
async def current_access_page(request: Request, current_user: str = Depends(get_current_active_user)):
    message = request.query_params.get("message", "")
    success = request.query_params.get("success", "true").lower() == "true"

    metadata = load_metadata()  # From your shared helper
    persons = []

    for person, image_filenames in metadata.items():
        image_list = []
        for filename in image_filenames:
            full_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            image_list.append({"filename": filename, "url": full_url})
        
        thumbnail_url = image_list[0]["url"] if image_list else "/static/images/default.jpg"

        persons.append({
            "name": person,
            "thumbnail": thumbnail_url,
            "images": image_list
        })

    return templates.TemplateResponse("current_access.html", {
        "request": request,
        "persons": persons,
        "message": message,
        "success": success
    })

@app.get("/allowed_access", response_class=HTMLResponse)
async def allowed_access(request: Request, current_user: str = Depends(get_current_active_user)):
    selected = request.query_params.get("person")
    metadata = load_metadata()
    persons = []

    for person, image_filenames in metadata.items():
        thumbnail = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filenames[0]}" if image_filenames else None
        persons.append({
            "name": person,
            "thumbnail": thumbnail,
            "images": image_filenames,
            "is_selected": (person == selected)
        })

    return templates.TemplateResponse("allowed_access.html", {
        "request": request,
        "persons": persons
    })


# --- Diagnostic Endpoints ---

@app.get("/debug/environment")
def debug_environment(current_user: str = Depends(get_current_active_user)):
    """Return filtered environment information"""
    # Filter out sensitive information
    safe_env = {}
    for key, value in os.environ.items():
        if any(sensitive in key.lower() for sensitive in 
               ["password", "secret", "key", "token", "auth"]):
            safe_env[key] = "***REDACTED***"
        else:
            safe_env[key] = value
    
    return safe_env

@app.get("/debug/opencv")
def debug_opencv(current_user: str = Depends(get_current_active_user)):
    """Return OpenCV and FFMPEG information"""
    try:
        import cv2
        
        # Check if FFMPEG is available
        ffmpeg_available = hasattr(cv2, "CAP_FFMPEG")
        
        # Try to get OpenCV build information
        build_info = cv2.getBuildInformation() if hasattr(cv2, "getBuildInformation") else "Not available"
        
        # Extract FFMPEG section from build info
        ffmpeg_section = "Not found"
        if "FFMPEG:" in build_info:
            start = build_info.find("FFMPEG:")
            end = build_info.find("\n\n", start)
            if end > start:
                ffmpeg_section = build_info[start:end]
        
        return {
            "opencv_version": cv2.__version__,
            "ffmpeg_available": ffmpeg_available,
            "ffmpeg_info": ffmpeg_section
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/rtsp_test")
async def test_rtsp_connection(current_user: str = Depends(get_current_active_user)):
    """Test RTSP connection and return detailed results"""
    ensure_video_stream_running()
    url, masked_url = get_rtsp_url()
    
    try:
        # Test with OpenCV
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        opencv_success = cap.isOpened()
        
        if opencv_success:
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Read one frame to confirm it works
            ret, frame = cap.read()
            frame_success = ret
            
            # Release the capture
            cap.release()
        else:
            width, height, fps, frame_success = None, None, None, False
        
        # Try with ffmpeg command line as fallback test
        try:
            # Run ffmpeg with a 3-second timeout
            ffmpeg_cmd = [
                'ffmpeg',
                '-loglevel', 'error',
                '-rtsp_transport', 'tcp',
                '-i', url,
                '-t', '1',  # Just capture 1 second
                '-f', 'null',
                '-'
            ]
            
            ffmpeg_process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3
            )
            
            ffmpeg_success = ffmpeg_process.returncode == 0
            ffmpeg_error = ffmpeg_process.stderr.decode('utf-8', errors='ignore')
        except (subprocess.SubprocessError, OSError) as e:
            ffmpeg_success = False
            ffmpeg_error = str(e)
        
        return {
            "url_tested": masked_url,
            "opencv_test": {
                "success": opencv_success,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_read_success": frame_success
            },
            "ffmpeg_test": {
                "success": ffmpeg_success,
                "error": ffmpeg_error if not ffmpeg_success else None
            }
        }
    except Exception as e:
        import traceback
        return {
            "url_tested": masked_url,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == '__main__':
    # Get port from environment or use 8000
    port = int(os.environ['PORT']) if 'PORT' in os.environ else 8000
    
    uvicorn.run(app, host="0.0.0.0", port=port)