from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, shutil, uvicorn
import asyncio, json
import traceback
import subprocess
from typing import Optional, Dict
from fastapi.responses import StreamingResponse
import cv2
import time
from video_stream import add_video_routes, start_video_stream, get_rtsp_url
import requests

# Load environment variables from a .env file (if available)
load_dotenv()

app = FastAPI()

# In-memory list (or switch to a DB later)
CURRENT_PEOPLE = []

# Mount static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATABASE_ROOT = os.environ['DATABASE_ROOT']

class Person(BaseModel):
    id: int = Field(..., alias='id')  
    name: str
    face_image: Optional[str] = None

# Add video routes to the app
add_video_routes(app)

# Initialize video stream at module level
print("About to start video stream thread")
video_thread = start_video_stream()
print(f"Video thread started: {video_thread}")

@app.get("/render_ip")
def get_render_ip():
    return {"render_ip": requests.get("https://api.ipify.org").text}

@app.post("/api/update_people_batch")
def update_people_batch(data: dict = Body(...)):
    global CURRENT_PEOPLE
    try:
        print("Incoming payload:", data)
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


@app.post("/api/update_people_batch")
def update_people_batch(data: dict = Body(...)):
    global CURRENT_PEOPLE
    try: 
        print("Incoming payload:", data)
        incoming_people = [Person(**p) for p in data.get("people", [])]

        # IDs currently visible
        active_ids = {p.id for p in incoming_people}

        # Update or add active people
        updated = []
        for new_p in incoming_people:
            found = False
            for i, old_p in enumerate(CURRENT_PEOPLE):
                if old_p.id == new_p.id:
                    if old_p.name == "Unknown" and new_p.name != "Unknown":
                        # Avoid name conflict
                        if not any(p.name == new_p.name for p in CURRENT_PEOPLE if p.id != new_p.id):
                            CURRENT_PEOPLE[i] = new_p
                    else:
                        CURRENT_PEOPLE[i] = new_p
                    found = True
                    break
            if not found:
                if new_p.name != "Unknown" and any(p.name == new_p.name for p in CURRENT_PEOPLE):
                    continue  # avoid duplicates
                CURRENT_PEOPLE.append(new_p)

        # Remove people not in current frame
        CURRENT_PEOPLE = [p for p in CURRENT_PEOPLE if p.id in active_ids]

        return {"status": "ok", "people_tracked": [p.dict() for p in CURRENT_PEOPLE]}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


@app.get("/api/persons")
async def list_persons():
    persons = [d for d in os.listdir(DATABASE_ROOT) if os.path.isdir(os.path.join(DATABASE_ROOT, d))]
    return {"persons": persons}

@app.get("/api/current_people")
def get_current_people():
    return {"people": [p.model_dump() for p in CURRENT_PEOPLE]}


@app.get("/api/person/{person_name}/images")
async def list_images(person_name: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    if not os.path.exists(person_dir):
        raise HTTPException(status_code=404, detail="Person directory not found")
    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return {"images": images}

    
@app.get("/api/person/{person_name}/images/{image_filename}")
async def get_image(person_name: str, image_filename: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    file_path = os.path.join(person_dir, image_filename)
    print(f"Looking for file: {file_path}")  # Debugging: verify file path in console
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Image '{image_filename}' not found for {person_name}")
    return FileResponse(file_path)


@app.post("/api/person/{person_name}/upload")
async def upload_image(person_name: str, file: UploadFile = File(...)):
    """
    Upload an image for a specific person.
    """
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {"info": f"File '{file.filename}' saved on cloud."}

@app.post("/api/person/{person_name}/delete/{filename}")
async def delete_image(person_name: str, filename: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    file_path = os.path.join(person_dir, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"info": f"File '{filename}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/api/person/{person_name}/delete")
async def delete_person(person_name: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        return {"info": f"Person '{person_name}' and all images deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Person not found")

# --- UI Endpoints ---

# Dashboard page
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    allowed_access = []
    if os.path.exists(DATABASE_ROOT):
        for person in os.listdir(DATABASE_ROOT):
            person_dir = os.path.join(DATABASE_ROOT, person)
            if os.path.isdir(person_dir):
                # Get list of image files for the person
                images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Use the first image as a thumbnail, or a default if none exist
                thumbnail = images[0] if images else None
                allowed_access.append({
                    "name": person,
                    "thumbnail": thumbnail,
                    "images": images,
                })
    else:
        # return error message if the database root does not exist
        return templates.TemplateResponse("error.html", {"request": request, "message": "Database root does not exist."})

    return templates.TemplateResponse("dashboard.html", {"request": request, "allowed_access": allowed_access})


# Upload page (GET shows the form)
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# Upload page (POST handles the form submission)
@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, person_name: str = Form(...), file: UploadFile = File(...)):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # No forwarding is required—just save the file
    return RedirectResponse(url="/", status_code=303)

# Alerts page
@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    # Replace with actual alert logic; here's dummy data for illustration
    alerts = [{"timestamp": "2023-04-01 12:00", "message": "Unknown person detected", "thumbnail": "/static/images/unknown1.jpg"}]
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})

# Current Access page
@app.get("/current-access", response_class=HTMLResponse)
async def current_access_page(request: Request):
    # Get message parameter if it exists
    message = request.query_params.get("message", "")
    success = request.query_params.get("success", "true").lower() == "true"
    
    persons = []
    if os.path.exists(DATABASE_ROOT):
        for person in os.listdir(DATABASE_ROOT):
            person_dir = os.path.join(DATABASE_ROOT, person)
            if os.path.isdir(person_dir):
                # Look for image files in the person's folder
                images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Create a list of image details for this person
                image_list = []
                for img in images:
                    image_url = f"/api/person/{person}/image/{img}"
                    image_list.append({"filename": img, "url": image_url})
                
                # Use the first image as the main thumbnail or a default if none exist
                thumbnail_url = image_list[0]["url"] if image_list else "/static/images/default.jpg"
                
                # Add this person with all their images
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

# UI endpoint for deleting a person (replaces direct API call)
@app.post("/delete-person/{person_name}")
async def ui_delete_person(person_name: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        return RedirectResponse(
            url=f"/current-access?message=Person '{person_name}' and all images deleted successfully&success=true",
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/current-access?message=Person '{person_name}' not found&success=false",
            status_code=303
        )
        
# UI endpoint for deleting a single image
@app.post("/delete-image/{person_name}/{filename}")
async def ui_delete_image(person_name: str, filename: str):
    person_dir = os.path.join(DATABASE_ROOT, person_name)
    file_path = os.path.join(person_dir, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return RedirectResponse(
            url=f"/current-access?message=Image '{filename}' deleted successfully&success=true",
            status_code=303
        )
    else:
        return RedirectResponse(
            url=f"/current-access?message=Image '{filename}' not found&success=false",
            status_code=303
        )

# --- Diagnostic Endpoints ---

@app.get("/debug/environment")
def debug_environment():
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
def debug_opencv():
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
async def test_rtsp_connection():
    """Test RTSP connection and return detailed results"""
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
    # Start the video streaming thread
    video_thread = start_video_stream()
    
    # Get port from environment or use 8000
    port = int(os.environ['PORT']) if 'PORT' in os.environ else 8000
    
    uvicorn.run(app, host="0.0.0.0", port=port)