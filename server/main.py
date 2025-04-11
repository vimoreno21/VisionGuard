from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os, shutil, uvicorn
import asyncio, json
import traceback
from typing import Optional, Dict
from fastapi.responses import StreamingResponse
import cv2
import time
from video_stream import add_video_routes, start_video_stream

# Load environment variables from a .env file (if available)
load_dotenv()

app = FastAPI()

# Use environment variables for configuration
DATABASE_ROOT = os.getenv("DATABASE_ROOT")
API_URL = os.getenv("API_URL")
PEOPLE_INSIDE_FILE = os.getenv("PEOPLE_INSIDE_FILE")
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME")
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
IP_ADDRESS = os.getenv("IP_ADDRESS")
PORT = os.getenv("PORT")
CAMERA_ID = os.getenv("CAMERA_ID")

# In-memory list (or switch to a DB later)
CURRENT_PEOPLE = []

# Mount static files and set up templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class Person(BaseModel):
    id: int = Field(..., alias='id')  
    name: str
    face_image: Optional[str] = None


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

# Add video routes to the app
add_video_routes(app)

if __name__ == '__main__':
    # Start the video streaming thread
    video_thread = start_video_stream()

    uvicorn.run(app, host="0.0.0.0", port=8000)