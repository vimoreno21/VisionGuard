from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
import shutil
import requests

# Load environment variables from a .env file (if available)
load_dotenv()

app = FastAPI()

# Use environment variables for configuration
UPLOAD_BASE = os.getenv("UPLOAD_BASE")
JETSON_URL = os.getenv("JETSON_URL")

os.makedirs(UPLOAD_BASE, exist_ok=True)

@app.get("/api/persons")
async def list_persons():
    persons = [d for d in os.listdir(UPLOAD_BASE) if os.path.isdir(os.path.join(UPLOAD_BASE, d))]
    return {"persons": persons}

@app.get("/api/person/{person_name}/images")
async def list_images(person_name: str):
    person_dir = os.path.join(UPLOAD_BASE, person_name)
    if not os.path.exists(person_dir):
        raise HTTPException(status_code=404, detail="Person directory not found")
    images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return {"images": images}

@app.get("/api/person/{person_name}/image/{filename}")
async def get_image(person_name: str, filename: str):
    person_dir = os.path.join(UPLOAD_BASE, person_name)
    file_path = os.path.join(person_dir, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/api/person/{person_name}/upload")
async def upload_image(person_name: str, file: UploadFile = File(...)):
    """
    Upload an image for a specific person. After saving it locally on the cloud server,
    the image is forwarded to the Jetson to be stored locally.
    """
    person_dir = os.path.join(UPLOAD_BASE, person_name)
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Forward the image to the Jetson's API endpoint
    try:
        with open(file_path, "rb") as image_file:
            files = {"file": (file.filename, image_file, file.content_type)}
            jetson_endpoint = f"{JETSON_URL}/api/person/{person_name}/upload"
            response = requests.post(jetson_endpoint, files=files)
            response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image uploaded but failed to forward to Jetson: {e}")
    
    return {"info": f"File '{file.filename}' saved on cloud and forwarded to Jetson."}

@app.post("/api/person/{person_name}/delete/{filename}")
async def delete_image(person_name: str, filename: str):
    person_dir = os.path.join(UPLOAD_BASE, person_name)
    file_path = os.path.join(person_dir, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        try:
            jetson_endpoint = f"{JETSON_URL}/api/person/{person_name}/delete?filename={filename}"
            response = requests.post(jetson_endpoint)
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File deleted locally but failed to notify Jetson: {e}")
        return {"info": f"File '{filename}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.post("/api/person/{person_name}/delete")
async def delete_person(person_name: str):
    person_dir = os.path.join(UPLOAD_BASE, person_name)
    
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        try:
            jetson_endpoint = f"{JETSON_URL}/api/person/{person_name}/delete"
            response = requests.post(jetson_endpoint)
            response.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Person deleted locally but failed to notify Jetson: {e}")
        return {"info": f"Person '{person_name}' and all images deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Person not found")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
