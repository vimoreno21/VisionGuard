from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import shutil

app = FastAPI()

# Local base folder for storing images on the Jetson
LOCAL_DB_BASE = "app/database"

@app.post("/api/person/{person_name}/upload")
async def upload_image(person_name: str, file: UploadFile = File(...)):
    """
    Receive an image upload for a specific person and store it locally.
    """
    # Define the local path: VisionGuard/app/database/<person>/img
    person_dir = os.path.join(LOCAL_DB_BASE, person_name, "img")
    os.makedirs(person_dir, exist_ok=True)
    file_path = os.path.join(person_dir, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {"info": f"File '{file.filename}' saved locally at '{file_path}'."}

@app.post("/api/person/{person_name}/delete{filename}")
async def delete_image(person_name: str, filename: str):
    """
    Delete an image for a specific person.
    """
    # Define the local path: /app/database/<person>/
    person_dir = os.path.join(LOCAL_DB_BASE, person_name, "img")
    file_path = os.path.join(person_dir, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"info": f"File '{filename}' deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Image not found")
    
@app.post("/api/person/{person_name}/delete")
async def delete_person(person_name: str):
    """
    Delete a person and all their images.
    """
    # Define the local path: VisionGuard/app/database/<person>
    person_dir = os.path.join(LOCAL_DB_BASE, person_name)
    
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
        return {"info": f"Person '{person_name}' and all their images deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Person not found")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
