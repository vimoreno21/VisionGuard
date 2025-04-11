from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import shutil

app = FastAPI()

def update_people_inside(new_person):
    """
    Add a new person to the people_inside list if they're not already present,
    and send the update to the backend API.
    """
    global people_inside
    with people_inside_lock:
        # Check if the person is already in the list
        if not any(p.get("name") == new_person.get("name") for p in people_inside):
            people_inside.append(new_person)
            logger.info(f"update_people_inside: Added new person: {new_person}")

            # Send the update to the backend API (Render)
            try:
                response = requests.post(API_URL, json={"people": people_inside})
                if response.status_code == 200:
                    logger.info(f"update_people_inside: Successfully updated backend with new people list.")
                else:
                    logger.error(f"update_people_inside: Failed to update backend, status code {response.status_code}")
            except Exception as e:
                logger.error(f"update_people_inside: Error sending data to backend: {e}")
        else:
            logger.info(f"update_people_inside: Person {new_person} already exists in people_inside")