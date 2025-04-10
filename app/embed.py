import os
import pickle
import time
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import cv2

from utils.directories import EMBEDDINGS_DIR, DATABASE_ROOT
from utils.photo_utils import findCosineDistance
from utils.logger import logger

# Global variables to store precomputed embeddings
reference_embeddings = {}
average_embeddings = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_saved_embeddings(model_name):
    """
    Try to load previously saved embeddings from disk
    Returns: (reference_embeddings, average_embeddings) if found, else (None, None)
    """
    global reference_embeddings, average_embeddings
    
    # Make sure the embeddings directory exists
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Path to saved embeddings
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{model_name}_embeddings.pkl")
    avg_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{model_name}_avg_embeddings.pkl")
    
    # Check if both files exist
    if os.path.exists(embeddings_path) and os.path.exists(avg_embeddings_path):
        logger.debug(f"Loading precomputed embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            reference_embeddings = pickle.load(f)
        with open(avg_embeddings_path, 'rb') as f:
            average_embeddings = pickle.load(f)
        logger.debug(f"Loaded embeddings for {len(reference_embeddings)} images")
        return reference_embeddings, average_embeddings
    else:
        logger.error(f"Embedding paths don't exist {embeddings_path} {avg_embeddings_path}")
    return None, None

def save_embeddings(reference_embeddings, average_embeddings, model_name):
    """Save computed embeddings to disk for future use"""
    
    # Make sure the embeddings directory exists
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Path to save embeddings
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{model_name}_embeddings.pkl")
    avg_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{model_name}_avg_embeddings.pkl")
    
    # Save embeddings to disk
    with open(embeddings_path, 'wb') as f:
        pickle.dump(reference_embeddings, f)
    
    with open(avg_embeddings_path, 'wb') as f:
        pickle.dump(average_embeddings, f)
    
    logger.debug(f"Saved embeddings to {embeddings_path} and {avg_embeddings_path}")

def compute_embedding_for_image(image, model_name):
    """Compute facial embedding for an image array using GPU acceleration."""
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                logger.error(f"Failed to load image from path: {image}")
                raise ValueError(f"Image not found: {image}")

        # Convert from BGR (cv2) to RGB if needed:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(tensor)
        return embedding.cpu().numpy()[0]
    except Exception as e:
        logger.exception(f"Error computing embedding: {e}")
        raise

    
def compute_average_embeddings(person_embeddings):
    """Compute average embedding for each person"""
      
    average_embs = {}

    logger.debug(f"🔎 Computing average embeddings for {len(person_embeddings)} people...")
    
    for person, embeddings in person_embeddings.items():
        if isinstance(embeddings[0], dict):
            logger.error(f"❌ ERROR: Embeddings for {person} contain dictionaries!")
            continue  # Skip to prevent crash
        
        if not embeddings:  # Skip empty lists
            logger.warning(f"⚠️ Warning: No embeddings found for {person}, skipping.")
            continue
        
        # Convert list to numpy array
        embeddings_array = np.array(embeddings)

        if embeddings_array.ndim != 2:  # Expecting shape (num_images, embedding_size)
            logger.error(f"❌ ERROR: Unexpected embedding shape for {person}: {embeddings_array.shape}")
            continue  # Skip person if embeddings are malformed
        
        avg_embedding = np.mean(embeddings_array, axis=0)
        average_embs[person] = avg_embedding

        logger.debug(f"✅ Created average embedding for {person} from {len(embeddings)} images")

    return average_embs

def precompute_embeddings(model_name, database_root=DATABASE_ROOT):
    """
    Main function to precompute and save embeddings for all images in the databases.
    Scans the database_root directory for subdirectories (each representing a person),
    computes embeddings for all images, groups them by person, computes average embeddings,
    and saves the results.
    """

    global reference_embeddings, average_embeddings
    
    logger.debug("Precomputing face embeddings for reference images...")
    
    # Try to load existing embeddings first
    loaded_ref, loaded_avg = load_saved_embeddings(model_name)
    if loaded_ref and loaded_avg:
        reference_embeddings = loaded_ref
        average_embeddings = loaded_avg
        logger.debug("Successfully loaded precomputed embeddings")
    else:
        logger.debug("No saved embeddings found. Computing now...")
    
    # Initialize dictionaries for reference embeddings and person-specific embeddings
    reference_embeddings = {}
    all_person_embeddings = {}  # To store embeddings grouped by person
    
    # Ensure the database root exists
    if not os.path.exists(database_root):
        logger.exception(f"Database root path {database_root} does not exist. Exiting precomputation.")
        raise FileNotFoundError(f"Database root path {database_root} does not exist.")
    
    # Iterate over each subdirectory in the database root, each assumed to represent a person
    for person_folder in os.listdir(database_root):
        db_path = os.path.join(database_root, person_folder)
        if not os.path.isdir(db_path):
            continue
        logger.debug(f"Processing database for person: {person_folder}")
        
        # Scan the person's folder and compute embeddings
        db_ref_embeddings, db_person_embeddings = scan_database(db_path, model_name)
        
        # Merge the computed embeddings into the global reference embeddings
        reference_embeddings.update(db_ref_embeddings)
        
        # Merge person-specific embeddings
        for person, embeddings in db_person_embeddings.items():
            if person not in all_person_embeddings:
                all_person_embeddings[person] = []
            all_person_embeddings[person].extend(embeddings)
    
    # Compute average embeddings for each person
    average_embeddings = compute_average_embeddings(all_person_embeddings)
    
    # Save the updated embeddings to disk for future use
    save_embeddings(reference_embeddings, average_embeddings, model_name)
    
    return reference_embeddings, average_embeddings

def find_match_with_embeddings(face_img, model_name):
    """
    Find matching face using precomputed embeddings
    Returns: (identity, distance, match_type) of the best match, or (None, 1.0, None) if no match
    """
    global reference_embeddings, average_embeddings
    
    # Ensure we have embeddings
    if not reference_embeddings or not average_embeddings:
        logger.debug("Did not find precomputed embeddings. Computing now...")
        precompute_embeddings(model_name)
    
    try:
        # Get embedding for current face
        face_embedding = compute_embedding_for_image(face_img, model_name)
        if face_embedding is None:
            return None, 1.0, None
        
        best_match = (None, 1.0, None)  # (identity, distance, match_type)
        
        # 1. Compare against individual reference images
        for ref_path, ref_embedding in reference_embeddings.items():
            # Calculate distance (cosine distance)
            distance = findCosineDistance(face_embedding, ref_embedding)
            
            # Update best match if better
            if distance < best_match[1]:
                best_match = (ref_path, distance, "individual")
        
        # 2. Compare against average embeddings (this can be more robust)
        for person, avg_embedding in average_embeddings.items():
            # Calculate distance
            distance = findCosineDistance(face_embedding, avg_embedding)
            
            # Update best match if better
            if distance < best_match[1]:
                best_match = (person, distance, "average")
        
        return best_match
        
    except Exception as e:
        logger.exception(f"Error in face matching: {e}")
        return None, 1.0, None
    
def scan_database(db_path, model_name):
    """Scan a single database folder and compute embeddings for all images"""
    
    ref_embeddings = {}
    person_embeddings = {}

    for root, dirs, files in os.walk(db_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                # Get person name
                person_name = os.path.basename(db_path)  
                
                # Compute embedding
                embedding = compute_embedding_for_image(image_path, model_name)
                 

                if embedding is not None:
                    ref_embeddings[image_path] = embedding
                    
                    if person_name not in person_embeddings:
                        person_embeddings[person_name] = []

                    embedding = np.array(embedding)  # Convert to NumPy array
                    
                    if embedding.ndim != 1:  # Ensure it's a flat vector
                        logger.exception(f"❌ ERROR: Unexpected embedding shape {embedding.shape} for {image_path}")
                        continue
                    
                    person_embeddings[person_name].append(embedding)

                    logger.debug(f"✅ Processed: {image_path} (Person: {person_name}) - Shape: {embedding.shape}")
    
    return ref_embeddings, person_embeddings

def update_pkls(model_name, database_root=DATABASE_ROOT):
    """
    Update saved embeddings (.pkl files) with new images added and remove embeddings for images no longer on disk.
    This function scans the database_root directory for subdirectories (each representing a person),
    loads current embeddings, checks for new images, computes embeddings for them,
    detects removed images, updates the average embeddings, and saves the results.
    
    Does not return any values.
    """
    # Load current embeddings if they exist; if not, perform a full precomputation.
    ref_embeddings, avg_embeddings = load_saved_embeddings(model_name)
    if ref_embeddings is None or avg_embeddings is None:
        logger.debug("No saved embeddings found. Running full precomputation.")
        precompute_embeddings(model_name, database_root)
        return

    new_changes = False  # Flag to indicate if any additions or removals occurred

    # Build a set of existing image paths from the filesystem
    existing_files = set()
    for person_folder in os.listdir(database_root):
        person_path = os.path.join(database_root, person_folder)
        if not os.path.isdir(person_path):
            continue
        for root, dirs, files in os.walk(person_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    existing_files.add(image_path)
                    # If this image hasn't been processed yet, compute and add its embedding
                    if image_path not in ref_embeddings:
                        logger.debug(f"New image detected for {person_folder}: {image_path}")
                        embedding = compute_embedding_for_image(image_path, model_name)
                        if embedding is not None:
                            ref_embeddings[image_path] = embedding
                            new_changes = True

    # Detect and remove embeddings for images that no longer exist on disk
    for image_path in list(ref_embeddings.keys()):
        if image_path not in existing_files:
            logger.debug(f"Image removed: {image_path}")
            del ref_embeddings[image_path]
            new_changes = True

    # Rebuild grouping of embeddings by person from the updated ref_embeddings
    person_embeddings = {}
    for image_path, embedding in ref_embeddings.items():
        parts = os.path.normpath(image_path).split(os.sep)
        try:
            # Use the immediate subdirectory after the database root as the person name.
            db_index = parts.index(os.path.basename(os.path.normpath(database_root)))
            person_name = parts[db_index + 1]
        except (ValueError, IndexError):
            person_name = "unknown"
        if person_name not in person_embeddings:
            person_embeddings[person_name] = []
        person_embeddings[person_name].append(embedding)

    # If any changes occurred, update the average embeddings and save everything to disk
    if new_changes:
        avg_embeddings = compute_average_embeddings(person_embeddings)
        save_embeddings(ref_embeddings, avg_embeddings, model_name)
        logger.debug("Embeddings updated with changes from new or removed images.")
    else:
        logger.debug("No changes detected. Embeddings remain unchanged.")

