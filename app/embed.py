import os
import pickle
import time
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import cv2

from utils.directories import EMBEDDINGS_DIR
from utils.photo_utils import findCosineDistance
from utils.logger import logger
from utils.load_database import get_image_map_from_metadata, supabase, SUPABASE_BUCKET

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
            logger.warning(f"Warning: No embeddings found for {person}, skipping.")
            continue
        
        # Convert list to numpy array
        embeddings_array = np.array(embeddings)

        if embeddings_array.ndim != 2:  # Expecting shape (num_images, embedding_size)
            logger.error(f"❌ ERROR: Unexpected embedding shape for {person}: {embeddings_array.shape}")
            continue  # Skip person if embeddings are malformed
        
        avg_embedding = np.mean(embeddings_array, axis=0)
        average_embs[person] = avg_embedding

        logger.debug(f"Created average embedding for {person} from {len(embeddings)} images")

    return average_embs

def precompute_embeddings(model_name):
    """
    Compute embeddings from Supabase using metadata.json.
    Saves both individual and average embeddings to disk.
    """
    global reference_embeddings, average_embeddings

    logger.debug("⚙️ Precomputing face embeddings using metadata.json...")

    # Step 1: Try loading saved embeddings
    loaded_ref, loaded_avg = load_saved_embeddings(model_name)
    if loaded_ref and loaded_avg:
        reference_embeddings = loaded_ref
        average_embeddings = loaded_avg
        logger.debug("Loaded saved embeddings from disk")
        return reference_embeddings, average_embeddings

    # Step 2: Fresh init
    reference_embeddings = {}
    all_person_embeddings = {}

    # Step 3: Get image list from metadata.json instead of scanning storage
    metadata = get_image_map_from_metadata()

    for person_name, filenames in metadata.items():
        for filename in filenames:
            image_key = filename
            try:
                res = supabase.storage.from_(SUPABASE_BUCKET).download(image_key)
                img_array = cv2.imdecode(np.frombuffer(res, np.uint8), cv2.IMREAD_COLOR)
                if img_array is None:
                    logger.warning(f"Could not decode image: {image_key}")
                    continue

                embedding = compute_embedding_for_image(img_array, model_name)
                if embedding is not None:
                    reference_embeddings[image_key] = embedding  # Save embedding under full key
                    all_person_embeddings.setdefault(person_name, []).append(embedding)
                    logger.debug(f"Embedded {image_key}")
            except Exception as e:
                logger.error(f"Failed to process {image_key}: {e}")

    # Step 4: Compute average embeddings per person
    average_embeddings = compute_average_embeddings(all_person_embeddings)

    # Step 5: Save everything locally (.pkl)
    save_embeddings(reference_embeddings, average_embeddings, model_name)
    logger.debug("Embeddings saved to disk")

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

def update_pkls(model_name):
    """
    Update saved embeddings with any new or deleted images based on Supabase metadata.json.
    Embeddings are saved to local .pkl files.
    """
    global reference_embeddings, average_embeddings

    # ✅ Try loading existing embeddings from disk
    ref_embeddings, avg_embeddings = load_saved_embeddings(model_name)
    if ref_embeddings is None or avg_embeddings is None:
        logger.debug("No saved embeddings found. Running full precomputation.")
        precompute_embeddings(model_name)
        return

    reference_embeddings = ref_embeddings
    average_embeddings = avg_embeddings
    new_changes = False

    # Load metadata from Supabase to get current known files
    metadata = get_image_map_from_metadata()
    current_files = set()
    
    for person, filenames in metadata.items():
        for filename in filenames:
            current_files.add(filename)
            if filename not in reference_embeddings:
                try:
                    res = supabase.storage.from_(SUPABASE_BUCKET).download(filename)
                    img_array = cv2.imdecode(np.frombuffer(res, np.uint8), cv2.IMREAD_COLOR)
                    if img_array is None:
                        logger.warning(f"Could not decode image: {filename}")
                        continue

                    embedding = compute_embedding_for_image(img_array, model_name)
                    if embedding is not None:
                        reference_embeddings[filename] = embedding
                        new_changes = True
                        logger.debug(f"Added new embedding for: {filename}")
                except Exception as e:
                    logger.error(f"Error downloading {filename}: {e}")

    # Remove embeddings that no longer exist in metadata
    for filename in list(reference_embeddings.keys()):
        if filename not in current_files:
            del reference_embeddings[filename]
            new_changes = True
            logger.debug(f"Removed stale embedding: {filename}")

    # Rebuild person-level embedding groups
    person_embeddings = {}
    for filename, embedding in reference_embeddings.items():
        person_name = filename.split("_")[0] 
        person_embeddings.setdefault(person_name, []).append(embedding)

    # Save if anything changed
    if new_changes:
        average_embeddings = compute_average_embeddings(person_embeddings)
        save_embeddings(reference_embeddings, average_embeddings, model_name)
        logger.debug("Embeddings updated and saved to disk")
    else:
        logger.debug("No changes detected. Embeddings remain unchanged.")


