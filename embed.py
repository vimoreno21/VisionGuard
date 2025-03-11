import os
import pickle
import time
import numpy as np
from deepface import DeepFace

from utils.directories import EMBEDDINGS_DIR, DB_PATHS
from utils.photo_utils import findCosineDistance

# Global variables to store precomputed embeddings
reference_embeddings = {}
average_embeddings = {}

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
        print(f"Loading precomputed embeddings from {embeddings_path}")
        with open(embeddings_path, 'rb') as f:
            reference_embeddings = pickle.load(f)
        with open(avg_embeddings_path, 'rb') as f:
            average_embeddings = pickle.load(f)
        print(f"Loaded embeddings for {len(reference_embeddings)} images")
        return reference_embeddings, average_embeddings
    else:
        print("Embedding paths don't exist", embeddings_path, " ", avg_embeddings_path)
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
    
    print(f"Saved embeddings to {embeddings_path} and {avg_embeddings_path}")

def compute_embedding_for_image(image_path, model_name):
    """Compute facial embedding for a single image"""
    try:
        embedding = DeepFace.represent(
            img_path=image_path, 
            model_name=model_name,
            enforce_detection=False
        )
        return embedding[0]["embedding"] # embedding is a list of a dictionary 
    except Exception as e:
        print(f"Error computing embedding for {image_path}: {e}")
        return None
    
def compute_average_embeddings(person_embeddings):
    """Compute average embedding for each person"""
      
    average_embs = {}

    print(f"🔎 Computing average embeddings for {len(person_embeddings)} people...")
    
    for person, embeddings in person_embeddings.items():
        if isinstance(embeddings[0], dict):
            print(f"❌ ERROR: Embeddings for {person} contain dictionaries!")
            continue  # Skip to prevent crash
        
        if not embeddings:  # Skip empty lists
            print(f"⚠️ Warning: No embeddings found for {person}, skipping.")
            continue
        
        # Convert list to numpy array
        embeddings_array = np.array(embeddings)

        if embeddings_array.ndim != 2:  # Expecting shape (num_images, embedding_size)
            print(f"❌ ERROR: Unexpected embedding shape for {person}: {embeddings_array.shape}")
            continue  # Skip person if embeddings are malformed
        
        avg_embedding = np.mean(embeddings_array, axis=0)
        average_embs[person] = avg_embedding

        print(f"✅ Created average embedding for {person} from {len(embeddings)} images")

    return average_embs

def precompute_embeddings(model_name):
    """
    Main function to precompute and save embeddings for all images in the databases
    Returns dictionaries of individual embeddings and average embeddings per person
    """
    global reference_embeddings, average_embeddings
    
    print("Precomputing face embeddings for reference images...")
    start_time = time.time()
    
    # Try to load existing embeddings first
    loaded_ref, loaded_avg = load_saved_embeddings(model_name)
    if loaded_ref and loaded_avg:
        reference_embeddings = loaded_ref
        average_embeddings = loaded_avg
        print("Sucessfully loaded precomputed embeddings")
        return reference_embeddings, average_embeddings
    else:
        print("No saved embeddings found. Computing now...")
        print("load_saved_embeddings returned", loaded_avg)
    # Initialize dictionaries
    reference_embeddings = {}
    all_person_embeddings = {}  # To store embeddings grouped by person
    
    # Process each database
    for db_name in DB_PATHS:
        db_path = f"./database/{db_name}"
        
        if not os.path.exists(db_path):
            print(f"Warning: Database path {db_path} does not exist. Skipping.")
            continue
        
        print(f"Processing database: {db_name}")
        
        # Scan database and compute embeddings
        db_ref_embeddings, db_person_embeddings = scan_database(db_path, model_name)
        
        # Merge with main dictionaries
        reference_embeddings.update(db_ref_embeddings)
        
        # Merge person embeddings
        for person, embeddings in db_person_embeddings.items():
            if person not in all_person_embeddings:
                all_person_embeddings[person] = []
            all_person_embeddings[person].extend(embeddings)
    
    # Compute average embeddings for all persons
    average_embeddings = compute_average_embeddings(all_person_embeddings)
    
    # Save embeddings to disk for future use
    save_embeddings(reference_embeddings, average_embeddings, model_name)
    
    end_time = time.time()
    print(f"Precomputed embeddings for {len(reference_embeddings)} images in {end_time - start_time:.2f} seconds")
    
    return reference_embeddings, average_embeddings

def find_match_with_embeddings(face_img_path, model_name):
    """
    Find matching face using precomputed embeddings
    Returns: (identity, distance, match_type) of the best match, or (None, 1.0, None) if no match
    """
    global reference_embeddings, average_embeddings
    
    # Ensure we have embeddings
    if not reference_embeddings or not average_embeddings:
        print("Did not find precomputed embeddings. Computing now...")
        precompute_embeddings(model_name)
    
    try:
        # Get embedding for current face
        face_embedding = compute_embedding_for_image(face_img_path, model_name)
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
        print(f"Error in face matching: {e}")
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
                        print(f"❌ ERROR: Unexpected embedding shape {embedding.shape} for {image_path}")
                        continue
                    
                    person_embeddings[person_name].append(embedding)

                    print(f"✅ Processed: {image_path} (Person: {person_name}) - Shape: {embedding.shape}")
    
    return ref_embeddings, person_embeddings