import datetime
import cv2
import os

def get_person_name_from_path(image_path, db_path):
    """Extract person name from image path based on folder structure"""
    
    # Extract person name from path (assuming structure: db_path/person_name/image.jpg)
    relative_path = os.path.relpath(image_path, db_path)
    parts = relative_path.split(os.path.sep)
    
    if len(parts) > 1:
        person_name = parts[0]  # First folder inside db_path
    else:
        person_name = "unknown"
    
    return person_name

def current_timestamp():
    """Return current timestamp as string"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def mark_frame_with_face(frame, location, name=None, confidence=None):
    """
    Draw a box around the detected face with optional name and confidence
    """
    # Create a copy to avoid modifying the original
    marked_frame = frame.copy()
    
    x, y, w, h = location
    
    # Draw rectangle
    cv2.rectangle(marked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Add text if name is provided
    if name:
        label = f"{name}"
        if confidence:
            label += f" ({confidence:.2f})"
        
        # Background for text
        cv2.rectangle(marked_frame, (x, y - 30), (x + len(label) * 12, y), (0, 255, 0), -1)
        cv2.putText(marked_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return marked_frame