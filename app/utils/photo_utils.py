import datetime
import cv2
from scipy.spatial.distance import cosine
import os
import numpy as np

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


def findCosineDistance(embedding1, embedding2):
    """Ensure embeddings are numpy arrays before computing cosine distance."""
    embedding1 = np.array(embedding1) if not isinstance(embedding1, np.ndarray) else embedding1
    embedding2 = np.array(embedding2) if not isinstance(embedding2, np.ndarray) else embedding2
    return cosine(embedding1, embedding2)
