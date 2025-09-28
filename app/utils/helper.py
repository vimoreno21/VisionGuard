import requests
import cv2
from io import BytesIO
from PIL import Image
import base64

from utils.constants import API_URL
from utils.logger import logger


def people_changed(current, last):
    if len(current) != len(last):
        return True
    for c, l in zip(current, last):
        if c["id"] != l["id"] or (c["name"] != l["name"] and c["name"] != "Unknown"):
            return True
    return False

def send_people_inside_batch(current_people):
    try:
        logger.debug(f"Sending to backend")
        response = requests.post(
            f"{API_URL}/api/update_people_batch",
            json={"people": current_people}
        )
        if response.status_code == 200:
            logger.debug("Sent people_inside batch update.")
        else:
            logger.warning(f"Server responded with {response.status_code}: {response.text}")
            raise Exception(f"Failed to update server: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending people_inside batch: {e}")
        raise e


def match_track_to_box(box, tracked_objects):
    """
    Find the track whose bbox has highest IoU with this face box.
    You can swap in your own centroid or IoU logic here.
    """
    x1, y1, x2, y2 = box
    best_id, best_iou = None, 0
    for t in tracked_objects:
        tx1, ty1, tx2, ty2 = map(int, t.to_tlbr())
        # compute IoU
        xi1, yi1 = max(x1, tx1), max(y1, ty1)
        xi2, yi2 = min(x2, tx2), min(y2, ty2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = (x2-x1)*(y2-y1) + (tx2-tx1)*(ty2-ty1) - inter
        iou = inter/union if union>0 else 0
        if iou > best_iou:
            best_iou, best_id = iou, t.track_id
    return best_id

def encode_to_base64(crop):
    """Crop → JPEG → base64 data URI."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def batch_detect_and_prepare(frame, detector, tracked_objects,
                             identified_identities, conf_thresh=0.9):
    """
    Returns:
      - people_visible: list of dicts for UI update
      - unknown_crops: list of (crop, track_id) for recognition
    """
    people_visible = []
    unknown_crops = []

    # 1) one MTCNN call
    boxes, probs = detector.detect(frame)
    if boxes is None:
        return people_visible, unknown_crops

    for box, p in zip(boxes, probs):
        if p < conf_thresh:
            continue
        x1,y1,x2,y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        tid  = match_track_to_box((x1,y1,x2,y2), tracked_objects)
        img_str = encode_to_base64(crop)

        people_visible.append({
            "id":         tid,
            "name":       identified_identities.get(tid, "Unknown"),
            "face_image": img_str
        })

        if tid not in identified_identities:
            unknown_crops.append((crop, tid))

    return people_visible, unknown_crops
