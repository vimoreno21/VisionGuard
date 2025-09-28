import time
import cv2


from utils.logger import logger
from utils.helper import encode_to_base64

from utils.constants import (pad, downscale, conf_thresh)


def check_tracked_objects(identified_identities, detector, frame, tracked_objects, people_visible, unknown_crops):
    # Detect faces only on unknown-track ROIs
    w, h = frame.shape[1], frame.shape[0] # initially using the one in main
    for t in tracked_objects:
        tid = t.track_id
        if tid in identified_identities:
            continue
        x1, y1, x2, y2 = map(int, t.to_tlbr())
        xa, ya = max(0, x1 - pad), max(0, y1 - pad)
        xb, yb = min(w, x2 + pad), min(h, y2 + pad)
        roi = frame[ya:yb, xa:xb]

        # validate ROI size
        rh, rw = roi.shape[:2]
        if rh <= 0 or rw <= 0:
            continue
        sw, sh = int(rw * downscale), int(rh * downscale)
        if sw < 2 or sh < 2:
            continue

        # downscale and detect
        small = cv2.resize(roi, (sw, sh))
        t1 = time.time()
        boxes, probs = detector.detect(small)
        logger.debug(f"Detection took {time.time() - t1:.3f}s on ROI {sw}x{sh}")
        if boxes is None:
            continue

        # rescale boxes and build UI entries
        for box, p in zip(boxes.astype(float) / downscale, probs):
            if p < conf_thresh:
                continue
            x1r, y1r, x2r, y2r = map(int, box)
            x1f, y1f = x1r + xa, y1r + ya
            x2f, y2f = x2r + xa, y2r + ya
            if x2f <= x1f or y2f <= y1f:
                continue
            face = frame[y1f:y2f, x1f:x2f]
            img_str = encode_to_base64(face)
            people_visible.append({"id": tid, "name": "Unknown", "face_image": img_str})
            unknown_crops.append((face, tid))

        