import cv2

def annotate_and_write(frame, people_visible, tracked_objects, video_writer):
    """
    Draws names + boxes (with padding) for each visible person, then writes.
    """
    annotated = frame.copy()
    for person in people_visible:
        tid   = person["id"]
        name  = person["name"]
        # find that track’s box
        track = next((t for t in tracked_objects if t.track_id == tid), None)
        if not track: 
            continue
        x1,y1,x2,y2 = map(int, track.to_tlbr())

        # padding (tweak as needed)
        y1p = max(0, y1 - 50)
        y2p = min(frame.shape[0], y2 + 50)
        x1p = max(0, x1 - 50)
        x2p = min(frame.shape[1], x2 + 50)

        cv2.rectangle(annotated, (x1p, y1p), (x2p, y2p), (0,255,0), 2)
        cv2.putText(
            annotated, f"{name}", 
            (x1p, y1p - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2
        )
    video_writer.write(annotated)
