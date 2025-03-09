import cv2
from deepface import DeepFace

cap = cv2.VideoCapture("rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/101")

if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed")
        break

    # Convert to RGB for DeepFace
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.analyze(rgb_frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        print(result)
    except Exception as e:
        print("DeepFace error:", e)

    cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
