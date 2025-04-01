import cv2
from deepface import DeepFace

# Open RTSP stream
cap = cv2.VideoCapture("rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/501")
if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Cannot grab frame")
    cap.release()
    exit()

cv2.imshow("Single Frame", frame)

# A short polling loop so Ctrl+C will work
while True:
    # If a key is pressed or window is closed, break
    if cv2.waitKey(50) != -1:
        break

cv2.destroyAllWindows()
cap.release()