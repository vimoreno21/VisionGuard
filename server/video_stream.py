import cv2
import threading
import time
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for video streaming
outputFrame = None
lock = threading.Lock()
stream_active = False

def stream_video():
    global outputFrame, lock, stream_active
    
    # Get camera details from environment
    CAMERA_USERNAME = os.getenv("CAMERA_USERNAME")
    CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD")
    IP_ADDRESS = os.getenv("IP_ADDRESS")
    PORT = os.getenv("PORT", "554")
    CAMERA_ID = os.getenv("CAMERA_ID", "1")
    
    rtsp_url = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{IP_ADDRESS}:{PORT}/Streaming/Channels/{CAMERA_ID}"
    
    logger.info("===== CAMERA CONNECTION ATTEMPT =====")
    logger.info(f"Connecting to: rtsp://username:password@{IP_ADDRESS}:{PORT}/Streaming/Channels/{CAMERA_ID}")
    
    # Try direct connection first
    logger.info("Attempting direct connection with OpenCV...")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logger.error("Direct connection failed!")
    else:
        logger.info("Direct connection successful!")
    
    # If still not open, exit
    if not cap.isOpened():
        logger.error("FATAL: Failed to connect to camera stream with all methods")
        stream_active = False
        
        # Try an alternative URL format (for some DVR systems)
        alt_rtsp_url = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{IP_ADDRESS}:{PORT}/cam/realmonitor?channel={CAMERA_ID}&subtype=0"
        logger.info(f"Trying alternative URL format: rtsp://username:password@{IP_ADDRESS}:{PORT}/cam/realmonitor?channel={CAMERA_ID}&subtype=0")
        
        cap = cv2.VideoCapture(alt_rtsp_url)
        if not cap.isOpened():
            logger.error("Alternative URL format also failed. Cannot connect to camera.")
            return
        else:
            logger.info("Connected with alternative URL format!")
    
    # Successfully connected!
    logger.info("Connected to camera stream!")
    stream_active = True
    
    # Test by reading the first frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Connected to stream but could not read first frame!")
        stream_active = False
        cap.release()
        return
    
    logger.info(f"Successfully read first frame! Size: {frame.shape}")
    
    # Update the frame
    with lock:
        outputFrame = frame.copy()
    
    # Continue with main capture loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            logger.warning("Failed to receive frame. Reconnecting...")
            stream_active = False
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        
        with lock:
            outputFrame = frame.copy()
            stream_active = True
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()
    stream_active = False

def generate_frames():
    global outputFrame, lock
    
    logger.info("Client connected to video feed")
    frames_sent = 0
    start_time = time.time()
    
    while True:
        with lock:
            if outputFrame is None:
                logger.warning("No frame available yet, waiting...")
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            if not flag:
                logger.warning("Failed to encode frame")
                continue
        
        # Log stats occasionally
        frames_sent += 1
        if frames_sent % 100 == 0:
            elapsed = time.time() - start_time
            fps = frames_sent / elapsed
            logger.info(f"Sent {frames_sent} frames to client at {fps:.2f} FPS")
        
        # Yield the frame
        yield b'--frame\r\n'
        yield b'Content-Type: image/jpeg\r\n\r\n'
        yield bytearray(encodedImage)
        yield b'\r\n'
        
        time.sleep(0.03)

def add_video_routes(app):
    @app.get("/video_feed")
    async def video_feed():
        logger.info("New client connected to /video_feed endpoint")
        return StreamingResponse(
            generate_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/stream_status")
    async def stream_status():
        status = "active" if stream_active else "inactive"
        logger.info(f"Stream status checked: {status}")
        return {"status": status}

def start_video_stream():
    logger.info("Starting video streaming thread")
    t = threading.Thread(target=stream_video)
    t.daemon = True
    t.start()
    return t