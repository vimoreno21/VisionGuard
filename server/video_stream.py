import os
import cv2
import time
import threading
import logging
import subprocess
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rtsp_stream")

# Global variables for video streaming
latest_frame = None
lock = threading.Lock()
stream_active = False

# Configuration
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
REFRESH_INTERVAL = float(os.getenv("REFRESH_INTERVAL", "0.033"))  # ~30 FPS
DEFAULT_RESOLUTION = (640, 480)

def get_rtsp_url():
    """Get RTSP URL from environment variables"""
    username = os.getenv("CAMERA_USERNAME")
    password = os.getenv("CAMERA_PASSWORD")
    ip_address = os.getenv("IP_ADDRESS")
    port = os.getenv("CAMERA_PORT")
    path = os.getenv("RTSP_PATH")
    
    # Build complete URL
    url = f"rtsp://{username}:{password}@{ip_address}:{port}/{path}"
    
    # For logging - mask the password
    masked_url = url.replace(password, "****")
    
    return url, masked_url

def create_status_frame(message="Connecting to camera...", resolution=DEFAULT_RESOLUTION):
    """Create a status image with text"""
    width, height = resolution
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add the message
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img, message, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, timestamp, (10, height - 20), font, 0.5, (200, 200, 200), 1)
    
    return img

def capture_rtsp_stream():
    """Capture RTSP stream using OpenCV"""
    global latest_frame, stream_active
    
    url, masked_url = get_rtsp_url()
    logger.info(f"Starting RTSP stream capture from: {masked_url}")
    
    # Set RTSP over TCP (more reliable than UDP)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
    
    reconnect_delay = 5  # seconds between reconnection attempts
    
    while True:
        try:
            # Open connection to camera
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error("Failed to open RTSP stream")
                time.sleep(reconnect_delay)
                continue
            
            logger.info("Successfully connected to RTSP stream")
            stream_active = True
            
            # Main capture loop
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame, reconnecting...")
                    break
                
                # Update the frame with thread-safety
                with lock:
                    latest_frame = frame.copy()
                
                # Log stats occasionally
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Capturing at {fps:.1f} FPS (frame count: {frame_count})")
                
                # Small sleep to prevent maxing out CPU
                time.sleep(0.01)
            
            # If we get here, the stream has failed
            stream_active = False
            cap.release()
            logger.warning("Stream connection lost, reconnecting...")
            time.sleep(reconnect_delay)
            
        except Exception as e:
            logger.error(f"Error in RTSP stream capture: {str(e)}")
            stream_active = False
            time.sleep(reconnect_delay)

def generate_mjpeg():
    """Generate MJPEG stream from the latest frame"""
    global latest_frame
    
    logger.info("Client connected to video feed")
    frames_sent = 0
    start_time = time.time()
    
    while True:
        try:
            # Get latest frame with thread safety
            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            
            # If no frame available, create a status frame
            if frame is None:
                status_msg = "Connecting to camera..." if not stream_active else "Waiting for video..."
                frame = create_status_frame(status_msg)
            
            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            # Yield the frame
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            
            # Update stats
            frames_sent += 1
            if frames_sent % 100 == 0:
                elapsed = time.time() - start_time
                fps = frames_sent / elapsed
                logger.info(f"Streaming at {fps:.1f} FPS (sent {frames_sent} frames)")
            
            # Control frame rate
            time.sleep(REFRESH_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error generating MJPEG: {str(e)}")
            time.sleep(1)  # Wait before retrying

def add_video_routes(app: FastAPI):
    """Add video streaming routes to FastAPI app"""
    
    @app.get("/video_feed")
    async def video_feed():
        """Stream video as MJPEG"""
        logger.info("New client connected to /video_feed endpoint")
        return StreamingResponse(
            generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/stream_status")
    async def stream_status():
        """Get current stream status"""
        status = "active" if stream_active else "inactive"
        has_frame = latest_frame is not None
        
        return {
            "status": status,
            "active": stream_active,
            "has_frame": has_frame,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    @app.get("/stream")
    async def stream_page():
        """HTML page for streaming video"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RTSP Stream Viewer</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
                h1 { color: #444; }
                .video-container { margin: 20px auto; max-width: 800px; }
                img { max-width: 100%; border: 1px solid #ddd; }
                .status { margin-top: 10px; padding: 8px; border-radius: 4px; font-weight: bold; }
                .active { background-color: #d4edda; color: #155724; }
                .inactive { background-color: #f8d7da; color: #721c24; }
                button { margin-top: 10px; padding: 8px 16px; background-color: #2196F3; color: white; 
                        border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #0b7dda; }
            </style>
        </head>
        <body>
            <h1>Camera Stream</h1>
            <div class="video-container">
                <img src="/video_feed" id="stream" alt="Camera Stream">
                <div class="status" id="status">Connecting...</div>
                <button onclick="refreshStream()">Refresh Stream</button>
            </div>
            
            <script>
                // Function to refresh the stream
                function refreshStream() {
                    const img = document.getElementById('stream');
                    const timestamp = new Date().getTime();
                    img.src = `/video_feed?t=${timestamp}`;
                }
                
                // Function to check status
                function checkStatus() {
                    fetch('/stream_status')
                        .then(response => response.json())
                        .then(data => {
                            const statusElement = document.getElementById('status');
                            
                            if (data.active) {
                                statusElement.textContent = 'Stream Active';
                                statusElement.className = 'status active';
                            } else {
                                statusElement.textContent = 'Stream Inactive - Connecting...';
                                statusElement.className = 'status inactive';
                            }
                        })
                        .catch(error => {
                            document.getElementById('status').textContent = 'Error checking status';
                            document.getElementById('status').className = 'status inactive';
                        });
                }
                
                // Check status periodically
                checkStatus();
                setInterval(checkStatus, 5000);
                
                // Auto-refresh stream if issues occur
                document.getElementById('stream').onerror = function() {
                    setTimeout(refreshStream, 2000);
                };
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

def start_video_stream():
    """Initialize and start video streaming thread"""
    print("STARTING VIDEO STREAM THREAD - PRINT STATEMENT")
    logger.info("Starting video streaming thread")
    
    try:
        # Start the capture thread
        thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
        thread.start()
        
        logger.info("Video streaming thread started successfully")
        return [thread]
    except Exception as e:
        logger.error(f"Error starting video thread: {str(e)}")
        print(f"ERROR STARTING VIDEO THREAD: {str(e)}")
        return None
