import os
import cv2
import time
import threading
import logging
import subprocess
import numpy as np
import re
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rtsp_stream")

# Global variables for video streaming
latest_frame = None
lock = threading.Lock()
stream_active = False

# Configuration
JPEG_QUALITY = int(os.environ['JPEG_QUALITY']) if 'JPEG_QUALITY' in os.environ else 80
REFRESH_INTERVAL = float(os.environ['REFRESH_INTERVAL']) if 'REFRESH_INTERVAL' in os.environ else 0.033  # ~30 FPS
DEFAULT_RESOLUTION = (640, 480)
MAX_FRAMES_BUFFER = 1

def get_rtsp_url():
    """Get RTSP URL from environment variables"""
    username = os.environ['CAMERA_USERNAME']
    password = os.environ['CAMERA_PASSWORD']
    ip_address = os.environ['IP_ADDRESS']
    port = os.environ['CAMERA_PORT']
    path = os.environ['RTSP_PATH']
    
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
    """Capture RTSP stream using OpenCV with enhanced error handling and diagnostics"""
    global latest_frame, stream_active
    
    url, masked_url = get_rtsp_url()
    logger.info(f"Starting RTSP stream capture from: {masked_url}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"FFMPEG environment options: {os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'Not set')}")
    
    # Set RTSP over TCP (more reliable than UDP)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
    
    reconnect_delay = 5  # seconds between reconnection attempts
    connection_attempts = 0
    max_connection_attempts = 10  # Reset counter after this many attempts
    
    while True:
        try:
            connection_attempts += 1
            logger.info(f"Connection attempt {connection_attempts} to RTSP stream")
            
            # Try to ping the IP address first (if not running in a restricted environment)
            try:
                ip_address = os.environ['IP_ADDRESS']
                ping_result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", ip_address], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    timeout=3
                )
                if ping_result.returncode == 0:
                    logger.info(f"Successfully pinged camera IP: {ip_address}")
                else:
                    logger.warning(f"Failed to ping camera IP: {ip_address}")
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning(f"Ping test failed or not available: {str(e)}")
            
            # Open connection to camera with more detailed error handling
            logger.info("Attempting to open RTSP stream with cv2.VideoCapture")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Log capabilities
            backend = cap.getBackendName() if hasattr(cap, 'getBackendName') else "Unknown"
            logger.info(f"OpenCV backend being used: {backend}")
            
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                logger.error(f"Failed to open RTSP stream (attempt {connection_attempts})")
                
                # After several failed attempts, try different options
                if connection_attempts % 3 == 0:
                    logger.info("Trying alternative connection method")
                    # Try UDP instead of TCP
                    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp|stimeout;5000000'
                elif connection_attempts % 5 == 0:
                    logger.info("Trying with default options")
                    if 'OPENCV_FFMPEG_CAPTURE_OPTIONS' in os.environ:
                        del os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']
                
                # Reset counter occasionally
                if connection_attempts >= max_connection_attempts:
                    logger.warning(f"Reset after {max_connection_attempts} failed attempts")
                    connection_attempts = 0
                
                time.sleep(reconnect_delay)
                continue
            
            # Get camera information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Successfully connected to RTSP stream: Resolution {width}x{height}, FPS: {fps}")
            
            stream_active = True
            connection_attempts = 0  # Reset counter on success
            
            # Main capture loop
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Read frame
                ret, frame = cap.read()

                if ret:
                    # Resize frame to reduce memory usage (640x480 or similar)
                    frame = cv2.resize(frame, (640, 480))
                    # Update the frame with thread-safety
                    with lock:
                        latest_frame = frame.copy()
                
                else:
                    logger.warning("Failed to read frame, reconnecting...")
                    break
                
                # Log stats occasionally
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Capturing at {fps:.1f} FPS (frame count: {frame_count})")
                
                if frame_count % 30 == 0:
                    gc.collect()
                
                # Small sleep to prevent maxing out CPU
                time.sleep(0.01)
            
            # If we get here, the stream has failed
            stream_active = False
            cap.release()
            logger.warning("Stream connection lost, reconnecting...")
            time.sleep(reconnect_delay)
            
        except cv2.error as e:
            logger.error(f"OpenCV error in RTSP stream capture: {str(e)}")
            stream_active = False
            time.sleep(reconnect_delay)
        except Exception as e:
            logger.error(f"Error in RTSP stream capture: {str(e)}")
            logger.exception("Detailed traceback:")
            stream_active = False
            time.sleep(reconnect_delay)

def start_ffmpeg_stream():
    """Start an FFMPEG process to convert RTSP to MJPEG stream as fallback"""
    global latest_frame, stream_active
    
    url, masked_url = get_rtsp_url()
    logger.info(f"Starting FFMPEG fallback for RTSP stream: {masked_url}")
    
    try:
        # Construct the ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-loglevel', 'warning',
            '-rtsp_transport', 'tcp',          # Use TCP for RTSP
            '-i', url,                         # Input RTSP URL
            '-r', '15',                        # Output framerate
            '-q:v', '5',                       # Quality (1-31, 1 is best)
            '-f', 'image2pipe',                # Output to pipe as individual images
            '-pix_fmt', 'bgr24',               # Output pixel format
            '-vcodec', 'rawvideo',             # Output codec
            '-'                                # Output to stdout
        ]
        
        # Start the ffmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer
        )
        
        logger.info("FFMPEG process started successfully")
        stream_active = True
        
        # Get first frame to determine dimensions
        frame_size = None
        
        # Read frames from ffmpeg's stdout
        while True:
            if process.poll() is not None:
                logger.error(f"FFMPEG process exited with code {process.returncode}")
                stderr = process.stderr.read().decode('utf-8', errors='ignore')
                logger.error(f"FFMPEG stderr: {stderr}")
                break
                
            if frame_size is None:
                # We need to read one frame to get its dimensions
                # Try reading a small amount first to determine resolution
                header = process.stdout.read(1024)
                if not header:
                    logger.error("Failed to read header from FFMPEG")
                    break
                    
                # Try to detect resolution from the FFMPEG command output
                # This method is simplified and might need adjustment
                try:
                    # We might need to read more data to find resolution info
                    stderr_output = process.stderr.read(1024).decode('utf-8', errors='ignore')
                    resolution_match = re.search(r'(\d+)x(\d+)', stderr_output)
                    if resolution_match:
                        width = int(resolution_match.group(1))
                        height = int(resolution_match.group(2))
                        frame_size = width * height * 3  # BGR format (3 bytes per pixel)
                        logger.info(f"Detected frame resolution: {width}x{height}")
                    else:
                        # Default to a common resolution if detection fails
                        width, height = 640, 480
                        frame_size = width * height * 3
                        logger.warning(f"Using default resolution: {width}x{height}")
                except Exception as e:
                    logger.error(f"Error detecting resolution: {str(e)}")
                    width, height = 640, 480
                    frame_size = width * height * 3
            
            # Read the frame data
            raw_image = process.stdout.read(frame_size)
            if len(raw_image) < frame_size:
                logger.warning(f"Incomplete frame received: {len(raw_image)} bytes instead of {frame_size}")
                break
                
            # Convert to numpy array and reshape to frame dimensions
            try:
                frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))
                
                # Update the latest frame with thread safety
                with lock:
                    latest_frame = frame.copy()
                    
            except Exception as e:
                logger.error(f"Error processing frame from FFMPEG: {str(e)}")
                break
                
        # If we get here, the process has ended
        logger.warning("FFMPEG process ended, restarting...")
        stream_active = False
        
        # Try to clean up
        try:
            process.kill()
        except:
            pass
            
    except Exception as e:
        logger.error(f"Error in FFMPEG stream: {str(e)}")
        logger.exception("Detailed traceback:")
        stream_active = False
    
    # Sleep before restarting
    time.sleep(5)
    
    # Recursively restart
    start_ffmpeg_stream()

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
            else:
                # Ensure the frame is resized to save memory
                frame = cv2.resize(frame, (640, 480))
            
            # Encode as JPEG with higher compression (lower quality)
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
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
    """Initialize and start video streaming thread with enhanced logging"""
    print("STARTING VIDEO STREAM THREAD - PRINT STATEMENT")
    logger.info("Starting video streaming thread")
    
    try:
        # Log RTSP URL (with masked password)
        _, masked_url = get_rtsp_url()
        logger.info(f"RTSP URL: {masked_url}")
        
        # Log OpenCV version and capabilities
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        # Start with OpenCV approach first
        thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
        thread.start()

        # Determine if we're running on Render
        on_render = 'RENDER' in os.environ
        logger.info(f"Running on Render: {on_render}")

        # Start the appropriate thread based on environment
        if on_render:
            # On Render, try both methods for maximum chance of success
            threads = []
            
            # Start standard OpenCV capture
            opencv_thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
            opencv_thread.start()
            threads.append(opencv_thread)
            
            # Also start FFMPEG fallback on a short delay
            def delayed_ffmpeg():
                logger.info("Waiting 10 seconds before starting FFMPEG fallback...")
                time.sleep(10)
                ffmpeg_thread = threading.Thread(target=start_ffmpeg_stream, daemon=True)
                ffmpeg_thread.start()
                
            threading.Thread(target=delayed_ffmpeg, daemon=True).start()
            
            logger.info("Started multiple video streaming threads with fallback")
            return threads
        else:
            # When local, just use OpenCV as it works locally
            thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
            thread.start()
            logger.info("Video streaming thread started successfully")
            return [thread]
            
    except Exception as e:
        logger.error(f"Error starting video thread: {str(e)}")
        logger.exception("Detailed stack trace:")
        print(f"ERROR STARTING VIDEO THREAD: {str(e)}")
        return None
