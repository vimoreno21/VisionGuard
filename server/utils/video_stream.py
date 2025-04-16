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
frame_last_accessed = time.time()  # Track when frame was last accessed

# Configuration
JPEG_QUALITY = int(os.environ.get('JPEG_QUALITY', 80))
REFRESH_INTERVAL = float(os.environ.get('REFRESH_INTERVAL', 0.033))  # ~30 FPS
DEFAULT_RESOLUTION = (640, 480)
MAX_FRAMES_BUFFER = 1

# New memory management settings
FRAME_CLEANUP_INTERVAL = 1.0  # Check and cleanup frames every second
FRAME_MAX_AGE = 2.0  # Clear frames if not accessed for 2 seconds

# Adaptive performance settings
ADAPTIVE_FPS = os.environ.get('ADAPTIVE_FPS', 'true').lower() == 'true'
MIN_REFRESH_INTERVAL = 0.033  # 30 FPS
MAX_REFRESH_INTERVAL = 0.1    # 10 FPS
MEMORY_THRESHOLD_MB = int(os.environ.get('MEMORY_THRESHOLD_MB', 1500))  # Throttle if memory exceeds this
CPU_THRESHOLD = float(os.environ.get('CPU_THRESHOLD', 80.0))  # Throttle if CPU exceeds this percentage

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

# Add this new function to manage memory
def get_system_metrics():
    """Get current system metrics for adaptive performance"""
    try:
        import psutil
        memory_usage_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return memory_usage_mb, cpu_percent
    except:
        return 0, 0

def adaptive_refresh_interval():
    """Dynamically adjust refresh interval based on system load"""
    if not ADAPTIVE_FPS:
        return REFRESH_INTERVAL
        
    memory_mb, cpu_percent = get_system_metrics()
    
    # Scale refresh interval based on memory and CPU usage
    memory_factor = min(1.0, max(0.0, (memory_mb - MEMORY_THRESHOLD_MB * 0.7) / (MEMORY_THRESHOLD_MB * 0.3)))
    cpu_factor = min(1.0, max(0.0, (cpu_percent - CPU_THRESHOLD * 0.7) / (CPU_THRESHOLD * 0.3)))
    
    # Use the higher of the two factors (more throttling)
    scale_factor = max(memory_factor, cpu_factor)
    
    # Scale the refresh interval between min and max
    interval = MIN_REFRESH_INTERVAL + scale_factor * (MAX_REFRESH_INTERVAL - MIN_REFRESH_INTERVAL)
    
    return interval

def cleanup_frames():
    """Periodically clean up frames that haven't been accessed recently"""
    global latest_frame, frame_last_accessed
    
    last_metrics_log = time.time()
    
    while True:
        try:
            current_time = time.time()
            
            # Check if frame is too old
            with lock:
                if latest_frame is not None and (current_time - frame_last_accessed > FRAME_MAX_AGE):
                    logger.debug("Clearing old frame from memory")
                    latest_frame = None
                    # Run garbage collection after clearing frame
                    gc.collect()
            
            # Log system metrics occasionally
            if current_time - last_metrics_log > 30:  # Every 30 seconds
                memory_mb, cpu_percent = get_system_metrics()
                logger.info(f"System metrics - Memory: {memory_mb:.1f} MB, CPU: {cpu_percent:.1f}%, " +
                           f"Current refresh interval: {adaptive_refresh_interval():.3f}s")
                last_metrics_log = current_time
            
            # Sleep for the cleanup interval
            time.sleep(FRAME_CLEANUP_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in frame cleanup: {str(e)}")
            time.sleep(FRAME_CLEANUP_INTERVAL)

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
                
                # Make sure to release the capture object
                cap.release()
                
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
                    # Get resolution from environment variables if available
                    target_width = int(os.environ.get('TARGET_WIDTH', DEFAULT_RESOLUTION[0]))
                    target_height = int(os.environ.get('TARGET_HEIGHT', DEFAULT_RESOLUTION[1]))
                    
                    # Resize frame if needed to match target resolution
                    if frame.shape[1] != target_width or frame.shape[0] != target_height:
                        frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Update the frame with thread-safety
                    with lock:
                        # Clear previous frame to help garbage collection
                        latest_frame = None
                        # Copy new frame
                        latest_frame = frame.copy()
                        # Update access time
                        global frame_last_accessed
                        frame_last_accessed = time.time()
                
                else:
                    logger.warning("Failed to read frame, reconnecting...")
                    break
                
                # Log stats occasionally
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Capturing at {fps:.1f} FPS (frame count: {frame_count})")
                
                # Run garbage collection periodically
                if frame_count % 300 == 0:
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
        finally:
            # Make sure we always release the capture object
            try:
                cap.release()
            except:
                pass

def start_ffmpeg_stream():
    """Start an FFMPEG process to convert RTSP to MJPEG stream as fallback"""
    global latest_frame, stream_active, frame_last_accessed
    
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
                    # Clear old frame first
                    latest_frame = None
                    latest_frame = frame.copy()
                    frame_last_accessed = time.time()
                    
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
    """Generate MJPEG stream from the latest frame with adaptive performance"""
    global latest_frame, frame_last_accessed
    
    logger.info("Client connected to video feed")
    frames_sent = 0
    start_time = time.time()
    last_gc_time = time.time()
    client_count = 1  # Start with the assumption that there's at least one client
    
    # Get quality settings from environment variables
    base_jpeg_quality = int(os.environ.get('JPEG_QUALITY', JPEG_QUALITY))
    
    while True:
        try:
            # Get dynamic refresh interval for adaptive performance
            current_refresh_interval = adaptive_refresh_interval()
            
            # Dynamically adjust JPEG quality based on system load
            memory_mb, cpu_percent = get_system_metrics()
            memory_pressure = memory_mb / MEMORY_THRESHOLD_MB
            
            # Reduce quality under memory pressure (min quality 40)
            jpeg_quality = max(40, int(base_jpeg_quality * (1.0 - min(0.5, memory_pressure))))
            
            # Get latest frame with thread safety
            current_frame = None
            with lock:
                if latest_frame is not None:
                    # Use a shallow copy when possible for better performance
                    current_frame = latest_frame.copy()
                    # Update access time
                    frame_last_accessed = time.time()
            
            # If no frame available, create a status frame
            if current_frame is None:
                status_msg = "Connecting to camera..." if not stream_active else "Waiting for video..."
                current_frame = create_status_frame(status_msg)
            
            # Encode as JPEG with dynamic quality control
            _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            
            # Explicitly delete the frame copy to help garbage collection
            current_frame = None
            
            # Yield the frame
            frame_data = jpeg.tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n'
            
            # Clear data after sending
            frame_data = None
            jpeg = None
            
            # Update stats
            frames_sent += 1
            current_time = time.time()
            
            # Log periodically
            if frames_sent % 300 == 0:  # Reduced frequency
                elapsed = current_time - start_time
                fps = frames_sent / elapsed
                logger.info(f"Streaming at {fps:.1f} FPS (sent {frames_sent} frames), " + 
                           f"Quality: {jpeg_quality}, Interval: {current_refresh_interval:.3f}s")
            
            # Run garbage collection periodically
            if current_time - last_gc_time > 10:  # Every 10 seconds
                gc.collect()
                last_gc_time = current_time
            
            # Control frame rate with adaptive interval
            time.sleep(current_refresh_interval)
            
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
        
    @app.post("/toggle_adaptive")
    async def toggle_adaptive():
        """Toggle adaptive FPS mode"""
        global ADAPTIVE_FPS
        ADAPTIVE_FPS = not ADAPTIVE_FPS
        logger.info(f"Adaptive FPS mode {'enabled' if ADAPTIVE_FPS else 'disabled'}")
        return {"adaptive_mode": ADAPTIVE_FPS}
    
    @app.get("/stream_status")
    async def stream_status():
        """Get current stream status with enhanced metrics"""
        status = "active" if stream_active else "inactive"
        has_frame = latest_frame is not None
        
        # Get detailed system metrics
        memory_mb, cpu_percent = get_system_metrics()
        
        # Get current frame dimensions if available
        frame_width = None
        frame_height = None
        if latest_frame is not None:
            with lock:
                if latest_frame is not None:
                    frame_height, frame_width = latest_frame.shape[:2]
        
        # Get current refresh interval (FPS)
        current_refresh_interval = adaptive_refresh_interval()
        approx_fps = 1.0 / current_refresh_interval if current_refresh_interval > 0 else 0
        
        # Get jpeg quality setting
        jpeg_quality = int(os.environ.get('JPEG_QUALITY', JPEG_QUALITY))
        
        return {
            "status": status,
            "active": stream_active,
            "has_frame": has_frame,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory_usage_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 1),
            "current_fps": round(approx_fps, 1),
            "jpeg_quality": jpeg_quality,
            "adaptive_mode": ADAPTIVE_FPS
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
                .memory-usage { margin-top: 10px; font-size: 0.9em; color: #666; }
                .metrics-container { margin-top: 20px; text-align: left; font-size: 0.9em; color: #555; 
                                    border: 1px solid #ddd; border-radius: 4px; padding: 10px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
                .metric { margin-bottom: 5px; }
                .metric-label { font-weight: bold; }
                .controls { margin-top: 15px; display: flex; gap: 10px; justify-content: center; }
            </style>
        </head>
        <body>
            <h1>Camera Stream</h1>
            <div class="video-container">
                <img src="/video_feed" id="stream" alt="Camera Stream">
                <div class="status" id="status">Connecting...</div>
                
                <div class="controls">
                    <button onclick="refreshStream()">Refresh Stream</button>
                    <button id="toggle-adaptive" onclick="toggleAdaptiveMode()">Toggle Adaptive Mode</button>
                </div>
                
                <div class="metrics-container">
                    <h3>Stream Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric">
                            <span class="metric-label">Status:</span> 
                            <span id="metric-status">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Memory Usage:</span> 
                            <span id="metric-memory">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">CPU Usage:</span> 
                            <span id="metric-cpu">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Resolution:</span> 
                            <span id="metric-resolution">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Current FPS:</span> 
                            <span id="metric-fps">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">JPEG Quality:</span> 
                            <span id="metric-quality">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Adaptive Mode:</span> 
                            <span id="metric-adaptive">Checking...</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Time:</span> 
                            <span id="metric-time">Checking...</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Function to refresh the stream
                function refreshStream() {
                    const img = document.getElementById('stream');
                    const timestamp = new Date().getTime();
                    img.src = `/video_feed?t=${timestamp}`;
                }
                
                // Function to toggle adaptive mode
                function toggleAdaptiveMode() {
                    fetch('/toggle_adaptive', { method: 'POST' })
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('metric-adaptive').textContent = 
                                data.adaptive_mode ? 'Enabled' : 'Disabled';
                            document.getElementById('toggle-adaptive').textContent = 
                                data.adaptive_mode ? 'Disable Adaptive Mode' : 'Enable Adaptive Mode';
                        });
                }
                
                // Function to check status
                function checkStatus() {
                    fetch('/stream_status')
                        .then(response => response.json())
                        .then(data => {
                            // Update status indicator
                            const statusElement = document.getElementById('status');
                            if (data.active) {
                                statusElement.textContent = 'Stream Active';
                                statusElement.className = 'status active';
                            } else {
                                statusElement.textContent = 'Stream Inactive - Connecting...';
                                statusElement.className = 'status inactive';
                            }
                            
                            // Update all metrics
                            document.getElementById('metric-status').textContent = 
                                data.active ? 'Active' : 'Inactive';
                            document.getElementById('metric-memory').textContent = 
                                `${data.memory_usage_mb} MB`;
                            document.getElementById('metric-cpu').textContent = 
                                `${data.cpu_percent}%`;
                            document.getElementById('metric-resolution').textContent = 
                                data.frame_width && data.frame_height ? 
                                `${data.frame_width} × ${data.frame_height}` : 'Unknown';
                            document.getElementById('metric-fps').textContent = 
                                `${data.current_fps} FPS`;
                            document.getElementById('metric-quality').textContent = 
                                `${data.jpeg_quality}`;
                            document.getElementById('metric-adaptive').textContent = 
                                data.adaptive_mode ? 'Enabled' : 'Disabled';
                            document.getElementById('metric-time').textContent = 
                                data.timestamp;
                            
                            // Update button text
                            document.getElementById('toggle-adaptive').textContent = 
                                data.adaptive_mode ? 'Disable Adaptive Mode' : 'Enable Adaptive Mode';
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
        # Try to install psutil for memory monitoring
        try:
            import psutil
        except ImportError:
            try:
                logger.info("Installing psutil for memory monitoring")
                subprocess.check_call(["pip", "install", "psutil"])
            except:
                logger.warning("Failed to install psutil - memory monitoring will be limited")
        
        # Log RTSP URL (with masked password)
        _, masked_url = get_rtsp_url()
        logger.info(f"RTSP URL: {masked_url}")
        
        # Log OpenCV version and capabilities
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        # Start frame cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_frames, daemon=True)
        cleanup_thread.start()
        logger.info("Started frame cleanup thread")
        
        # Start with OpenCV approach first
        opencv_thread = threading.Thread(target=capture_rtsp_stream, daemon=True)
        opencv_thread.start()

        # Determine if we're running on Render
        on_render = 'RENDER' in os.environ
        logger.info(f"Running on Render: {on_render}")

        # Also start FFMPEG fallback with a delay
        def delayed_ffmpeg():
            logger.info("Waiting 10 seconds before starting FFMPEG fallback...")
            time.sleep(10)
            # Only start FFMPEG if OpenCV is still not providing frames
            global latest_frame
            if latest_frame is None:
                ffmpeg_thread = threading.Thread(target=start_ffmpeg_stream, daemon=True)
                ffmpeg_thread.start()
                logger.info("Started FFMPEG fallback thread")
            else:
                logger.info("OpenCV stream is working, skipping FFMPEG fallback")
                
        threading.Thread(target=delayed_ffmpeg, daemon=True).start()
        
        logger.info("Video streaming thread started successfully")
        return [opencv_thread, cleanup_thread]
            
    except Exception as e:
        logger.error(f"Error starting video thread: {str(e)}")
        logger.exception("Detailed stack trace:")
        print(f"ERROR STARTING VIDEO THREAD: {str(e)}")
        return None