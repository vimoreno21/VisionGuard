import os
os.environ["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

import torch
import cv2
from ultralytics import YOLO

# Initialize early to ensure libraries load properly
yolo_test = YOLO("yolov8n.pt")  # Use smaller model just for initialization
print("Libraries successfully pre-loaded")