import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
from datetime import datetime
import base64
import asyncio
import zipfile
from pathlib import Path
import tempfile
from typing import List, Callable
from fastapi import WebSocket

# Global variables
MODEL = None  # Initialize model as None

# Constants
CLASS_NAMES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", 
    "NO-Safety Vest", "Person", "Safety Cone", 
    "Safety Vest", "machinery", "vehicle"
]

def load_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = YOLO("ppe.pt")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return MODEL

async def process_video(
    input_path: str, 
    output_path: str, 
    csv_path: str,
    active_connections: List[WebSocket],
    broadcast_frame: Callable,
    broadcast_progress: Callable
):
    print("Starting video processing...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Timestamp', 'Class', 'Confidence', 'Bounding Box'])
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 7 == 0:  # Process every 7th frame
            try:
                results = MODEL(frame, conf=0.25)[0]
                
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = r
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cls = int(cls)
                    
                    color = (0, 255, 0) if "NO-" not in CLASS_NAMES[cls] else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{CLASS_NAMES[cls]} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            frame_count,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            CLASS_NAMES[cls],
                            f"{conf:.2f}",
                            f"({x1}, {y1}, {x2}, {y2})"
                        ])
                
                out.write(frame)
                
                # Convert frame to base64 for streaming
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Broadcast frame to all active connections
                for connection in active_connections:
                    await broadcast_frame(connection, frame_base64)
                
                # Update progress
                progress = min(99, int((frame_count / total_frames) * 100))
                for connection in active_connections:
                    await broadcast_progress(connection, progress)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        frame_count += 1
        await asyncio.sleep(0.01)  # Prevent blocking
    
    cap.release()
    out.release()
    print("Video processing completed")
    
    # Create zip file
    zip_path = os.path.join(tempfile.gettempdir(), "detection_results.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        if os.path.exists(output_path):
            zipf.write(output_path, "processed_video.mp4")
        if os.path.exists(csv_path):
            zipf.write(csv_path, "detections.csv")
    
    # Notify clients of completion
    for connection in active_connections:
        try:
            await connection.send_json({
                "type": "complete",
                "downloadUrl": f"/download/{os.path.basename(zip_path)}"
            })
        except Exception as e:
            print(f"Error sending completion message: {e}") 