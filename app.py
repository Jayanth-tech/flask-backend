from main import load_model, process_video
from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import asyncio
from typing import List
import os
import tempfile
import zipfile
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)

async def broadcast_frame(websocket: WebSocket, frame_data: str):
    try:
        await websocket.send_json({
            "type": "frame",
            "frame": frame_data
        })
    except Exception as e:
        print(f"Error broadcasting frame: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_progress(websocket: WebSocket, progress: int):
    try:
        await websocket.send_json({
            "type": "progress",
            "progress": progress
        })
    except Exception as e:
        print(f"Error broadcasting progress: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    from main import load_model, process_video  # Import here to avoid circular imports
    
    # Ensure model is loaded
    if load_model() is None:
        return {"status": "error", "message": "Model loading failed"}

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    output_path = os.path.join(temp_dir, "output.mp4")
    csv_path = os.path.join(temp_dir, "detections.csv")
    
    # Save uploaded video
    try:
        with open(input_path, "wb") as buffer:
            buffer.write(await video.read())
    except Exception as e:
        print(f"File save error: {e}")
        return {"status": "error", "message": f"Failed to save video: {str(e)}"}

    # Start processing in background
    asyncio.create_task(process_video(input_path, output_path, csv_path, active_connections, broadcast_frame, broadcast_progress))
    return {"status": "success", "message": "Processing started"}

@app.get("/download/{filename}")
async def download_results(filename: str):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    return FileResponse(
        file_path,
        media_type="application/zip",
        filename="detection_results.zip"
    )

if __name__ == "__main__":
    print("Starting server...")
    print("Loading YOLO model...")
    from main import load_model  # Import here to avoid circular imports
    load_model()  # Load model at startup
    uvicorn.run("app:app", host="localhost", port=8001, reload=True)
    # uvicorn.run("app:app")