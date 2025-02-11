from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import tempfile
import zipfile
import asyncio
from typing import List
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
app.static_folder = 'static'
active_connections = []

@socketio.on('connect')
def handle_connect():
    active_connections.append(request.sid)
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_connections:
        active_connections.remove(request.sid)
        print(f"Client disconnected: {request.sid}")

@socketio.on('message')
def handle_message(message):
    pass

async def broadcast_frame(sid, frame_data: str):
    try:
        socketio.emit('message', {
            "type": "frame",
            "frame": frame_data
        }, room=sid)
    except Exception as e:
        print(f"Error broadcasting frame: {e}")
        if sid in active_connections:
            active_connections.remove(sid)

async def broadcast_progress(sid, progress: int):
    try:
        socketio.emit('message', {
            "type": "progress",
            "progress": progress
        }, room=sid)
    except Exception as e:
        print(f"Error broadcasting progress: {e}")
        if sid in active_connections:
            active_connections.remove(sid)


def async_process_wrapper(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/upload', methods=['POST'])
def upload_video():
    from main import load_model, process_video
    
    if load_model() is None:
        return jsonify({"status": "error", "message": "Model loading failed"})

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"})

    video = request.files['video']
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    output_path = os.path.join(temp_dir, "output.mp4")
    csv_path = os.path.join(temp_dir, "detections.csv")

    try:
        video.save(input_path)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save video: {str(e)}"})

    from threading import Thread
    coro = process_video(
        input_path, 
        output_path, 
        csv_path, 
        active_connections,
        broadcast_frame,
        broadcast_progress
    )
    thread = Thread(target=lambda: async_process_wrapper(coro))
    thread.start()

    return jsonify({"status": "success", "message": "Processing started"})

@app.route('/download/{filename}')
def download_results(filename):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"})

    return send_file(
        file_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="detection_results.zip"
    )

if __name__ == "__main__":
    print("Starting server...")
    print("Loading YOLO model...")
    from main import load_model
    load_model()
    # socketio.run(app, host="127.0.0.1", port=8001, debug=True)
    app.run(debug=True)