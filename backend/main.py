import cv2
import uvicorn
import os
import sqlite3
import time
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse
from ultralytics import YOLO

app = FastAPI(title="VisionGuard AI API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_PATH = os.path.join(BASE_DIR, "frontend", "index.html")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "data", "events.db")

obj_model = YOLO(os.path.join(MODEL_DIR, "yolo26n.pt"))
pose_model = YOLO(os.path.join(MODEL_DIR, "yolo26n-pose.pt"))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
            event_type TEXT,
            label TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_event(event_type: str, label: str, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (event_type, label, confidence) VALUES (?, ?, ?)", 
        (event_type, label, confidence)
    )
    conn.commit()
    conn.close()

def infer_action(keypoints):
    try:
        l_sh, r_sh = keypoints[5], keypoints[6]
        l_hip, r_hip = keypoints[11], keypoints[12]
        l_ank, r_ank = keypoints[15], keypoints[16]
        
        avg_sh_y = (l_sh[1] + r_sh[1]) / 2
        avg_hip_y = (l_hip[1] + r_hip[1]) / 2
        avg_ank_y = (l_ank[1] + r_ank[1]) / 2
        
        torso = abs(avg_hip_y - avg_sh_y)
        legs = abs(avg_ank_y - avg_hip_y)
        
        return "Sitting" if legs < (torso * 0.8) else "Standing"
    except IndexError:
        return "Unknown"

def generate_frames(mode: str, source: str, conf: float):
    # Support webcam index integers (e.g. "0") or RTSP/HTTP URLs
    camera_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(camera_source)
    last_log_time = time.time()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        current_time = time.time()
        should_log = (current_time - last_log_time) > 2.0
            
        if mode == "detection":
            results = obj_model(frame, conf=conf)[0]
            frame = results.plot()
            
            if should_log and len(results.boxes) > 0:
                best_box = max(results.boxes, key=lambda x: x.conf[0])
                label = obj_model.names[int(best_box.cls[0])]
                score = round(float(best_box.conf[0]), 2)
                log_event("Object Detection", label, score)
                last_log_time = current_time
                
        else:
            results = pose_model(frame, conf=conf)[0]
            frame = results.plot()
            
            if results.keypoints is not None and len(results.keypoints) > 0:
                for kpt in results.keypoints.data.cpu().numpy():
                    action = infer_action(kpt)
                    cv2.putText(frame, f"Action: {action}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if should_log and action != "Unknown":
                        log_event("Action Engine", action, 0.95)
                        last_log_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
               
    cap.release()

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_PATH)

@app.get("/api/video_feed")
async def video_feed(
    mode: str = "detection",
    source: str = "0",
    conf: float = Query(0.5, ge=0.1, le=1.0)
):
    return StreamingResponse(
        generate_frames(mode, source, conf), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/events")
async def get_events():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, event_type, label, confidence FROM events ORDER BY id DESC LIMIT 10")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"events": rows}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)