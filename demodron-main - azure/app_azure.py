#!/usr/bin/env python3
"""
Unified Drone Detection Server
- Low-latency YOLO inference with queue-latest strategy
- Auto-tuning performance feedback
- Accurate geolocation (bearing-only, triangulation, optional size-based)
"""
import os
import argparse
import time
import threading
import sqlite3
import json
from datetime import datetime, timedelta
from collections import deque
import math

import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

# Try YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[ERROR] ultralytics not installed: pip install ultralytics")
    YOLO_AVAILABLE = False

# Try torch for device detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================================================================
# CONFIGURATION - All tuning parameters in one place
# ============================================================================
MODEL_PATH = os.environ.get("WEIGHTS", "best.pt")
CONF_THRESH = float(os.environ.get("CONF", "0.52"))
IOU_THRESH = float(os.environ.get("IOU", "0.52"))

# Auto-tuning targets
TARGET_LATENCY_MS = 150  # Target end-to-end latency
MAX_LATENCY_MS = 250     # Maximum acceptable latency
MIN_INFER_MS = 40        # If below this, can increase quality
MAX_INFER_MS = 120       # If above this, must decrease quality

# Resolution ladder (auto-tuning steps)
RESOLUTION_LADDER = [480, 640, 720, 960, 1280, 1600]
DEFAULT_WIDTH = 960
MIN_WIDTH = 480
MAX_WIDTH = 1280

# Frame rate control
DEFAULT_CADENCE_HZ = 20
MIN_CADENCE_HZ = 10
MAX_CADENCE_HZ = 30

# JPEG quality
DEFAULT_QUALITY = 0.70
MIN_QUALITY = 0.50
MAX_QUALITY = 0.85

# Triangulation parameters
TRIANGULATION_WINDOW_S = 15  # Time window for multi-observer
MIN_OBSERVER_DISTANCE_M = 10  # Minimum separation for valid triangulation
MAX_TRIANGULATION_ERROR_M = 100  # Reject solutions with high residual

# Database
DB_PATH = "detections.sqlite"

# ============================================================================
# DEVICE DETECTION
# ============================================================================
def pick_device():
    """Detect best available device"""
    if not TORCH_AVAILABLE:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        print(f"[WARN] Device detection error: {e}")
    return "cpu"

device = pick_device()

# ============================================================================
# MODEL LOADING
# ============================================================================
model = None
model_loaded = False
load_error = None

print(f"[INFO] Loading YOLO model: {MODEL_PATH}")
print(f"[INFO] Device: {device}")
print(f"[INFO] Thresholds - CONF: {CONF_THRESH}, IOU: {IOU_THRESH}")

if not YOLO_AVAILABLE:
    load_error = "ultralytics not installed"
    print(f"[ERROR] {load_error}")
elif not os.path.exists(MODEL_PATH):
    load_error = f"Model file not found: {MODEL_PATH}"
    print(f"[ERROR] {load_error}")
else:
    try:
        model = YOLO(MODEL_PATH)
        model_loaded = True
        print("[SUCCESS] YOLO model loaded")
        
        # Try optimization
        try:
            model.fuse()
            print("[INFO] Model layers fused")
        except:
            pass
        
        # Try half precision on CUDA
        if device == "cuda:0":
            try:
                # Test half precision
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = model.predict(test_img, half=True, verbose=False)
                print("[INFO] Half precision enabled")
            except:
                print("[INFO] Half precision not available")
        
        if hasattr(model, 'names'):
            print(f"[INFO] Model classes: {model.names}")
    except Exception as e:
        load_error = str(e)
        print(f"[ERROR] Failed to load model: {e}")

# ============================================================================
# DATABASE SETUP
# ============================================================================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            device_id TEXT NOT NULL,
            obs_lat REAL,
            obs_lon REAL,
            obs_acc_m REAL,
            bearing_deg REAL,
            class TEXT,
            conf REAL,
            x1n REAL,
            y1n REAL,
            x2n REAL,
            y2n REAL,
            hfov_deg REAL,
            vfov_deg REAL,
            heading_deg REAL,
            pitch_deg REAL,
            roll_deg REAL,
            track_id TEXT,
            mode TEXT,
            est_lat REAL,
            est_lon REAL,
            err_m REAL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON detections(ts_utc)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_device ON detections(device_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_track ON detections(track_id)")
    conn.commit()
    conn.close()
    print(f"[INFO] Database initialized: {DB_PATH}")

init_db()

# ============================================================================
# FLASK APP
# ============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = 'drone-detection-secret'
# Use threading mode instead of eventlet due to YOLO library compatibility
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", 
                    ping_timeout=60, ping_interval=25, max_http_buffer_size=10*1024*1024)

# ============================================================================
# STATS & PERFORMANCE TRACKING
# ============================================================================
class PerformanceTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.infer_times = deque(maxlen=20)  # Last 20 inference times
        self.frame_intervals = deque(maxlen=20)
        self.last_frame_time = None
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        
        # Current settings
        self.current_width = DEFAULT_WIDTH
        self.current_cadence = DEFAULT_CADENCE_HZ
        self.current_quality = DEFAULT_QUALITY
        
        # Last advice sent
        self.last_advice_time = 0
        self.advice_interval = 3.0  # Send advice every 3s max
    
    def record_inference(self, infer_ms):
        with self.lock:
            self.infer_times.append(infer_ms)
    
    def record_frame(self):
        with self.lock:
            self.frames_received += 1
            now = time.time()
            if self.last_frame_time:
                interval = now - self.last_frame_time
                self.frame_intervals.append(interval)
            self.last_frame_time = now
    
    def record_processed(self):
        with self.lock:
            self.frames_processed += 1
    
    def record_dropped(self):
        with self.lock:
            self.frames_dropped += 1
    
    def get_avg_infer_ms(self):
        with self.lock:
            if not self.infer_times:
                return 0
            return sum(self.infer_times) / len(self.infer_times)
    
    def get_current_fps(self):
        with self.lock:
            if not self.frame_intervals:
                return 0
            avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
            return 1.0 / avg_interval if avg_interval > 0 else 0
    
    def get_drop_rate(self):
        with self.lock:
            total = self.frames_received
            if total == 0:
                return 0
            return self.frames_dropped / total
    
    def should_send_advice(self):
        now = time.time()
        if now - self.last_advice_time >= self.advice_interval:
            self.last_advice_time = now
            return True
        return False
    
    def compute_advice(self):
        """Compute performance advice based on current metrics"""
        avg_infer = self.get_avg_infer_ms()
        drop_rate = self.get_drop_rate()
        
        new_width = self.current_width
        new_cadence = self.current_cadence
        new_quality = self.current_quality
        
        # If inference too slow, reduce quality
        if avg_infer > MAX_INFER_MS or drop_rate > 0.20:
            # Step down resolution
            idx = RESOLUTION_LADDER.index(self.current_width) if self.current_width in RESOLUTION_LADDER else 2
            if idx > 0:
                new_width = RESOLUTION_LADDER[idx - 1]
            # Reduce cadence
            new_cadence = max(MIN_CADENCE_HZ, self.current_cadence - 2)
            print(f"[PERF] Reducing quality - infer={avg_infer:.1f}ms, drop={drop_rate:.1%}")
        
        # If inference fast and no drops, increase quality
        elif avg_infer < MIN_INFER_MS and drop_rate < 0.05:
            # Step up resolution
            idx = RESOLUTION_LADDER.index(self.current_width) if self.current_width in RESOLUTION_LADDER else 2
            if idx < len(RESOLUTION_LADDER) - 1 and self.current_width < MAX_WIDTH:
                new_width = RESOLUTION_LADDER[idx + 1]
            # Increase cadence
            new_cadence = min(MAX_CADENCE_HZ, self.current_cadence + 2)
            print(f"[PERF] Increasing quality - infer={avg_infer:.1f}ms, drop={drop_rate:.1%}")
        
        # Clamp values
        new_width = max(MIN_WIDTH, min(MAX_WIDTH, new_width))
        new_cadence = max(MIN_CADENCE_HZ, min(MAX_CADENCE_HZ, new_cadence))
        
        # Update current settings
        self.current_width = new_width
        self.current_cadence = new_cadence
        self.current_quality = new_quality
        
        return {
            "sendWidth": new_width,
            "maxCadenceHz": new_cadence,
            "jpegQuality": new_quality
        }

perf_tracker = PerformanceTracker()

# ============================================================================
# QUEUE-LATEST INFERENCE
# ============================================================================
class InferenceQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.processing = False
    
    def push(self, frame_data):
        """Push frame, replacing any existing one (queue-latest)"""
        with self.lock:
            self.latest_frame = frame_data
            if self.latest_frame and not self.processing:
                return True  # Signal to process
            elif self.latest_frame and self.processing:
                perf_tracker.record_dropped()
        return False
    
    def pop(self):
        """Get latest frame and mark as processing"""
        with self.lock:
            if self.latest_frame and not self.processing:
                frame = self.latest_frame
                self.latest_frame = None
                self.processing = True
                return frame
        return None
    
    def done(self):
        """Mark processing complete"""
        with self.lock:
            self.processing = False

inference_queue = InferenceQueue()

def inference_worker():
    """Background worker that processes frames"""
    print("[INFO] Inference worker started")
    while True:
        frame_data = inference_queue.pop()
        if frame_data:
            try:
                result = process_frame(frame_data)
                perf_tracker.record_processed()
                # Emit result to client
                socketio.emit('predictions', result, room=frame_data.get('sid'))
            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                inference_queue.done()
        else:
            time.sleep(0.001)  # Small sleep to prevent busy-wait

# Start inference worker
threading.Thread(target=inference_worker, daemon=True).start()

def process_frame(frame_data):
    """Process a single frame with YOLO"""
    t0 = time.time()
    
    sid = frame_data.get('sid')
    w = frame_data.get('w', 0)
    h = frame_data.get('h', 0)
    jpg_bytes = frame_data.get('jpg')
    
    if not model_loaded:
        print("[ERROR] Model not loaded")
        return {"boxes": [], "error": "model_not_loaded", "infer_ms": 0}
    
    if not jpg_bytes:
        print("[ERROR] No JPEG data")
        return {"boxes": [], "error": "no_data", "infer_ms": 0}
    
    try:
        # Decode JPEG
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"[ERROR] cv2.imdecode failed, data length: {len(jpg_bytes)}")
            return {"boxes": [], "error": "decode_failed", "infer_ms": 0}
        
        h_img, w_img = img.shape[:2]
        print(f"[PROCESS] Decoded image: {w_img}x{h_img}")
        
        # Run YOLO
        t_infer = time.time()
        try:
            # Try half precision on CUDA
            if device == "cuda:0":
                results = model.predict(img, imgsz=max(w_img, h_img), conf=CONF_THRESH, 
                                       iou=IOU_THRESH, device=device, verbose=False, half=True)
            else:
                results = model.predict(img, imgsz=max(w_img, h_img), conf=CONF_THRESH, 
                                       iou=IOU_THRESH, device=device, verbose=False)
        except:
            # Fallback without half
            results = model.predict(img, imgsz=max(w_img, h_img), conf=CONF_THRESH, 
                                   iou=IOU_THRESH, device=device, verbose=False)
        
        infer_ms = int((time.time() - t_infer) * 1000)
        perf_tracker.record_inference(infer_ms)
        
        # Parse boxes and normalize
        boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy_all = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                print(f"[DETECT] Found {len(xyxy_all)} boxes")
                
                # Simple tracking: use box center as track ID (improve later)
                for idx, (xyxy, conf, cls) in enumerate(zip(xyxy_all, confs, classes)):
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    track_id = f"t{int(cx)}_{int(cy)}"
                    
                    class_name = model.names.get(int(cls), "drone") if hasattr(model, 'names') else "drone"
                    
                    box_data = {
                        "x1n": float(x1 / w_img),
                        "y1n": float(y1 / h_img),
                        "x2n": float(x2 / w_img),
                        "y2n": float(y2 / h_img),
                        "conf": float(conf),
                        "label": class_name,
                        "class_id": int(cls),
                        "track_id": track_id
                    }
                    boxes.append(box_data)
                    print(f"[BOX] {class_name} conf={conf:.2f} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
            else:
                print("[DETECT] No boxes in result")
        else:
            print("[DETECT] No results from model")
        
        total_ms = int((time.time() - t0) * 1000)
        
        return {
            "boxes": boxes,
            "infer_ms": infer_ms,
            "total_ms": total_ms,
            "src_wh": [w_img, h_img]
        }
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Frame processing: {e}")
        traceback.print_exc()
        return {"boxes": [], "error": str(e), "infer_ms": 0}

# ============================================================================
# GEOLOCATION FUNCTIONS
# ============================================================================
EARTH_RADIUS_M = 6371000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters between two WGS84 points"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c

def bearing_to(lat1, lon1, lat2, lon2):
    """Bearing in degrees from point 1 to point 2"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

def project_point(lat, lon, bearing_deg, distance_m):
    """Project a point along a bearing"""
    lat, lon, bearing = map(math.radians, [lat, lon, bearing_deg])
    lat2 = math.asin(math.sin(lat) * math.cos(distance_m / EARTH_RADIUS_M) +
                     math.cos(lat) * math.sin(distance_m / EARTH_RADIUS_M) * math.cos(bearing))
    lon2 = lon + math.atan2(math.sin(bearing) * math.sin(distance_m / EARTH_RADIUS_M) * math.cos(lat),
                            math.cos(distance_m / EARTH_RADIUS_M) - math.sin(lat) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def wgs84_to_enu(lat, lon, lat_ref, lon_ref):
    """Convert WGS84 to local ENU coordinates (meters)"""
    lat, lon, lat_ref, lon_ref = map(math.radians, [lat, lon, lat_ref, lon_ref])
    dlon = lon - lon_ref
    dlat = lat - lat_ref
    
    e = EARTH_RADIUS_M * dlon * math.cos(lat_ref)
    n = EARTH_RADIUS_M * dlat
    return e, n

def enu_to_wgs84(e, n, lat_ref, lon_ref):
    """Convert local ENU coordinates back to WGS84"""
    lat_ref, lon_ref = map(math.radians, [lat_ref, lon_ref])
    
    dlat = n / EARTH_RADIUS_M
    dlon = e / (EARTH_RADIUS_M * math.cos(lat_ref))
    
    lat = lat_ref + dlat
    lon = lon_ref + dlon
    return math.degrees(lat), math.degrees(lon)

def triangulate_observations(observations):
    """
    Triangulate target position from multiple bearing observations
    Returns: (est_lat, est_lon, err_m) or (None, None, None)
    """
    if len(observations) < 2:
        return None, None, None
    
    # Use first observation as reference
    ref = observations[0]
    lat_ref, lon_ref = ref['obs_lat'], ref['obs_lon']
    
    # Convert all observer positions to ENU
    points = []
    bearings = []
    for obs in observations:
        e, n = wgs84_to_enu(obs['obs_lat'], obs['obs_lon'], lat_ref, lon_ref)
        points.append((e, n))
        bearings.append(math.radians(obs['bearing_deg']))
    
    # Check observer separation
    min_sep = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            min_sep = min(min_sep, dist)
    
    if min_sep < MIN_OBSERVER_DISTANCE_M:
        return None, None, None  # Observers too close
    
    # Intersect rays using least squares
    # Each ray: p = obs + t * [cos(bearing), sin(bearing)]
    # Solve for intersection point that minimizes distance to all rays
    
    intersections = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            # Ray 1: p1 + t1 * d1
            p1 = np.array(points[i])
            d1 = np.array([math.cos(bearings[i]), math.sin(bearings[i])])
            
            # Ray 2: p2 + t2 * d2
            p2 = np.array(points[j])
            d2 = np.array([math.cos(bearings[j]), math.sin(bearings[j])])
            
            # Solve: p1 + t1*d1 = p2 + t2*d2
            # Rearrange: t1*d1 - t2*d2 = p2 - p1
            A = np.column_stack([d1, -d2])
            b = p2 - p1
            
            try:
                t = np.linalg.lstsq(A, b, rcond=None)[0]
                # Midpoint of closest approach
                point1 = p1 + t[0] * d1
                point2 = p2 + t[1] * d2
                intersection = (point1 + point2) / 2
                intersections.append(intersection)
            except:
                continue
    
    if not intersections:
        return None, None, None
    
    # Average all intersections
    avg_intersection = np.mean(intersections, axis=0)
    
    # Compute error (standard deviation)
    errors = [np.linalg.norm(pt - avg_intersection) for pt in intersections]
    err_m = np.std(errors) if len(errors) > 1 else errors[0] if errors else 0
    
    if err_m > MAX_TRIANGULATION_ERROR_M:
        return None, None, None  # Solution too uncertain
    
    # Convert back to WGS84
    est_lat, est_lon = enu_to_wgs84(avg_intersection[0], avg_intersection[1], lat_ref, lon_ref)
    
    return est_lat, est_lon, err_m

# ============================================================================
# SOCKET.IO HANDLERS
# ============================================================================
@socketio.on("connect")
def handle_connect():
    print(f"[SOCKET] Client connected: {request.sid}")
    emit("status", {
        "model_loaded": model_loaded,
        "device": device,
        "conf": CONF_THRESH,
        "iou": IOU_THRESH
    })
    
    # Send initial performance advice
    advice = perf_tracker.compute_advice()
    emit("perf_advice", advice)

@socketio.on("disconnect")
def handle_disconnect():
    print(f"[SOCKET] Client disconnected: {request.sid}")

@socketio.on("frame")
def handle_frame(data):
    """
    Receive frame for inference
    Expected: { w: int, h: int, jpg: ArrayBuffer or bytes }
    """
    perf_tracker.record_frame()
    
    # Extract binary data - Socket.IO with ArrayBuffer sends bytes directly
    jpg_bytes = None
    w, h = 0, 0
    
    if isinstance(data, dict):
        jpg_data = data.get('jpg')
        w = data.get('w', 0)
        h = data.get('h', 0)
        
        # Handle different data types
        if isinstance(jpg_data, bytes):
            jpg_bytes = jpg_data
        elif isinstance(jpg_data, (bytearray, memoryview)):
            jpg_bytes = bytes(jpg_data)
        else:
            print(f"[WARN] Unexpected jpg type: {type(jpg_data)}")
            emit("predictions", {"boxes": [], "error": "invalid_data_type"})
            return
    elif isinstance(data, bytes):
        jpg_bytes = data
    
    if not jpg_bytes:
        emit("predictions", {"boxes": [], "error": "no_data"})
        return
    
    print(f"[FRAME] Received {len(jpg_bytes)} bytes, size {w}x{h}")
    
    # Push to queue (queue-latest)
    frame_data = {
        'sid': request.sid,
        'w': w,
        'h': h,
        'jpg': jpg_bytes
    }
    
    should_process = inference_queue.push(frame_data)
    
    # Send performance advice periodically
    if perf_tracker.should_send_advice():
        advice = perf_tracker.compute_advice()
        emit("perf_advice", advice)

@socketio.on("observation")
def handle_observation(data):
    """
    Store observation with geolocation data
    Expected: { ts_utc, device_id, obs_lat, obs_lon, obs_acc_m, heading_deg,
                hfov_deg, vfov_deg, box:{x1n,y1n,x2n,y2n}, class, conf, track_id }
    """
    try:
        # Calculate bearing from box center
        box = data.get('box', {})
        cx = (box.get('x1n', 0) + box.get('x2n', 0)) / 2
        cy = (box.get('y1n', 0) + box.get('y2n', 0)) / 2
        
        hfov_deg = data.get('hfov_deg', 63)
        heading_deg = data.get('heading_deg', 0)
        
        # Bearing offset from camera center
        offset_norm = (cx - 0.5) / 0.5  # -1 to 1
        offset_rad = math.atan(offset_norm * math.tan(math.radians(hfov_deg / 2)))
        bearing_deg = (heading_deg + math.degrees(offset_rad)) % 360
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO detections 
            (ts_utc, device_id, obs_lat, obs_lon, obs_acc_m, bearing_deg,
             class, conf, x1n, y1n, x2n, y2n, hfov_deg, vfov_deg,
             heading_deg, pitch_deg, roll_deg, track_id, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'bearing')
        """, (
            data.get('ts_utc'),
            data.get('device_id'),
            data.get('obs_lat'),
            data.get('obs_lon'),
            data.get('obs_acc_m'),
            bearing_deg,
            data.get('class'),
            data.get('conf'),
            box.get('x1n'),
            box.get('y1n'),
            box.get('x2n'),
            box.get('y2n'),
            hfov_deg,
            data.get('vfov_deg'),
            heading_deg,
            data.get('pitch_deg'),
            data.get('roll_deg'),
            data.get('track_id'),
        ))
        conn.commit()
        conn.close()
        
        # Check for triangulation opportunities
        attempt_triangulation(data.get('track_id'))
        
        # Broadcast update
        socketio.emit('detections_update')
        
    except Exception as e:
        print(f"[ERROR] Storing observation: {e}")
        import traceback
        traceback.print_exc()

def attempt_triangulation(track_id):
    """Attempt to triangulate position for a track"""
    if not track_id:
        return
    
    try:
        # Get recent observations for this track
        cutoff = (datetime.utcnow() - timedelta(seconds=TRIANGULATION_WINDOW_S)).isoformat()
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT device_id, obs_lat, obs_lon, obs_acc_m, bearing_deg, ts_utc
            FROM detections
            WHERE track_id = ? AND ts_utc > ? AND obs_lat IS NOT NULL
            ORDER BY ts_utc DESC
        """, (track_id, cutoff))
        
        rows = c.fetchall()
        conn.close()
        
        if len(rows) < 2:
            return
        
        # Group by unique device IDs
        unique_devices = {}
        for row in rows:
            device_id = row[0]
            if device_id not in unique_devices:
                unique_devices[device_id] = {
                    'device_id': device_id,
                    'obs_lat': row[1],
                    'obs_lon': row[2],
                    'obs_acc_m': row[3],
                    'bearing_deg': row[4],
                    'ts_utc': row[5]
                }
        
        observations = list(unique_devices.values())
        
        if len(observations) < 2:
            return
        
        # Triangulate
        est_lat, est_lon, err_m = triangulate_observations(observations)
        
        if est_lat is not None:
            print(f"[TRIANGULATION] Track {track_id}: ({est_lat:.6f}, {est_lon:.6f}) ±{err_m:.1f}m")
            
            # Update recent observations with triangulated position
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                UPDATE detections
                SET mode = 'triangulated', est_lat = ?, est_lon = ?, err_m = ?
                WHERE track_id = ? AND ts_utc > ?
            """, (est_lat, est_lon, err_m, track_id, cutoff))
            conn.commit()
            conn.close()
            
            # Emit target update
            target = {
                "target_id": track_id,
                "mode": "triangulated",
                "lat": est_lat,
                "lon": est_lon,
                "bearing_deg": observations[0]['bearing_deg'],
                "confidence": 0.85,  # TODO: aggregate from detections
                "err_m": err_m,
                "timestamp_utc": datetime.utcnow().isoformat() + 'Z',
                "ttl_s": TRIANGULATION_WINDOW_S
            }
            socketio.emit('target_update', target)
    
    except Exception as e:
        print(f"[ERROR] Triangulation: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# REST API
# ============================================================================
@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")

@app.route("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "ok": model_loaded,
        "model": MODEL_PATH,
        "model_loaded": model_loaded,
        "load_error": load_error,
        "device": device,
        "conf": CONF_THRESH,
        "iou": IOU_THRESH,
        "perf": {
            "avg_infer_ms": perf_tracker.get_avg_infer_ms(),
            "current_fps": perf_tracker.get_current_fps(),
            "drop_rate": perf_tracker.get_drop_rate(),
            "frames_received": perf_tracker.frames_received,
            "frames_processed": perf_tracker.frames_processed,
            "frames_dropped": perf_tracker.frames_dropped
        }
    })

@app.route("/api/test_infer")
def test_infer():
    """Test inference with a dummy image"""
    if not model_loaded:
        return jsonify({"error": "model_not_loaded", "load_error": load_error}), 500
    
    try:
        # Create test image (640x480 random noise)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        t0 = time.time()
        results = model.predict(test_img, conf=CONF_THRESH, iou=IOU_THRESH, 
                               device=device, verbose=False)
        infer_ms = int((time.time() - t0) * 1000)
        
        num_boxes = 0
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                num_boxes = len(r.boxes)
        
        return jsonify({
            "ok": True,
            "infer_ms": infer_ms,
            "num_boxes": num_boxes,
            "test_image_size": "640x480",
            "device": device,
            "message": "Model inference working"
        })
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route("/api/detections")
def get_detections():
    """Get recent detections"""
    limit = request.args.get('limit', 200, type=int)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM detections
        ORDER BY ts_utc DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    
    detections = [dict(row) for row in rows]
    return jsonify({"detections": detections})

@app.route("/api/target/current")
def get_current_target():
    """Get current best target for UAV handoff"""
    try:
        # Get most recent triangulated detection
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT track_id, est_lat, est_lon, err_m, conf, bearing_deg, ts_utc
            FROM detections
            WHERE mode = 'triangulated' AND est_lat IS NOT NULL
            ORDER BY ts_utc DESC
            LIMIT 1
        """)
        row = c.fetchone()
        conn.close()
        
        if not row:
            return '', 204
        
        # Check if still fresh
        ts = datetime.fromisoformat(row['ts_utc'].replace('Z', ''))
        age_s = (datetime.utcnow() - ts).total_seconds()
        
        if age_s > TRIANGULATION_WINDOW_S:
            return '', 204
        
        target = {
            "target_id": row['track_id'],
            "mode": "triangulated",
            "lat": row['est_lat'],
            "lon": row['est_lon'],
            "bearing_deg": row['bearing_deg'],
            "confidence": row['conf'],
            "err_m": row['err_m'],
            "timestamp_utc": row['ts_utc'],
            "ttl_s": TRIANGULATION_WINDOW_S - int(age_s)
        }
        
        return jsonify(target)
    
    except Exception as e:
        print(f"[ERROR] Getting current target: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Drone Detection Server - Azure")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")  # Cambiado a 8000
    # ELIMINAMOS --cert y --key para Azure
    args = parser.parse_args()
    
    # En Azure, no usamos certificados locales
    print("[INFO] Azure Mode - Running without SSL certificates")
    print("[INFO] Azure will handle SSL termination at load balancer")
    
    # Start server SIN SSL
    print(f"[INFO] Starting Azure server on {args.host}:{args.port}")
    
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )

if __name__ == "__main__":
    main()
