# app_optimized.py
"""
Optimized IN/OUT counter
- Improvements over original app.py:
  * Dedicated capture thread per RTSP with frame queue and drop-old policy (reduces RTSP buffering and lag)
  * Reduced detection frequency (skip frames) and drawing frequency to lower CPU/GPU load
  * Optional MPS device for Apple Silicon (falls back to 'cpu')
  * Lower input resolution and forced capture size (configurable) to speed up inference
  * Profile logging for per-stage timings (detect, track, reid, draw)
  * ReID: only compute embedding for stable tracks; uses skip-frame guard
  * Health/watchdog improvements and cleaner shutdown
"""

import threading
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
import queue
from functools import wraps

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash, Response
import os

# Import user ReID module and database
from person_reid import PersonReIDManager
from database import (
    verify_user, get_active_cameras, get_camera_by_id, 
    create_camera, update_camera, delete_camera, build_rtsp_url,
    log_tracking_event, get_tracking_stats
)

# ------------------ CONFIG ------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "mps" if cv2.cuda.getCudaEnabledDeviceCount() == 0 and hasattr(cv2, 'ocl') else "cpu"

CONF_THRESHOLD = 0.25  # Lowered for better detection
IOU_THRESHOLD = 0.45
MAX_COSINE_DISTANCE = 0.2

# Dynamic RTSP URLs - will be loaded from database
CAMERA_OUT_RTSP = None
CAMERA_IN_RTSP = None


DRAW = True
DRAW_EVERY_N_FRAMES = 2

LINE_REL = ((0.0, 0.5), (1.0, 0.5))

DEEP_SORT_CONFIG = {
    "max_age": 30,  # Reduced from 50 to remove stale tracks faster
    "n_init": 1,    # Minimum frames for confirmation
    "max_cosine_distance": 0.4,  # Increased from 0.2 for more lenient appearance matching
}

REID_CONFIG = {
    "similarity_threshold": 0.96,
    "embedding_dim": 512,
    "ttl_seconds": 300,
    "enable_reid": True,
    "reid_skip_frames": 10,
}

# Media and Snapshot Configuration
MEDIA_CONFIG = {
    "enable_snapshots": True,
    "media_directory": "./media/persons",
    "snapshot_quality": 95,  # JPEG quality
    "min_snapshot_size": (64, 128),  # Min width x height for snapshots
}

CAPTURE_QUEUE_SIZE = 1
DETECTION_SKIP_FRAMES = 1  # Reduced from 2 to ensure better tracking continuity
INPUT_MAX_WIDTH = 640
INPUT_MAX_HEIGHT = 360

# Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("in_out_counter_opt")

counts = {"IN": 0, "OUT": 0}
counts_lock = threading.Lock()
stats = {
    "camera_in": {"processed_frames": 0, "last_seen": None},
    "camera_out": {"processed_frames": 0, "last_seen": None},
}
display_frames = {"IN": None, "OUT": None}
display_lock = threading.Lock()


# ------------------ HELPERS ------------------
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def resize_keep_aspect(frame, max_w=INPUT_MAX_WIDTH, max_h=INPUT_MAX_HEIGHT):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return frame
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame, (nw, nh))


# ------------------ STREAM PROCESSOR ------------------
class StreamProcessor(threading.Thread):
    def __init__(self, name, rtsp_url, logical_direction, model, device, count_physical_direction="BOTH"):
        super().__init__(daemon=True)
        self.name = name
        self.rtsp_url = rtsp_url
        self.logical_direction = logical_direction
        self.count_physical_direction = count_physical_direction
        self.model = model
        self.device = device

        self.frame_queue = queue.Queue(maxsize=CAPTURE_QUEUE_SIZE)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

        self.running = threading.Event()
        self.running.set()

        # Tracker & ReID
        self.tracker = DeepSort(
            max_age=DEEP_SORT_CONFIG["max_age"],
            n_init=DEEP_SORT_CONFIG["n_init"],
            max_cosine_distance=DEEP_SORT_CONFIG["max_cosine_distance"],
        )
        self.track_hist = defaultdict(lambda: deque(maxlen=16))
        self.track_stability = {}
        self.reid_manager = None
        self.known_persons = {}
        self.reid_frame_counter = 0

        # Watchdog
        self.last_frame_time = time.time()
        self.frame_id = 0
        self.last_draw_frame = 0

    # ---------------- capture ----------------
    def start_capture(self):
        self.capture_thread.start()

    def _capture_loop(self):
        logger.info(f"[{self.name}] Starting capture thread")
        self.vs = None
        while self.running.is_set():
            if self.vs is None or not self.vs.isOpened():
                try:
                    self.vs = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    try: self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception: pass
                except Exception as e:
                    logger.error(f"[{self.name}] Capture open error: {e}")
                    time.sleep(3)
                    continue

            ok, frame = self.vs.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            self.last_frame_time = time.time()
            frame = resize_keep_aspect(frame)

            try:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        if self.vs:
            self.vs.release()
        logger.info(f"[{self.name}] Capture loop exiting")

    # ---------------- utils ----------------
    def get_line_points(self, frame_shape):
        h, w = frame_shape[:2]
        (rx1, ry1), (rx2, ry2) = LINE_REL
        return (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h))

    def crossed_line(self, p0, p1, a, b):
        def side(a, b, p):
            return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
        side1, side2 = side(a,b,p0), side(a,b,p1)
        return side1*side2 < 0

    # ---------------- detection ----------------
    def detect_persons(self, frame):
        try:
            results = self.model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=self.device, verbose=False)
        except Exception as e:
            logger.error(f"[{self.name}] Detection error: {e}")
            return []
        detections = []
        if len(results) == 0: return detections
        r = results[0]
        if r.boxes is None: return detections
        for box in r.boxes:
            cls = int(box.cls.cpu())
            if cls != 0: continue
            score = float(box.conf.cpu())
            if score < CONF_THRESHOLD: continue
            x1, y1, x2, y2 = map(float, box.xyxy.cpu().numpy().flatten())
            if x2>x1 and y2>y1: detections.append(Detection(x1, y1, x2, y2, score))
        return detections

    # ---------------- main run ----------------
    def run(self):
        logger.info(f"[{self.name}] StreamProcessor started")
        self.start_capture()
        last_profile_log = time.time()
        while self.running.is_set():
            try:
                try: frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - self.last_frame_time > 6.0:
                        if hasattr(self, 'vs') and self.vs: self.vs.release(); self.vs=None
                    continue

                self.frame_id += 1
                stats[self.name]["processed_frames"] += 1
                stats[self.name]["last_seen"] = time.time()
                line_p1, line_p2 = self.get_line_points(frame.shape)

                run_detector = (self.frame_id % max(1, DETECTION_SKIP_FRAMES) == 0)
                detections = self.detect_persons(frame) if run_detector else []

                # Log detection activity
                if len(detections) > 0:
                    logger.info(f"[{self.name}] 🔍 Detected {len(detections)} person(s)")

                raw_dets = []
                for d in detections:
                    # Validate bounding box before adding to tracker
                    bbox_width = d.x2 - d.x1
                    bbox_height = d.y2 - d.y1
                    
                    # Filter out invalid or too small bounding boxes
                    if bbox_width < 20 or bbox_height < 40 or d.score < 0.25:
                        logger.debug(f"[{self.name}] Filtering detection: w={bbox_width:.1f}, h={bbox_height:.1f}, score={d.score:.3f}")
                        continue
                    
                    bbox = [d.x1, d.y1, bbox_width, bbox_height]
                    raw_dets.append((bbox, d.score, 'person'))
                    
                    if len(raw_dets) <= 3:  # Debug first few detections
                        logger.debug(f"[{self.name}] Valid detection: x1={d.x1:.1f}, y1={d.y1:.1f}, w={bbox_width:.1f}, h={bbox_height:.1f}, score={d.score:.3f}")

                tracks = self.tracker.update_tracks(raw_dets if raw_dets else [], frame=frame)
                
                # Enhanced tracking activity logging
                confirmed_tracks = [t for t in tracks if t.is_confirmed()]
                tentative_tracks = [t for t in tracks if not t.is_confirmed()]
                all_tracks = len(tracks)
                
                if len(detections) > 0:  # Only log when we have detections
                    logger.debug(f"[{self.name}] 🎯 Tracks: {all_tracks} total, {len(confirmed_tracks)} confirmed, {len(tentative_tracks)} tentative")
                    
                    # Debug individual track states
                    for t in tracks:
                        track_state = "CONFIRMED" if t.is_confirmed() else "TENTATIVE"
                        track_age = getattr(t, 'time_since_update', 0)
                        logger.debug(f"[{self.name}] Track {t.track_id}: {track_state}, age={track_age}")
                
                if len(confirmed_tracks) > 0:
                    logger.info(f"[{self.name}] 📍 Tracking {len(confirmed_tracks)} person(s)")

                # ---------------- process tracks ----------------
                for tr in tracks:
                    if not tr.is_confirmed(): continue
                    tid = tr.track_id
                    l,t,w,h = map(int, tr.to_ltwh())
                    cx,cy = int(l+w/2), int(t+h/2)
                    curr_centroid = (cx,cy)
                    hist = self.track_hist[tid]
                    self.track_stability[tid] = self.track_stability.get(tid,0)+1

                    # ReID - FIXED LOGIC
                    if REID_CONFIG["enable_reid"] and self.reid_manager:
                        track_stable = self.track_stability.get(tid,0) >= 3  # Reduced to 3 frames for faster ReID
                        not_processed_yet = tid not in self.known_persons
                        bbox_area = w*h
                        
                        # Debug tracking requirements
                        if not_processed_yet:
                            logger.debug(f"[{self.name}] Track {tid}: stability={self.track_stability.get(tid,0)}/3, area={bbox_area}, stable={track_stable}")
                        
                        # Process ReID only ONCE per track when it becomes stable
                        if track_stable and not_processed_yet and bbox_area > 500:  # Further lowered for better detection
                            try:
                                is_known_person, person_id, reid_similarity = self.reid_manager.store_or_match_embedding(
                                    frame, [l,t,l+w,t+h], self.name, tid, MEDIA_CONFIG
                                )
                                if person_id:
                                    self.known_persons[tid] = {
                                        "person_id": person_id,
                                        "is_known": is_known_person,
                                        "similarity": reid_similarity,
                                        "first_seen": None if is_known_person else time.time(),
                                        "processed": True
                                    }
                                    status = " KNOWN" if is_known_person else " NEW"
                                    logger.info(f"[{self.name}] ReID: track {tid} -> {status} {person_id} (sim={reid_similarity:.3f})")
                            except Exception as e:
                                logger.error(f"[{self.name}] ReID error for track {tid}: {e}")
                    
                    # Get person status from cache
                    person_status = self.known_persons.get(tid, {})
                    is_known_person = person_status.get("is_known", False)

                    # Always draw bounding boxes and IDs on frame (not just when DRAW condition is met)
                    if DRAW:
                        box_color = (0,255,0) if not is_known_person else (0,0,255)
                        cv2.rectangle(frame,(l,t),(l+w,t+h),box_color,2)
                        cv2.circle(frame,curr_centroid,3,(255,0,255),-1)
                        cv2.putText(frame,f"ID {tid}",(l,t-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                        
                        # Show person ID and ReID info
                        person_info = self.known_persons.get(tid, {})
                        if person_info.get('person_id'):
                            person_text = f"{person_info['person_id'][:12]}"
                            if is_known_person:
                                person_text += f" ({person_info.get('similarity', 0):.2f})"
                            cv2.putText(frame, person_text, (l, t-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                    # Crossing Detection with ReID Check
                    if len(hist)>0:
                        prev_centroid = hist[-1]
                        if self.crossed_line(prev_centroid,curr_centroid,line_p1,line_p2):
                            dy = curr_centroid[1]-prev_centroid[1]
                            physical_dir = "DOWN" if dy>0 else "UP"
                            allowed = self.count_physical_direction=="BOTH" or physical_dir==self.count_physical_direction
                            
                            if allowed:
                                # ReID Check: Only count if person is NOT known (prevents double counting)
                                should_count = True
                                if REID_CONFIG["enable_reid"] and is_known_person:
                                    should_count = False
                                    logger.info(f"[{self.name}] 🚫 Person {tid} NOT counted - already seen")
                                
                                if should_count:
                                    with counts_lock:
                                        counts[self.logical_direction]+=1
                                        logger.info(f"[{self.name}] ✅ Person {tid} crossed => {self.logical_direction}={counts[self.logical_direction]}")
                                else:
                                    logger.info(f"[{self.name}] ❌ Person {tid} ignored - wrong direction ({physical_dir} != {self.count_physical_direction})")
                    hist.append(curr_centroid)

                # Draw overlay
                if DRAW and (self.frame_id-self.last_draw_frame)>=DRAW_EVERY_N_FRAMES:
                    with display_lock:
                        f = frame.copy()
                        cv2.line(f,line_p1,line_p2,(255,255,0),2)
                        cv2.putText(f,f"Camera: {self.name}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                        cv2.putText(f,f"IN: {counts['IN']}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                        cv2.putText(f,f"OUT: {counts['OUT']}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                        
                        # Add detection and tracking info
                        cv2.putText(f,f"Detections: {len(detections)}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                        confirmed_tracks = len([t for t in tracks if t.is_confirmed()])
                        cv2.putText(f,f"Tracks: {confirmed_tracks}",(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                        
                        # Add ReID info
                        if REID_CONFIG["enable_reid"] and self.reid_manager:
                            try:
                                reid_stats = self.reid_manager.get_stats()
                                unique_persons = reid_stats.get('unique_persons', 0)
                                reid_text = f"ReID: {unique_persons} unique persons"
                                cv2.putText(f, reid_text, (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                            except Exception as e:
                                cv2.putText(f, f"ReID: Error", (10, 150),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        
                        # Add legend for box colors
                        cv2.putText(f, "Green=New, Red=Known", (10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        
                        display_frames[self.logical_direction]=f
                    self.last_draw_frame=self.frame_id

            except Exception as e:
                logger.exception(f"[{self.name}] Processing loop error: {e}")

        logger.info(f"[{self.name}] StreamProcessor stopping")

    def stop(self):
        self.running.clear()
        try:
            if self.capture_thread.is_alive(): self.capture_thread.join(timeout=1.0)
        except Exception: pass


# ------------------ FLASK ------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# Flask session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'tracking_session:'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

reid_manager = None

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load dynamic RTSP URLs from database
def load_camera_rtsp_urls():
    global CAMERA_OUT_RTSP, CAMERA_IN_RTSP
    cameras = get_active_cameras()
    
    CAMERA_OUT_RTSP = None
    CAMERA_IN_RTSP = None
    
    for camera in cameras:
        rtsp_url = build_rtsp_url(camera)
        if camera['camera_type'] == 'OUT':
            CAMERA_OUT_RTSP = rtsp_url
        elif camera['camera_type'] == 'IN':
            CAMERA_IN_RTSP = rtsp_url
    
    logger.info(f"Loaded RTSP URLs - OUT: {CAMERA_OUT_RTSP}, IN: {CAMERA_IN_RTSP}")
    return CAMERA_OUT_RTSP, CAMERA_IN_RTSP

# Authentication Routes
@app.route('/')
def index():
    logger.info(f"Index route accessed. Session contents: {dict(session)}")
    if 'user_id' in session:
        logger.info(f"User {session.get('username')} already logged in, redirecting to dashboard")
        return redirect(url_for('dashboard'))
    logger.info("No user session found, redirecting to login")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = verify_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    logger.info(f"Dashboard accessed by user: {session.get('username')}")
    
    # Get tracking statistics
    try:
        stats = get_tracking_stats(24)  # Last 24 hours
    except Exception as e:
        logger.error(f"Error getting tracking stats: {e}")
        stats = {'total_detections': 0, 'total_in': 0, 'total_out': 0}
    
    # Get active cameras for surveillance section
    try:
        cameras = get_active_cameras()
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        cameras = []
    
    return render_template('dashboard.html', 
                         username=session.get('username'),
                         stats=stats,
                         cameras=cameras,
                         counts=counts)

@app.route('/cameras')
@login_required
def cameras():
    cameras_list = get_active_cameras()
    return render_template('cameras.html', cameras=cameras_list)

@app.route('/cameras/add', methods=['GET', 'POST'])
@login_required
def add_camera():
    if request.method == 'POST':
        camera_data = {
            'name': request.form['name'],
            'description': request.form.get('description', ''),
            'ip_address': request.form['ip_address'],
            'port': int(request.form['port']),
            'username': request.form['username'],
            'password': request.form['password'],
            'stream_path': request.form.get('stream_path', '/stream1'),
            'camera_type': request.form['camera_type'],
            'is_active': 1
        }
        
        try:
            camera_id = create_camera(camera_data)
            flash(f'Camera "{camera_data["name"]}" added successfully!', 'success')
            
            # Reload RTSP URLs
            load_camera_rtsp_urls()
            
            return redirect(url_for('cameras'))
        except Exception as e:
            flash(f'Error adding camera: {str(e)}', 'error')
    
    return render_template('cameras.html', show_add_form=True)

@app.route('/cameras/<int:camera_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_camera(camera_id):
    camera = get_camera_by_id(camera_id)
    if not camera:
        flash('Camera not found', 'error')
        return redirect(url_for('cameras'))
    
    if request.method == 'POST':
        camera_data = {
            'name': request.form['name'],
            'description': request.form.get('description', ''),
            'ip_address': request.form['ip_address'],
            'port': int(request.form['port']),
            'username': request.form['username'],
            'password': request.form['password'],
            'stream_path': request.form.get('stream_path', '/stream1'),
            'camera_type': request.form['camera_type'],
            'is_active': int(request.form.get('is_active', 1))
        }
        
        try:
            update_camera(camera_id, camera_data)
            flash(f'Camera "{camera_data["name"]}" updated successfully!', 'success')
            
            # Reload RTSP URLs
            load_camera_rtsp_urls()
            
            return redirect(url_for('cameras'))
        except Exception as e:
            flash(f'Error updating camera: {str(e)}', 'error')
    
    return render_template('cameras.html', camera=camera, show_edit_form=True)

@app.route('/cameras/<int:camera_id>/delete', methods=['POST'])
@login_required
def delete_camera_route(camera_id):
    try:
        delete_camera(camera_id)
        flash('Camera deleted successfully!', 'success')
        
        # Reload RTSP URLs
        load_camera_rtsp_urls()
        
    except Exception as e:
        flash(f'Error deleting camera: {str(e)}', 'error')
    
    return redirect(url_for('cameras'))

@app.route("/counts")
def get_counts():
    with counts_lock:
        return jsonify(counts)

@app.route("/reset", methods=["POST"])
def reset_counts():
    global counts
    with counts_lock:
        counts = {"IN": 0, "OUT": 0}
    logger.info("Counts reset to zero")
    return jsonify({"status": "reset", "counts": counts})

@app.route("/reid_stats")
def get_reid_stats():
    """Get ReID system statistics"""
    if not reid_manager:
        return jsonify({"error": "ReID manager not initialized"}), 500
    
    try:
        stats = reid_manager.get_stats()
        return jsonify({
            "status": "ok",
            "reid_enabled": REID_CONFIG["enable_reid"],
            "stats": stats,
            "config": {
                "similarity_threshold": REID_CONFIG["similarity_threshold"],
                "ttl_seconds": REID_CONFIG["ttl_seconds"],
                "embedding_dim": REID_CONFIG["embedding_dim"],
            }
        })
    except Exception as e:
        logger.error(f"Error getting ReID stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/reid_reset", methods=["POST"])
def reset_reid():
    """Reset ReID database and clear all stored embeddings"""
    if not reid_manager:
        return jsonify({"error": "ReID manager not initialized"}), 500
    
    try:
        reid_manager.reset_database()
        logger.info("ReID database reset successfully")
        return jsonify({"status": "reset", "message": "ReID database cleared"})
    except Exception as e:
        logger.error(f"Error resetting ReID database: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    now = time.time()
    result = {}
    for cam, s in stats.items():
        last = s["last_seen"]
        result[cam] = {
            "processed_frames": s["processed_frames"],
            "last_seen": last,
            "healthy": bool(last and ((now-last)<10.0))
        }
    reid_health = {
        "enabled": REID_CONFIG["enable_reid"],
        "manager_initialized": reid_manager is not None
    }
    if reid_manager:
        try: 
            rstats = reid_manager.get_stats()
            reid_health.update({
                "unique_persons": rstats.get("unique_persons", 0),
                "total_embeddings": rstats.get("total_embeddings", 0),
                "embeddings_per_person": rstats.get("embeddings_per_person", 0)
            })
        except Exception as e: 
            reid_health.update({"error": str(e)})
    return jsonify({"service":"ok","cameras":result,"reid":reid_health})

# Camera health check endpoint
@app.route('/camera/<int:camera_id>/health')
@login_required
def camera_health(camera_id):
    """Check if a specific camera is reachable"""
    camera = get_camera_by_id(camera_id)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404
    
    rtsp_url = build_rtsp_url(camera)
    
    try:
        # Try to open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            # Try to read one frame
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return jsonify({
                    "camera_id": camera_id,
                    "name": camera['name'], 
                    "status": "healthy",
                    "rtsp_url": rtsp_url,
                    "frame_size": f"{frame.shape[1]}x{frame.shape[0]}"
                })
            else:
                return jsonify({
                    "camera_id": camera_id,
                    "name": camera['name'],
                    "status": "connected_no_frame", 
                    "rtsp_url": rtsp_url,
                    "error": "Stream opened but no frame received"
                }), 503
        else:
            return jsonify({
                "camera_id": camera_id,
                "name": camera['name'],
                "status": "connection_failed",
                "rtsp_url": rtsp_url, 
                "error": "Could not open RTSP stream"
            }), 503
            
    except Exception as e:
        return jsonify({
            "camera_id": camera_id,
            "name": camera['name'],
            "status": "error",
            "rtsp_url": rtsp_url,
            "error": str(e)
        }), 500

# Streaming endpoints for camera feeds
@app.route('/stream/<int:camera_id>')
@login_required
def video_stream(camera_id):
    """Stream video feed with model inference output for specific camera"""
    def generate_frames():
        camera = get_camera_by_id(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return
        
        logger.info(f"Stream request for camera {camera_id}: {camera['name']}")
        
        # Build RTSP URL for direct streaming
        rtsp_url = build_rtsp_url(camera)
        camera_type = camera['camera_type']
        
        logger.info(f"Starting stream for camera {camera_id}: {camera['name']} ({rtsp_url})")
        
        # Try to get processed frame first (if AI is running)
        with display_lock:
            if camera_type == 'IN':
                ai_frame = display_frames.get("IN")
            elif camera_type == 'OUT':
                ai_frame = display_frames.get("OUT")
            else:
                ai_frame = None
        
        # If AI processing is available, use processed frames
        if ai_frame is not None:
            logger.info(f"Using AI-processed frames for camera {camera_id}")
            while True:
                with display_lock:
                    if camera_type == 'IN':
                        frame = display_frames.get("IN")
                    elif camera_type == 'OUT':
                        frame = display_frames.get("OUT")
                    else:
                        frame = None
                
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.1)
        else:
            # Fallback: Direct RTSP streaming (web-only mode)
            logger.info(f"Using direct RTSP streaming for camera {camera_id} (web-only mode)")
            cap = None
            try:
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open RTSP stream: {rtsp_url}")
                    return
                
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame from camera {camera_id}")
                        time.sleep(1)
                        continue
                    
                    frame_count += 1
                    
                    # Resize frame for better performance
                    height, width = frame.shape[:2]
                    if width > 640:
                        scale = 640 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Add overlay information
                    cv2.putText(frame, f"Camera: {camera['name']}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Type: {camera_type}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Status: Live Feed", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    time.sleep(0.05)  # ~20 FPS
                    
            except Exception as e:
                logger.error(f"Error streaming camera {camera_id}: {e}")
            finally:
                if cap:
                    cap.release()
                    logger.info(f"Released camera {camera_id} capture")
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/cameras')
@login_required
def api_cameras():
    """API endpoint to get cameras data"""
    cameras = get_active_cameras()
    # Add RTSP URL to each camera
    for camera in cameras:
        camera['rtsp_url'] = build_rtsp_url(camera)
    return jsonify(cameras)

@app.route('/api/stats')
@login_required
def api_stats():
    """API endpoint to get real-time statistics"""
    tracking_stats = get_tracking_stats(24)
    with counts_lock:
        current_counts = counts.copy()
    
    return jsonify({
        'current_counts': current_counts,
        'tracking_stats': tracking_stats,
        'timestamp': time.time()
    })

# ------------------ MAIN ------------------
def main():
    global reid_manager
    logger.info("[INFO] Loading YOLO model")
    try:
        # Set environment variable to avoid weights_only warning
        import os
        os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY_LOAD_WARNING'] = '1'
        
        # Import torch and set safe loading approach
        import torch
        
        # Temporary solution: monkey-patch torch.load to use weights_only=False
        original_torch_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = patched_load
        
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        
        # Restore original torch.load
        torch.load = original_torch_load
        logger.info("[INFO] YOLO model loaded successfully")
        
        try: 
            model.to(DEVICE) 
            logger.info(f"[INFO] Model moved to device: {DEVICE}")
        except Exception: 
            logger.warning("Could not move model to device, using CPU")
            DEVICE = "cpu"
            model.to(DEVICE)
    except Exception as e:
        logger.error(f"Model load error: {e}")
        try:
            # Fallback - create web-only mode without AI
            logger.warning("[WARNING] YOLO model failed to load. Starting in web-only mode.")
            logger.warning("AI tracking will be disabled, but camera management will work.")
            model = None
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return

    # Load dynamic RTSP URLs from database
    logger.info("[INFO] Loading camera configurations from database...")
    load_camera_rtsp_urls()
    
    # Check if we have required RTSP URLs
    if not CAMERA_OUT_RTSP or not CAMERA_IN_RTSP:
        logger.error("[ERROR] Missing camera configurations in database!")
        logger.error("Please add cameras through the web interface first.")
        logger.error("Run the web application and go to Camera Management to add cameras.")
        
        # Start Flask in web-only mode for camera configuration
        logger.info("[INFO] Starting Flask in configuration-only mode...")
        app.run(host='0.0.0.0', port=5003, debug=False)
        return

    # Only start AI components if model loaded successfully
    if model is not None:
        # ReID manager
        if REID_CONFIG["enable_reid"]:
            try:
                reid_manager = PersonReIDManager(
                    embedding_dim=REID_CONFIG["embedding_dim"],
                    similarity_threshold=REID_CONFIG["similarity_threshold"],
                    ttl_seconds=REID_CONFIG["ttl_seconds"],
                )
                logger.info("[INFO] ReID manager initialized")
            except Exception as e:
                logger.error(f"Failed to init ReID manager: {e}")
                reid_manager = None
                REID_CONFIG["enable_reid"]=False

        cam_out = StreamProcessor("camera_out", CAMERA_OUT_RTSP, "OUT", model, DEVICE, "UP")
        cam_in = StreamProcessor("camera_in", CAMERA_IN_RTSP, "IN", model, DEVICE, "DOWN")

        if reid_manager:
            cam_out.reid_manager = reid_manager
            cam_in.reid_manager = reid_manager

        cam_out.start()
        cam_in.start()
        
        logger.info("[INFO] AI tracking components started")

    # Start Flask web server
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5003, debug=False, use_reloader=False), daemon=True).start()
    logger.info("[INFO] Flask started on port 5003")

    try:
        while True:
            with display_lock:
                f_in = display_frames["IN"]
                f_out = display_frames["OUT"]
            if f_in is not None: cv2.imshow("CAMERA_IN", f_in)
            if f_out is not None: cv2.imshow("CAMERA_OUT", f_out)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'): cam_out.stop(); cam_in.stop(); break
            time.sleep(0.01)
    except KeyboardInterrupt:
        cam_out.stop(); cam_in.stop()
    finally:
        if reid_manager:
            try: reid_manager.cleanup()
            except Exception: pass
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == '__main__':
    main()
