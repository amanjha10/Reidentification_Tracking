# app.py
"""
Final production-ready IN/OUT counter
- YOLOv8 + DeepSORT
- Two RTSP cameras (IN & OUT)
- Unique ID tracking
- Counts only on valid line-direction crossing
- Shows overlays (IN, OUT counters)
- Flask API
"""

import threading
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, jsonify

# Import ReID module
from person_reid import PersonReIDManager


MODEL_PATH = "yolov8n.pt"     # ✅ MUST BE .pt file
DEVICE = "cpu"

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4
MAX_COSINE_DISTANCE = 0.2

# ✅ Update your RTSP streams here
CAMERA_OUT_RTSP = "rtsp://admin:14562%40@192.168.1.5:554/stream1"
CAMERA_IN_RTSP  = "rtsp://admin:14562%40@192.168.1.12:554/stream1"

DRAW = True

# line is horizontal center
LINE_REL = ((0.0, 0.5), (1.0, 0.5))

DEEP_SORT_CONFIG = {
    "max_age": 50,
    "n_init": 2,
    "max_cosine_distance": MAX_COSINE_DISTANCE,
}

# ReID Configuration
REID_CONFIG = {
    "similarity_threshold": 0.96,  # Optimal similarity threshold based on analysis (96% for good balance)
    "embedding_dim": 512,          # Embedding dimension
    "ttl_seconds": 300,            # 5 minutes TTL for embeddings
    "enable_reid": True,           # Enable/disable ReID system
    "reid_skip_frames": 10,        # Process ReID every N frames to reduce load
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("in_out_counter")


# -----------------------------------------------------
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


# -----------------------------------------------------
counts = {"IN": 0, "OUT": 0}
counts_lock = threading.Lock()

stats = {
    "camera_in": {"processed_frames": 0, "last_seen": None},
    "camera_out": {"processed_frames": 0, "last_seen": None},
}

# ✅ frame sharing
display_frames = {"IN": None, "OUT": None}
display_lock   = threading.Lock()

# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def xyxy_to_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# -----------------------------------------------------
class StreamProcessor(threading.Thread):
    def __init__(self, name, rtsp_url, logical_direction, model, device, count_physical_direction="BOTH"):
        super().__init__(daemon=True)
        self.name = name
        self.rtsp_url = rtsp_url
        self.logical_direction = logical_direction
        self.count_physical_direction = count_physical_direction
        self.model = model
        self.device = device

        self.vs = None
        self.running = threading.Event()
        self.running.set()

        self.tracker = DeepSort(
            max_age=DEEP_SORT_CONFIG["max_age"],
            n_init=DEEP_SORT_CONFIG["n_init"],
            max_cosine_distance=DEEP_SORT_CONFIG["max_cosine_distance"],
        )

        self.track_hist = defaultdict(lambda: deque(maxlen=16))
        
        # Watchdog variables
        self.last_frame_time = time.time()
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.frame_timeout = 5.0  # seconds
        
        # ReID tracking
        self.reid_manager = None
        self.known_persons = {}  # track_id -> person_id mapping
        self.reid_frame_counter = 0
        self.reid_skip_frames = REID_CONFIG["reid_skip_frames"]
        self.track_stability = {}  # track_id -> frame_count for stability checking

    def open_stream(self):
        """Open RTSP stream with retry logic"""
        while self.connection_attempts < self.max_connection_attempts and self.running.is_set():
            logger.info(f"[{self.name}] Opening stream (attempt {self.connection_attempts + 1}): {self.rtsp_url}")
            
            if self.vs:
                self.vs.release()
                
            self.vs = cv2.VideoCapture(self.rtsp_url)
            
            # Set buffer size to reduce latency
            self.vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.vs.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = self.vs.read()
                if ret and test_frame is not None:
                    # Log frame info for debugging
                    h, w = test_frame.shape[:2]
                    channels = test_frame.shape[2] if len(test_frame.shape) == 3 else 1
                    logger.info(f"[{self.name}] Stream opened successfully - Frame: {w}x{h}, Channels: {channels}")
                    
                    # Check if frame looks like it might be black/white or corrupted
                    mean_val = np.mean(test_frame)
                    if mean_val < 10:
                        logger.warning(f"[{self.name}] Warning: Frame appears very dark (mean={mean_val:.1f})")
                    elif channels == 1:
                        logger.warning(f"[{self.name}] Warning: Frame is grayscale")
                    
                    self.connection_attempts = 0  # Reset on success
                    self.last_frame_time = time.time()
                    return True
                else:
                    logger.warning(f"[{self.name}] Stream opened but cannot read frames")
            
            self.connection_attempts += 1
            logger.error(f"[{self.name}] Failed to open stream (attempt {self.connection_attempts})")
            
            if self.connection_attempts < self.max_connection_attempts:
                wait_time = min(5 * self.connection_attempts, 30)  # Exponential backoff, max 30s
                logger.info(f"[{self.name}] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        
        logger.error(f"[{self.name}] Failed to open stream after {self.max_connection_attempts} attempts")
        return False

    def restart_stream(self):
        """Restart the stream connection"""
        logger.warning(f"[{self.name}] Restarting stream...")
        if self.vs:
            self.vs.release()
            self.vs = None
        
        # Reset tracker to avoid stale tracks
        self.tracker = DeepSort(
            max_age=DEEP_SORT_CONFIG["max_age"],
            n_init=DEEP_SORT_CONFIG["n_init"],
            max_cosine_distance=DEEP_SORT_CONFIG["max_cosine_distance"],
        )
        self.track_hist.clear()
        
        return self.open_stream()

    def stop(self):
        self.running.clear()

    def get_line_points(self, frame_shape):
        h, w = frame_shape[:2]
        (rx1, ry1), (rx2, ry2) = LINE_REL
        return (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h))

    def crossed_line(self, p0, p1, a, b):
        """Check if line segment p0->p1 crosses line segment a->b"""
        def side(a, b, p):
            return (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0])
        
        side1 = side(a, b, p0)
        side2 = side(a, b, p1)
        
        # Check if points are on opposite sides of the line
        crossed = side1 * side2 < 0
        
        if crossed:
            logger.debug(f"Line crossed: {p0} -> {p1}, line: {a} -> {b}")
        
        return crossed

    def detect_persons(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=self.device, verbose=False)
        detections = []
        
        if len(results) == 0:
            return detections
            
        r = results[0]
        
        # Check if there are any boxes detected
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        for box in r.boxes:
            cls = int(box.cls.cpu())
            if cls != 0:  # ✅ ONLY PERSON (class 0 in COCO dataset)
                continue
            
            score = float(box.conf.cpu())
            if score < CONF_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = map(float, box.xyxy.cpu().numpy().flatten())
            
            # Ensure valid bounding box
            if x2 > x1 and y2 > y1:
                detections.append(Detection(x1, y1, x2, y2, score))

        return detections

    def run(self):
        if not self.open_stream():
            return

        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running.is_set():
            ok, frame = self.vs.read()
            if not ok:
                consecutive_failures += 1
                logger.warning(f"[{self.name}] Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"[{self.name}] Too many consecutive failures, restarting stream...")
                    if self.restart_stream():
                        consecutive_failures = 0
                        continue
                    else:
                        logger.error(f"[{self.name}] Failed to restart stream, stopping...")
                        break
                
                time.sleep(0.5)
                continue
            
            # Reset failure counter on successful frame read
            consecutive_failures = 0
            frame_count += 1
            self.last_frame_time = time.time()
            stats[self.name]["processed_frames"] += 1
            stats[self.name]["last_seen"] = self.last_frame_time
            
            # Watchdog check - restart if no frames for too long
            if time.time() - self.last_frame_time > self.frame_timeout:
                logger.warning(f"[{self.name}] Frame timeout detected, restarting stream...")
                if not self.restart_stream():
                    logger.error(f"[{self.name}] Failed to restart after timeout")
                    break
                continue

            # Get line points
            line_p1, line_p2 = self.get_line_points(frame.shape)

            # PERSON DETECTION
            detections = self.detect_persons(frame)
            
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"[{self.name}] Frame {frame_count}: {len(detections)} persons detected")
            
            # Prepare inputs for tracker - DeepSORT expects format: [[[x, y, w, h], confidence], ...]
            if detections:
                # Build detection data properly for DeepSORT
                raw_detections = []
                for d in detections:
                    # Convert from (x1,y1,x2,y2) to (x,y,w,h) format
                    bbox = [d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1]
                    raw_detections.append([bbox, d.score])
                
                # Update tracks
                tracks = self.tracker.update_tracks(raw_detections, frame=frame)
            else:
                # No detections - pass empty list
                tracks = self.tracker.update_tracks([], frame=frame)

            # DRAW line
            if DRAW:
                cv2.line(frame, line_p1, line_p2, (255, 255, 0), 2)

            for tr in tracks:
                if not tr.is_confirmed():
                    continue

                tid = tr.track_id
                bbox = tr.to_ltwh()
                l, t, w, h = map(int, bbox)
                
                # Calculate centroid
                cx = int(l + w / 2)
                cy = int(t + h / 2)
                curr_centroid = (cx, cy)

                # Get track history
                hist = self.track_hist[tid]
                
                # ReID Processing - OPTIMIZED: Process only once per stable track
                person_id = None
                is_known_person = False
                reid_similarity = 0.0
                
                # Track stability - increment frame count for this track
                if tid not in self.track_stability:
                    self.track_stability[tid] = 0
                self.track_stability[tid] += 1
                
                if REID_CONFIG["enable_reid"] and self.reid_manager:
                    # CRITICAL: Only process ReID ONCE when track becomes stable
                    track_stable = self.track_stability[tid] >= 8  # Require 8 stable frames (more stable)
                    need_reid_check = tid not in self.known_persons
                    
                    # Additional quality checks
                    bbox_area = w * h
                    bbox_too_small = bbox_area < 2000  # Skip very small detections
                    
                    if track_stable and need_reid_check and not bbox_too_small:
                        try:
                            # Convert bbox from ltwh to x1y1x2y2 format
                            x1, y1, x2, y2 = l, t, l + w, t + h
                            
                            # PROCESS REID ONLY ONCE PER TRACK
                            is_known_person, person_id, reid_similarity = self.reid_manager.store_or_match_embedding(
                                frame, [x1, y1, x2, y2], self.name, tid
                            )
                            
                            if person_id:
                                # STORE RESULT TO PREVENT REPROCESSING
                                self.known_persons[tid] = {
                                    'person_id': person_id,
                                    'is_known': is_known_person,
                                    'similarity': reid_similarity,
                                    'first_seen': time.time() if not is_known_person else None,
                                    'stable_frame': self.track_stability[tid],
                                    'processed': True  # Mark as processed
                                }
                                logger.info(f"[{self.name}] 🔍 ReID processed for track {tid}: {'🔴 Known' if is_known_person else '🟢 New'} person {person_id}")
                        except Exception as e:
                            logger.error(f"[{self.name}] ReID error for track {tid}: {e}")
                
                # Get person info if available (from cache to avoid reprocessing)
                if tid in self.known_persons:
                    person_info = self.known_persons[tid]
                    person_id = person_info['person_id']
                    is_known_person = person_info['is_known']
                    reid_similarity = person_info['similarity']
                
                # Draw track box and info
                if DRAW:
                    # Color coding: Green for new person, Red for known person
                    box_color = (0, 255, 0) if not is_known_person else (0, 0, 255)
                    
                    cv2.putText(frame, f"ID {tid}", (l, t - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    
                    if person_id:
                        # Show person ID and similarity
                        person_text = f"{person_id[:12]}"
                        if is_known_person:
                            person_text += f" ({reid_similarity:.2f})"
                        cv2.putText(frame, person_text, (l, t - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    
                    cv2.rectangle(frame, (l, t), (l + w, t + h), box_color, 2)
                    cv2.circle(frame, curr_centroid, 4, (255,0,255), -1)

                # Crossing detection - check if we have previous centroid
                if len(hist) > 0:
                    prev_centroid = hist[-1]
                    
                    # Check if line was crossed
                    if self.crossed_line(prev_centroid, curr_centroid, line_p1, line_p2):
                        dy = curr_centroid[1] - prev_centroid[1]
                        physical_dir = "DOWN" if dy > 0 else "UP"
                        
                        logger.info(f"[{self.name}] Person {tid} ({person_id}) moving {physical_dir}, logical={self.logical_direction}, physical_filter={self.count_physical_direction}")

                        # Check if this direction is allowed for counting
                        allowed = (self.count_physical_direction == "BOTH") or (
                            physical_dir == self.count_physical_direction)

                        if allowed:
                            # ReID Check: Only count if person is NOT known (prevents double counting)
                            should_count = True
                            if REID_CONFIG["enable_reid"] and is_known_person:
                                should_count = False
                                logger.info(f"[{self.name}] 🚫 Person {tid} ({person_id}) NOT counted - already seen (similarity: {reid_similarity:.3f})")
                            
                            if should_count:
                                with counts_lock:
                                    counts[self.logical_direction] += 1
                                    logger.info(f"[{self.name}] ✅ Person {tid} ({person_id}) crossed => {self.logical_direction}={counts[self.logical_direction]}")
                        else:
                            logger.info(f"[{self.name}] ❌ Person {tid} ({person_id}) ignored - wrong direction ({physical_dir} != {self.count_physical_direction})")

                # Update history
                hist.append(curr_centroid)

            # Increment ReID frame counter
            self.reid_frame_counter += 1

            # ✅ Store frame with better overlay
            if DRAW:
                with display_lock:
                    f = frame.copy()
                    
                    # Add camera name
                    cv2.putText(f, f"Camera: {self.name}", (20, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    
                    # Add counts
                    cv2.putText(f, f"IN: {counts['IN']}", (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(f, f"OUT: {counts['OUT']}", (20, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
                    # Add detection count
                    cv2.putText(f, f"Detections: {len(detections)}", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    
                    # Add tracking info
                    confirmed_tracks = len([t for t in tracks if t.is_confirmed()])
                    cv2.putText(f, f"Tracks: {confirmed_tracks}", (20, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    
                    # Add ReID info
                    if REID_CONFIG["enable_reid"] and self.reid_manager:
                        try:
                            reid_stats = self.reid_manager.get_stats()
                            unique_persons = reid_stats.get('unique_persons', 0)
                            reid_text = f"ReID: {unique_persons} unique persons"
                            cv2.putText(f, reid_text, (20, 155),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        except Exception as e:
                            cv2.putText(f, f"ReID: Error", (20, 155),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    
                    # Add legend for box colors
                    cv2.putText(f, "Green=New, Red=Known", (20, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    
                    display_frames[self.logical_direction] = f

        if self.vs:
            self.vs.release()


# -----------------------------------------------------
# FLASK
# -----------------------------------------------------
app = Flask(__name__)

# Global ReID manager instance
reid_manager = None

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
                "skip_frames": REID_CONFIG["reid_skip_frames"]
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
            "healthy": last and ((now - last) < 10.0)
        }
    
    # Add ReID health status
    reid_health = {
        "enabled": REID_CONFIG["enable_reid"],
        "manager_initialized": reid_manager is not None
    }
    
    if reid_manager:
        try:
            reid_stats = reid_manager.get_stats()
            reid_health.update({
                "total_persons": reid_stats.get("total_persons", 0),
                "active_persons": reid_stats.get("active_persons", 0),
                "database_healthy": True
            })
        except Exception as e:
            reid_health.update({
                "database_healthy": False,
                "error": str(e)
            })
    
    return jsonify({
        "service": "ok", 
        "cameras": result, 
        "reid": reid_health
    })


def start_flask():
    app.run(host="0.0.0.0", port=5003, debug=False, use_reloader=False)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    global reid_manager
    
    logger.info("[INFO] Loading YOLO...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        logger.info("Downloading fresh YOLOv8n model...")
        model = YOLO("yolov8n.pt")  # This will download if not exists

    # Initialize ReID Manager
    if REID_CONFIG["enable_reid"]:
        logger.info("[INFO] Initializing Person ReID Manager...")
        try:
            reid_manager = PersonReIDManager(
                embedding_dim=REID_CONFIG["embedding_dim"],
                similarity_threshold=REID_CONFIG["similarity_threshold"],
                ttl_seconds=REID_CONFIG["ttl_seconds"]
            )
            logger.info("[INFO] Person ReID Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ReID Manager: {e}")
            logger.warning("Continuing without ReID capabilities...")
            reid_manager = None
            REID_CONFIG["enable_reid"] = False
    else:
        logger.info("[INFO] Person ReID disabled in configuration")

    # Create stream processors with shared ReID manager
    cam_out = StreamProcessor("camera_out", CAMERA_OUT_RTSP, "OUT", model, DEVICE, "UP")
    cam_in = StreamProcessor("camera_in", CAMERA_IN_RTSP, "IN", model, DEVICE, "DOWN")
    
    # Share ReID manager with stream processors
    if reid_manager:
        cam_out.reid_manager = reid_manager
        cam_in.reid_manager = reid_manager
        logger.info("[INFO] ReID manager shared with both cameras")

    cam_out.start()
    cam_in.start()

    threading.Thread(target=start_flask, daemon=True).start()
    logger.info("[INFO] Flask server started on port 5003")

    # Display loop
    try:
        logger.info("[INFO] Starting display loop. Press 'q' to quit, 'r' to reset counters...")
        while True:
            with display_lock:
                f_in = display_frames["IN"]
                f_out = display_frames["OUT"]

            if f_in is not None:
                cv2.imshow("CAMERA_IN", f_in)

            if f_out is not None:
                cv2.imshow("CAMERA_OUT", f_out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit key pressed, stopping cameras...")
                cam_out.stop()
                cam_in.stop()
                break
            elif key == ord('r'):
                # Reset counts with 'r' key
                with counts_lock:
                    counts["IN"] = 0
                    counts["OUT"] = 0
                logger.info("Counts reset to zero via keyboard")

            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
        cam_out.stop()
        cam_in.stop()
    finally:
        # Cleanup ReID manager
        if reid_manager:
            try:
                logger.info("[INFO] Cleaning up ReID manager...")
                reid_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during ReID cleanup: {e}")

    cv2.destroyAllWindows()
    logger.info("[INFO] Application shutdown complete")


if __name__ == "__main__":
    main()
