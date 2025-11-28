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

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from flask import Flask, jsonify

# Import user ReID module
from person_reid import PersonReIDManager

# ------------------ CONFIG ------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "mps" if cv2.cuda.getCudaEnabledDeviceCount() == 0 and hasattr(cv2, 'ocl') else "cpu"

CONF_THRESHOLD = 0.25  # Lowered for better detection
IOU_THRESHOLD = 0.45
MAX_COSINE_DISTANCE = 0.2

CAMERA_OUT_RTSP = "rtsp://admin:14562%40@192.168.1.5:554/stream1"
CAMERA_IN_RTSP  = "rtsp://admin:14562%40@192.168.1.12:554/stream1"


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


# ------------------ MAIN ------------------
def main():
    global reid_manager
    logger.info("[INFO] Loading YOLO model")
    try:
        model = YOLO(MODEL_PATH)
        try: model.to(DEVICE); logger.info(f"[INFO] Model moved to device: {DEVICE}")
        except Exception: logger.warning("Could not move model to device")
    except Exception as e:
        logger.error(f"Model load error: {e}")
        model = YOLO("yolov8n.pt")

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
