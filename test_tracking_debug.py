#!/usr/bin/env python3
"""
Tracking Debug Test - Simulates detections to test DeepSORT confirmation
This helps debug tracking issues without relying on RTSP cameras
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tracking_debug")

# DeepSORT config - same as your app
DEEP_SORT_CONFIG = {
    "max_age": 30,
    "n_init": 1,
    "max_cosine_distance": 0.4,
}

def create_synthetic_frame(width=640, height=480, person_boxes=None):
    """Create a synthetic frame with person boxes drawn"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    if person_boxes:
        for i, (x, y, w, h) in enumerate(person_boxes):
            # Draw a colored rectangle to simulate a person
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, -1)
            cv2.putText(frame, f"Person {i+1}", (int(x), int(y-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def test_tracking():
    """Test DeepSORT tracking with synthetic detections"""
    print("🧪 Starting DeepSORT Tracking Debug Test")
    print("=" * 50)
    
    # Initialize tracker
    tracker = DeepSort(**DEEP_SORT_CONFIG)
    
    # Test scenarios: moving persons
    test_scenarios = [
        # Frame 1: One person appears
        [(100, 100, 80, 160)],
        # Frame 2: Same person, slightly moved
        [(105, 105, 80, 160)],
        # Frame 3: Person moves more + new person appears
        [(110, 110, 80, 160), (300, 150, 70, 150)],
        # Frame 4: Both persons move
        [(115, 115, 80, 160), (305, 155, 70, 150)],
        # Frame 5: First person disappears, second remains
        [(310, 160, 70, 150)],
    ]
    
    for frame_idx, person_boxes in enumerate(test_scenarios):
        print(f"\n📸 Frame {frame_idx + 1}")
        print(f"   Detections: {len(person_boxes)} person(s)")
        
        # Create synthetic frame
        frame = create_synthetic_frame(person_boxes=person_boxes)
        
        # Convert to DeepSORT format: ([x, y, w, h], confidence, class)
        raw_dets = []
        for x, y, w, h in person_boxes:
            raw_dets.append(([x, y, w, h], 0.85, 'person'))
            print(f"   Detection: x={x}, y={y}, w={w}, h={h}, conf=0.85")
        
        # Update tracker
        tracks = tracker.update_tracks(raw_dets, frame=frame)
        
        # Analyze results
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]
        tentative_tracks = [t for t in tracks if not t.is_confirmed()]
        
        print(f"   Results: {len(tracks)} total tracks, {len(confirmed_tracks)} confirmed, {len(tentative_tracks)} tentative")
        
        for t in tracks:
            state = "✅ CONFIRMED" if t.is_confirmed() else "⏳ TENTATIVE"
            age = getattr(t, 'time_since_update', 'unknown')
            print(f"   Track {t.track_id}: {state} (age={age})")
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")
    
    # Final summary
    final_tracks = tracker.update_tracks([], frame=create_synthetic_frame())
    confirmed_final = [t for t in final_tracks if t.is_confirmed()]
    
    if len(confirmed_final) > 0:
        print("✅ SUCCESS: DeepSORT confirmed tracks successfully!")
        for t in confirmed_final:
            print(f"   Final confirmed track: {t.track_id}")
    else:
        print("❌ ISSUE: No tracks were confirmed during the test")
        print("   This suggests a DeepSORT configuration or detection format problem")

if __name__ == "__main__":
    test_tracking()
