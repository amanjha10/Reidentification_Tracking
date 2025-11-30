#!/usr/bin/env python3
"""
Test script to verify dashboard template renders correctly
"""

from flask import Flask, render_template
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_active_cameras, get_tracking_stats

app = Flask(__name__)

@app.route('/test-dashboard')
def test_dashboard():
    # Mock data for testing
    try:
        cameras = get_active_cameras()
        stats = get_tracking_stats(24)
    except Exception as e:
        print(f"Database error: {e}")
        cameras = [
            {'id': 1, 'name': 'Test Camera 1', 'camera_type': 'IN', 'description': 'Test IN camera'},
            {'id': 2, 'name': 'Test Camera 2', 'camera_type': 'OUT', 'description': 'Test OUT camera'}
        ]
        stats = {'total_detections': 10, 'total_in': 5, 'total_out': 3}
    
    counts = {"IN": 5, "OUT": 3}
    
    try:
        return render_template('dashboard.html', 
                             username='test_user',
                             stats=stats,
                             cameras=cameras,
                             counts=counts)
    except Exception as e:
        return f"Template rendering error: {str(e)}"

if __name__ == '__main__':
    print("Testing dashboard template rendering...")
    print("Access at: http://localhost:5005/test-dashboard")
    app.run(host='0.0.0.0', port=5005, debug=True)
