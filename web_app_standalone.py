#!/usr/bin/env python3
"""
Standalone Flask Web Application for Camera Management
This app runs independently from the main tracking system and allows
users to configure cameras through the web interface.
"""

from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
from functools import wraps
import logging

# Import database functions
from database import (
    verify_user, get_active_cameras, get_camera_by_id, 
    create_camera, update_camera, delete_camera, build_rtsp_url,
    log_tracking_event, get_tracking_stats
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock counts for the standalone app (will be replaced by actual counts from main app)
counts = {"IN": 0, "OUT": 0}

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Authentication Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
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
    # Get tracking statistics
    try:
        stats = get_tracking_stats(24)  # Last 24 hours
    except:
        stats = {'total_detections': 0, 'total_in': 0, 'total_out': 0}
    
    # Get active cameras for surveillance section
    cameras = get_active_cameras()
    
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
    except Exception as e:
        flash(f'Error deleting camera: {str(e)}', 'error')
    
    return redirect(url_for('cameras'))

@app.route("/counts")
def get_counts():
    return jsonify(counts)

@app.route("/reset", methods=["POST"])
def reset_counts():
    global counts
    counts = {"IN": 0, "OUT": 0}
    logger.info("Counts reset to zero")
    return jsonify({"status": "reset", "counts": counts})

@app.route("/health")
def health():
    """Health check endpoint"""
    cameras = get_active_cameras()
    return jsonify({
        "service": "ok",
        "mode": "standalone_web_config",
        "cameras_configured": len(cameras),
        "cameras": [{"id": c["id"], "name": c["name"], "type": c["camera_type"]} for c in cameras]
    })

@app.route("/test_camera/<int:camera_id>")
@login_required
def test_camera(camera_id):
    """Test camera connection by building RTSP URL"""
    camera = get_camera_by_id(camera_id)
    if not camera:
        return jsonify({"error": "Camera not found"}), 404
    
    try:
        rtsp_url = build_rtsp_url(camera)
        return jsonify({
            "status": "ok",
            "camera": camera["name"],
            "rtsp_url": rtsp_url,
            "message": "RTSP URL generated successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting standalone Flask web application for camera configuration...")
    logger.info("This app allows you to configure cameras before running the main tracking system.")
    logger.info("Access the web interface at: http://localhost:5001")
    logger.info("Default credentials: admin / admin123")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
