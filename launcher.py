#!/usr/bin/env python3
"""
Smart Launcher for Camera Tracking System
This script checks if cameras are configured and launches the appropriate application:
- If cameras are configured: launches the full tracking system
- If no cameras: launches the web configuration interface
"""

import sys
import subprocess
import time
from database import get_active_cameras

def check_cameras():
    """Check if we have the minimum required cameras configured"""
    cameras = get_active_cameras()
    
    has_in = any(c['camera_type'] == 'IN' for c in cameras)
    has_out = any(c['camera_type'] == 'OUT' for c in cameras)
    
    return has_in and has_out, cameras

def launch_config_app():
    """Launch the standalone configuration web app"""
    print("🌐 Launching camera configuration interface...")
    print("📝 Configure your cameras at: http://localhost:5001")
    print("🔑 Login credentials: admin / admin123")
    print()
    
    try:
        subprocess.run([sys.executable, "web_app_standalone.py"])
    except KeyboardInterrupt:
        print("\n👋 Configuration interface stopped.")

def launch_tracking_app():
    """Launch the full tracking application"""
    print("🎥 Launching full tracking system...")
    print("🌐 Web interface will be available at: http://localhost:5003")
    print()
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Tracking system stopped.")

def main():
    print("🚀 Camera Tracking System Launcher")
    print("=" * 50)
    
    # Check camera configuration
    has_required_cameras, cameras = check_cameras()
    
    if has_required_cameras:
        print(f"✅ Found {len(cameras)} configured cameras")
        
        # List configured cameras
        for camera in cameras:
            print(f"   📹 {camera['name']} ({camera['camera_type']})")
        
        print("\n🎯 All required cameras found. Starting tracking system...")
        print("   Note: Make sure your cameras are accessible and streaming")
        print("   Press Ctrl+C to stop the system")
        print()
        
        time.sleep(2)
        launch_tracking_app()
        
    else:
        print("❌ Required cameras not configured")
        print("   You need at least one IN camera and one OUT camera")
        
        if cameras:
            print(f"\n📹 Currently configured ({len(cameras)} cameras):")
            for camera in cameras:
                print(f"   - {camera['name']} ({camera['camera_type']})")
        else:
            print("\n📝 No cameras configured yet")
        
        print("\n🔧 Launching camera configuration interface...")
        print("   Add your cameras and then restart this launcher")
        print("   Press Ctrl+C to stop the configuration interface")
        print()
        
        time.sleep(2)
        launch_config_app()

if __name__ == "__main__":
    main()
