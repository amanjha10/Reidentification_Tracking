#!/usr/bin/env python3
"""
Test script to verify dashboard access and camera management functionality
"""

import requests
import json

# Flask app URL
BASE_URL = "http://localhost:5003"

def test_login_and_cameras():
    """Test login and camera access"""
    
    # Create session to maintain cookies
    session = requests.Session()
    
    print("=== Testing Flask Dashboard Access ===\n")
    
    # Test 1: Check if app is running
    try:
        response = session.get(BASE_URL)
        print(f"✅ App responding: Status {response.status_code}")
    except Exception as e:
        print(f"❌ App not accessible: {e}")
        return
    
    # Test 2: Login
    login_data = {
        'username': 'admin',
        'password': 'admin123'
    }
    
    try:
        response = session.post(f"{BASE_URL}/login", data=login_data)
        if response.status_code == 302 or "dashboard" in response.url:
            print("✅ Login successful")
        else:
            print(f"❌ Login failed: Status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Login error: {e}")
        return
    
    # Test 3: Access dashboard
    try:
        response = session.get(f"{BASE_URL}/dashboard")
        if response.status_code == 200:
            print("✅ Dashboard accessible")
            if "AI Tracking System Dashboard" in response.text:
                print("✅ Dashboard content loaded correctly")
            else:
                print("❌ Dashboard content missing")
        else:
            print(f"❌ Dashboard not accessible: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    
    # Test 4: Get cameras via API
    try:
        response = session.get(f"{BASE_URL}/api/cameras")
        if response.status_code == 200:
            cameras = response.json()
            print(f"✅ Camera API accessible: Found {len(cameras)} cameras")
            
            for i, camera in enumerate(cameras, 1):
                print(f"   Camera {i}: {camera.get('name')} ({camera.get('camera_type')}) - {camera.get('ip_address')}")
        else:
            print(f"❌ Camera API error: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Camera API error: {e}")
    
    # Test 5: Get stats via API
    try:
        response = session.get(f"{BASE_URL}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats API accessible")
            current_counts = stats.get('current_counts', {})
            print(f"   Current counts: IN={current_counts.get('IN', 0)}, OUT={current_counts.get('OUT', 0)}")
        else:
            print(f"❌ Stats API error: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Stats API error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_login_and_cameras()
