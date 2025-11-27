#!/usr/bin/env python3
"""
Test camera management interface display
"""

import requests
from bs4 import BeautifulSoup

def test_camera_management():
    session = requests.Session()
    
    # Login
    login_data = {'username': 'admin', 'password': 'admin123'}
    session.post("http://localhost:5003/login", data=login_data)
    
    # Get cameras page
    response = session.get("http://localhost:5003/cameras")
    
    if response.status_code == 200:
        print(" Camera management page accessible")
        
        # Parse HTML to check for camera entries
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for camera table rows
        camera_rows = soup.find_all('tr')
        camera_count = len([row for row in camera_rows if any(cell.get_text().strip() in ['Camera IN', 'Camera OUT'] for cell in row.find_all('td'))])
        
        print(f" Camera table displays {camera_count} cameras")
        
        # Check for RTSP URLs in the page
        if "rtsp://admin:14562" in response.text:
            print(" RTSP URLs are displayed in the interface")
        else:
            print(" RTSP URLs not found in interface")
            
    else:
        print(f" Camera management page error: {response.status_code}")

if __name__ == "__main__":
    test_camera_management()
