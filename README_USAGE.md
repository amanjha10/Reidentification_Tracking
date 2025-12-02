# 🎥 AI-Powered Person Tracking System

A comprehensive Flask web application with YOLOv8 + DeepSORT tracking for counting people entering/exiting through camera feeds.

## ✨ Features

- **🔐 Web Authentication System** - Secure login with user management
- **📹 Dynamic Camera Management** - Add/edit/delete RTSP cameras via web interface
- **🤖 AI Person Tracking** - YOLOv8 detection + DeepSORT tracking
- **🔍 Person Re-identification** - Advanced ReID system to prevent double counting
- **📊 Real-time Dashboard** - Live counts, statistics, and camera surveillance
- **🎬 Live Video Streams** - View processed camera feeds with AI annotations
- **📱 Responsive Web UI** - Modern, mobile-friendly interface

## 🚀 Quick Start

### 1. Launch the System
```bash
python3 launcher.py
```

The launcher will automatically:
- ✅ Check if cameras are configured
- 🔧 Launch configuration interface if needed (port 5001)  
- 🎥 Launch full tracking system if cameras are ready (port 5003)

### 2. First Time Setup
If no cameras are configured, the launcher will start the configuration interface:

**🌐 Web Interface:** http://localhost:5001  
**🔑 Login Credentials:** admin / admin123

### 3. Configure Cameras
1. Login to the web interface
2. Go to **Camera Management**
3. Add at least one **IN** camera and one **OUT** camera
4. Fill in RTSP details:
   - **Name:** Descriptive name (e.g., "Front Door IN")
   - **IP Address:** Camera IP (e.g., 192.168.1.100)
   - **Port:** RTSP port (usually 554)
   - **Username/Password:** Camera credentials
   - **Stream Path:** RTSP path (e.g., /stream1)
   - **Camera Type:** IN (entry) or OUT (exit)

### 4. Run Tracking System
After cameras are configured, restart the launcher:
```bash
python3 launcher.py
```

The system will detect configured cameras and launch the full tracking application.

**🌐 Tracking Dashboard:** http://localhost:5003  
**🔑 Login Credentials:** admin / admin123

## 📋 System Requirements

### Required Cameras
- **Minimum:** 1 IN camera + 1 OUT camera
- **Camera Types:**
  - **IN:** Entry/entrance camera
  - **OUT:** Exit camera
- **Protocol:** RTSP streaming

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Main dependencies:
- Flask (web framework)
- OpenCV (computer vision)
- YOLOv8/Ultralytics (object detection)
- DeepSORT (tracking)
- SQLite (database)

## 🔧 Manual Operations

### Check System Status
```bash
python3 check_status.py
```

### Configuration Only (No Tracking)
```bash
python3 web_app_standalone.py
```
Access at: http://localhost:5001

### Full System (Requires Cameras)
```bash
python3 app.py
```
Access at: http://localhost:5003

## 📖 Usage Guide

### Dashboard Features
- **📊 Live Counts:** Real-time IN/OUT person counts
- **📈 Statistics:** Historical tracking data
- **🎥 Surveillance:** Live camera feeds with AI annotations
- **⚙️ Camera Management:** Add/edit/delete cameras

### Camera Management
- **Add Camera:** Configure new RTSP cameras
- **Edit Camera:** Modify existing camera settings
- **Test Camera:** Verify RTSP connection
- **Delete Camera:** Remove cameras from system

### AI Tracking Features
- **Person Detection:** YOLOv8-based human detection
- **Multi-Object Tracking:** DeepSORT tracking across frames
- **Re-identification:** Prevents double counting with AI embeddings
- **Direction Detection:** Automatic IN/OUT classification
- **Real-time Processing:** Live video processing and counting

## 🗂️ Database Schema

### Users Table
- User authentication and management
- Default admin user: admin/admin123

### Cameras Table
- RTSP camera configurations
- Dynamic URL generation
- Camera type classification (IN/OUT)

### Tracking Events (Optional)
- Historical tracking data
- Person count statistics
- Event logging

## 🔍 Troubleshooting

### Camera Connection Issues
1. Verify camera IP and credentials
2. Check RTSP stream path
3. Ensure camera is accessible on network
4. Test RTSP URL manually

### Web Interface Not Loading
1. Check if port is available (5001 or 5003)
2. Verify Flask application is running
3. Check firewall settings
4. Try accessing via IP instead of localhost

### AI Tracking Issues
1. Ensure sufficient GPU/CPU resources
2. Check camera stream quality
3. Verify YOLOv8 model file (yolov8n.pt)
4. Review error logs for details

## 📁 Project Structure

```
├── launcher.py              # Smart system launcher
├── app.py                   # Main tracking application  
├── web_app_standalone.py    # Configuration-only web app
├── database.py              # Database operations
├── person_reid.py           # Re-identification system
├── check_status.py          # System status checker
├── requirements.txt         # Python dependencies
├── templates/               # HTML templates
│   ├── login.html          # Login page
│   ├── dashboard.html      # Main dashboard
│   └── cameras.html        # Camera management
└── tracking_system.db      # SQLite database
```

## 🎯 Next Steps

1. **Configure Cameras:** Use web interface to add your RTSP cameras
2. **Test System:** Verify AI tracking works with your camera feeds  
3. **Customize Settings:** Adjust detection thresholds and tracking parameters
4. **Monitor Performance:** Check dashboard for real-time statistics
5. **Scale System:** Add more cameras as needed

---

**📞 Support:** Check logs and status with `python3 check_status.py`  
**🌐 Web Access:** Configuration (port 5001) | Tracking (port 5003)  
**🔑 Default Login:** admin / admin123
