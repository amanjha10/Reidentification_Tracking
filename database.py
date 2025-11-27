"""
Database initialization and management for the AI Tracking System
"""
import sqlite3
import hashlib
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = 'tracking_system.db'

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database with required tables"""
    with get_db_connection() as conn:
        # Create users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create cameras table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                ip_address TEXT NOT NULL,
                port INTEGER NOT NULL DEFAULT 554,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                stream_path TEXT NOT NULL DEFAULT '/stream1',
                camera_type TEXT NOT NULL CHECK(camera_type IN ('IN', 'OUT')),
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create tracking_logs table for counting data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tracking_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                direction TEXT NOT NULL CHECK(direction IN ('IN', 'OUT')),
                person_reid_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
        """)
        
        # Create system_settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("✅ Database initialized successfully")

def create_default_user():
    """Create default admin user if no users exist"""
    with get_db_connection() as conn:
        # Check if any users exist
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        
        if user_count == 0:
            # Create default admin user
            default_password = hash_password('admin123')
            conn.execute("""
                INSERT INTO users (username, password_hash, email, is_active)
                VALUES (?, ?, ?, ?)
            """, ('admin', default_password, 'admin@tracking.system', 1))
            conn.commit()
            print("✅ Default admin user created (username: admin, password: admin123)")
            print("⚠️  Please change the default password after first login!")

def create_default_cameras():
    """Create default camera configurations"""
    with get_db_connection() as conn:
        # Check if any cameras exist
        camera_count = conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        
        if camera_count == 0:
            # Create default cameras based on current RTSP URLs
            cameras = [
                {
                    'name': 'Camera OUT',
                    'description': 'Exit monitoring camera',
                    'ip_address': '192.168.1.5',
                    'port': 554,
                    'username': 'admin',
                    'password': '14562@',
                    'stream_path': '/stream1',
                    'camera_type': 'OUT',
                    'is_active': 1
                },
                {
                    'name': 'Camera IN',
                    'description': 'Entry monitoring camera', 
                    'ip_address': '192.168.1.12',
                    'port': 554,
                    'username': 'admin',
                    'password': '14562@',
                    'stream_path': '/stream1',
                    'camera_type': 'IN',
                    'is_active': 1
                }
            ]
            
            for camera in cameras:
                conn.execute("""
                    INSERT INTO cameras (name, description, ip_address, port, username, password, stream_path, camera_type, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    camera['name'], camera['description'], camera['ip_address'], 
                    camera['port'], camera['username'], camera['password'],
                    camera['stream_path'], camera['camera_type'], camera['is_active']
                ))
            
            conn.commit()
            print("✅ Default cameras created successfully")

def setup_database():
    """Complete database setup"""
    print("🔧 Setting up database...")
    init_database()
    create_default_user()
    create_default_cameras()
    print("✅ Database setup complete!")

# Database helper functions
def verify_user(username, password):
    """Verify user credentials"""
    with get_db_connection() as conn:
        user = conn.execute("""
            SELECT id, username, password_hash, is_active 
            FROM users 
            WHERE username = ? AND is_active = 1
        """, (username,)).fetchone()
        
        if user and user['password_hash'] == hash_password(password):
            # Update last login
            conn.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (user['id'],))
            conn.commit()
            return dict(user)
    return None

def get_active_cameras():
    """Get all active cameras"""
    with get_db_connection() as conn:
        cameras = conn.execute("""
            SELECT * FROM cameras 
            WHERE is_active = 1 
            ORDER BY camera_type, name
        """).fetchall()
        return [dict(camera) for camera in cameras]

def get_camera_by_id(camera_id):
    """Get camera by ID"""
    with get_db_connection() as conn:
        camera = conn.execute("""
            SELECT * FROM cameras WHERE id = ?
        """, (camera_id,)).fetchone()
        return dict(camera) if camera else None

def create_camera(camera_data):
    """Create new camera"""
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO cameras (name, description, ip_address, port, username, password, stream_path, camera_type, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            camera_data['name'], camera_data['description'], camera_data['ip_address'],
            camera_data['port'], camera_data['username'], camera_data['password'],
            camera_data['stream_path'], camera_data['camera_type'], camera_data['is_active']
        ))
        conn.commit()
        return cursor.lastrowid

def update_camera(camera_id, camera_data):
    """Update existing camera"""
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE cameras 
            SET name = ?, description = ?, ip_address = ?, port = ?, 
                username = ?, password = ?, stream_path = ?, camera_type = ?, 
                is_active = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            camera_data['name'], camera_data['description'], camera_data['ip_address'],
            camera_data['port'], camera_data['username'], camera_data['password'],
            camera_data['stream_path'], camera_data['camera_type'], camera_data['is_active'],
            camera_id
        ))
        conn.commit()

def delete_camera(camera_id):
    """Delete camera (soft delete by setting is_active = 0)"""
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE cameras SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (camera_id,))
        conn.commit()

def build_rtsp_url(camera):
    """Build RTSP URL from camera data"""
    # URL encode the password to handle special characters
    import urllib.parse
    encoded_password = urllib.parse.quote(camera['password'], safe='')
    
    return f"rtsp://{camera['username']}:{encoded_password}@{camera['ip_address']}:{camera['port']}{camera['stream_path']}"

def log_tracking_event(camera_id, direction, person_reid_id=None, confidence=None):
    """Log a tracking event"""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO tracking_logs (camera_id, direction, person_reid_id, confidence)
            VALUES (?, ?, ?, ?)
        """, (camera_id, direction, person_reid_id, confidence))
        conn.commit()

def get_tracking_stats(hours=24):
    """Get tracking statistics for the last N hours"""
    with get_db_connection() as conn:
        stats = conn.execute("""
            SELECT 
                direction,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM tracking_logs 
            WHERE timestamp >= datetime('now', '-{} hours')
            GROUP BY direction
        """.format(hours)).fetchall()
        
        return {stat['direction']: {'count': stat['count'], 'avg_confidence': stat['avg_confidence']} 
                for stat in stats}

if __name__ == '__main__':
    setup_database()
