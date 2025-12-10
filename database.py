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
        
        # Create report_summaries table for efficient reporting
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date DATE NOT NULL,
                report_type TEXT NOT NULL CHECK(report_type IN ('daily', 'weekly', 'monthly', 'quarterly', 'yearly')),
                total_users INTEGER DEFAULT 0,
                total_in INTEGER DEFAULT 0,
                total_out INTEGER DEFAULT 0,
                cameras_used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(report_date, report_type)
            )
        """)
        
        # Create generated_reports table to track Excel exports
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generated_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT NOT NULL,
                report_period TEXT NOT NULL,
                file_path TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generated_by TEXT
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

# ------------------ REPORTING FUNCTIONS ------------------

def populate_dummy_data():
    """Populate the system with dummy tracking data for testing"""
    import random
    from datetime import datetime, timedelta
    
    with get_db_connection() as conn:
        # Check if dummy data already exists
        existing_count = conn.execute("SELECT COUNT(*) FROM tracking_logs").fetchone()[0]
        if existing_count > 100:
            print("✅ Dummy data already exists, skipping population")
            return
            
        cameras = conn.execute("SELECT id FROM cameras WHERE is_active = 1").fetchall()
        if not cameras:
            print("❌ No active cameras found. Please add cameras first.")
            return
            
        camera_ids = [camera['id'] for camera in cameras]
        
        # Generate dummy data for the past 365 days
        start_date = datetime.now() - timedelta(days=365)
        
        print("🔄 Populating dummy tracking data...")
        
        for day_offset in range(365):
            current_date = start_date + timedelta(days=day_offset)
            
            # Generate random number of events per day (50-200)
            daily_events = random.randint(50, 200)
            
            for _ in range(daily_events):
                # Random timestamp during the day
                hour = random.randint(6, 22)  # Active hours 6 AM to 10 PM
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                
                event_time = current_date.replace(hour=hour, minute=minute, second=second)
                
                # Random data
                camera_id = random.choice(camera_ids)
                direction = random.choice(['IN', 'OUT'])
                confidence = random.uniform(0.7, 0.95)
                person_reid_id = f"person_{random.randint(1, 1000)}"
                
                conn.execute("""
                    INSERT INTO tracking_logs (camera_id, direction, person_reid_id, timestamp, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (camera_id, direction, person_reid_id, event_time, confidence))
        
        conn.commit()
        print(f"✅ Generated {365 * 125} dummy tracking events (average 125/day)")


def get_daily_report(date_str):
    """Get daily report data for specific date"""
    with get_db_connection() as conn:
        # Get tracking data for the specific day
        data = conn.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_users,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                COUNT(DISTINCT camera_id) as cameras_used
            FROM tracking_logs 
            WHERE DATE(timestamp) = ?
            GROUP BY DATE(timestamp)
        """, (date_str,)).fetchone()
        
        if data:
            return dict(data)
        else:
            return {
                'date': date_str,
                'total_users': 0,
                'total_in': 0,
                'total_out': 0,
                'cameras_used': 0
            }


def get_monthly_report(year, month):
    """Get monthly report data"""
    with get_db_connection() as conn:
        # Get daily breakdown for the month
        daily_data = conn.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_users,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                COUNT(DISTINCT camera_id) as cameras_used
            FROM tracking_logs 
            WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp)
        """, (str(year), str(month).zfill(2))).fetchall()
        
        # Calculate monthly summary
        if daily_data:
            total_users = sum(row['total_users'] for row in daily_data)
            total_in = sum(row['total_in'] for row in daily_data)
            total_out = sum(row['total_out'] for row in daily_data)
            cameras_used = len(set(conn.execute("""
                SELECT DISTINCT camera_id FROM tracking_logs 
                WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
            """, (str(year), str(month).zfill(2))).fetchall()))
            
            return {
                'period': f"{year}-{str(month).zfill(2)}",
                'daily_data': [dict(row) for row in daily_data],
                'summary': {
                    'total_users': total_users,
                    'total_in': total_in,
                    'total_out': total_out,
                    'cameras_used': cameras_used
                }
            }
        else:
            return {
                'period': f"{year}-{str(month).zfill(2)}",
                'daily_data': [],
                'summary': {
                    'total_users': 0,
                    'total_in': 0,
                    'total_out': 0,
                    'cameras_used': 0
                }
            }


def get_quarterly_report(year, quarter):
    """Get quarterly report data"""
    # Determine months for the quarter
    quarter_months = {
        1: [1, 2, 3],
        2: [4, 5, 6], 
        3: [7, 8, 9],
        4: [10, 11, 12]
    }
    
    months = quarter_months.get(quarter, [1, 2, 3])
    
    with get_db_connection() as conn:
        # Get monthly breakdown for the quarter
        monthly_data = []
        total_users = total_in = total_out = 0
        all_cameras = set()
        
        for month in months:
            month_data = conn.execute("""
                SELECT 
                    strftime('%Y-%m', timestamp) as month,
                    COUNT(*) as total_users,
                    SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                    SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                    COUNT(DISTINCT camera_id) as cameras_used
                FROM tracking_logs 
                WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
                GROUP BY strftime('%Y-%m', timestamp)
            """, (str(year), str(month).zfill(2))).fetchone()
            
            if month_data:
                monthly_data.append(dict(month_data))
                total_users += month_data['total_users']
                total_in += month_data['total_in']
                total_out += month_data['total_out']
                
                # Get unique cameras for this month
                month_cameras = conn.execute("""
                    SELECT DISTINCT camera_id FROM tracking_logs 
                    WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
                """, (str(year), str(month).zfill(2))).fetchall()
                all_cameras.update(row['camera_id'] for row in month_cameras)
            else:
                monthly_data.append({
                    'month': f"{year}-{str(month).zfill(2)}",
                    'total_users': 0,
                    'total_in': 0,
                    'total_out': 0,
                    'cameras_used': 0
                })
        
        return {
            'period': f"Q{quarter} {year}",
            'monthly_data': monthly_data,
            'summary': {
                'total_users': total_users,
                'total_in': total_in,
                'total_out': total_out,
                'cameras_used': len(all_cameras)
            }
        }


def get_yearly_report(year):
    """Get yearly report data"""
    with get_db_connection() as conn:
        # Get monthly breakdown for the year
        monthly_data = conn.execute("""
            SELECT 
                strftime('%Y-%m', timestamp) as month,
                COUNT(*) as total_users,
                SUM(CASE WHEN direction = 'IN' THEN 1 ELSE 0 END) as total_in,
                SUM(CASE WHEN direction = 'OUT' THEN 1 ELSE 0 END) as total_out,
                COUNT(DISTINCT camera_id) as cameras_used
            FROM tracking_logs 
            WHERE strftime('%Y', timestamp) = ?
            GROUP BY strftime('%Y-%m', timestamp)
            ORDER BY strftime('%Y-%m', timestamp)
        """, (str(year),)).fetchall()
        
        # Calculate yearly summary
        if monthly_data:
            total_users = sum(row['total_users'] for row in monthly_data)
            total_in = sum(row['total_in'] for row in monthly_data)
            total_out = sum(row['total_out'] for row in monthly_data)
            cameras_used = len(set(conn.execute("""
                SELECT DISTINCT camera_id FROM tracking_logs 
                WHERE strftime('%Y', timestamp) = ?
            """, (str(year),)).fetchall()))
            
            return {
                'period': str(year),
                'monthly_data': [dict(row) for row in monthly_data],
                'summary': {
                    'total_users': total_users,
                    'total_in': total_in,
                    'total_out': total_out,
                    'cameras_used': cameras_used
                }
            }
        else:
            return {
                'period': str(year),
                'monthly_data': [],
                'summary': {
                    'total_users': 0,
                    'total_in': 0,
                    'total_out': 0,
                    'cameras_used': 0
                }
            }


def log_generated_report(report_type, report_period, file_path, generated_by):
    """Log a generated report"""
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO generated_reports (report_type, report_period, file_path, generated_by)
            VALUES (?, ?, ?, ?)
        """, (report_type, report_period, file_path, generated_by))
        conn.commit()


def get_report_history():
    """Get history of generated reports"""
    with get_db_connection() as conn:
        reports = conn.execute("""
            SELECT * FROM generated_reports 
            ORDER BY generated_at DESC 
            LIMIT 20
        """).fetchall()
        
        return [dict(report) for report in reports]

if __name__ == '__main__':
    setup_database()
