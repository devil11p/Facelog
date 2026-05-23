"""
database/db_manager.py
-----------------------
SQLite database manager for Smart Attendance AI System
Handles: Students, Attendance, Users, Security Logs
"""

import sqlite3
import os
import json
import hashlib
import bcrypt
from datetime import datetime, date
import pytz
import logging

logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'attendance.db')


def get_connection():
    """Get database connection with row factory."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_database():
    """Initialize all database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    # ── Students Table ──────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  TEXT UNIQUE NOT NULL,
            name        TEXT NOT NULL,
            roll_no     TEXT UNIQUE,
            class_name  TEXT,
            section     TEXT,
            email       TEXT,
            phone       TEXT,
            face_encoding TEXT,          -- JSON array of 512-dim FaceNet vector
            face_images   TEXT,          -- JSON array of image paths
            registered_at TEXT NOT NULL,
            updated_at    TEXT,
            is_active     INTEGER DEFAULT 1,
            created_by    TEXT DEFAULT 'admin'
        )
    """)

    # ── Attendance Table ─────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   TEXT NOT NULL,
            date         TEXT NOT NULL,
            time_in      TEXT,
            time_out     TEXT,
            status       TEXT DEFAULT 'Present',   -- Present / Absent / Late
            confidence   REAL,
            method       TEXT DEFAULT 'Face',       -- Face / Manual
            marked_by    TEXT DEFAULT 'system',
            device_info  TEXT,
            ip_address   TEXT,
            notes        TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
    """)

    # ── Admin Users Table ────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT DEFAULT 'teacher',  -- admin / teacher / viewer
            full_name     TEXT,
            email         TEXT,
            last_login    TEXT,
            login_attempts INTEGER DEFAULT 0,
            locked_until  TEXT,
            is_active     INTEGER DEFAULT 1,
            created_at    TEXT NOT NULL
        )
    """)

    # ── Security Logs Table ──────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,   -- login / logout / failed_login / face_registered / attendance_marked
            username   TEXT,
            student_id TEXT,
            details    TEXT,
            ip_address TEXT,
            timestamp  TEXT NOT NULL,
            severity   TEXT DEFAULT 'INFO'  -- INFO / WARNING / CRITICAL
        )
    """)

    # ── Classes Table ────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL,
            section    TEXT,
            teacher    TEXT,
            subject    TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()

    # Create default admin user if not exists
    _create_default_admin(cursor, conn)

    conn.close()
    logger.info("Database initialized successfully.")
    print("✅ Database initialized.")


def _create_default_admin(cursor, conn):
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        hashed = bcrypt.hashpw("Admin@123".encode(), bcrypt.gensalt()).decode()
        cursor.execute("""
            INSERT INTO users (username, password_hash, role, full_name, created_at)
            VALUES (?, ?, 'admin', 'System Admin', ?)
        """, ('admin', hashed, datetime.now(IST).isoformat()))
        conn.commit()
        print("✅ Default admin created → username: admin | password: Admin@123")


# ──────────────────────────────────────────────────────────────────
#  STUDENT OPERATIONS
# ──────────────────────────────────────────────────────────────────

def add_student(name, roll_no, class_name='', section='', email='', phone='',
                face_encoding=None, face_images=None, created_by='admin'):
    """Add new student to database."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        student_id = f"STU{roll_no.zfill(4)}" if roll_no else f"STU{int(datetime.now().timestamp())}"
        encoding_json = json.dumps(face_encoding) if face_encoding else None
        images_json   = json.dumps(face_images)   if face_images   else None

        cursor.execute("""
            INSERT INTO students
              (student_id, name, roll_no, class_name, section, email, phone,
               face_encoding, face_images, registered_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_id, name, roll_no, class_name, section, email, phone,
              encoding_json, images_json,
              datetime.now(IST).isoformat(), created_by))
        conn.commit()
        log_security_event('face_registered', student_id=student_id,
                           details=f"Student '{name}' registered", severity='INFO')
        return student_id
    except sqlite3.IntegrityError as e:
        logger.error(f"Student add error: {e}")
        return None
    finally:
        conn.close()


def update_student_encoding(student_id, face_encoding, face_images=None):
    """Update face encoding for an existing student."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE students
            SET face_encoding = ?, face_images = ?, updated_at = ?
            WHERE student_id = ?
        """, (json.dumps(face_encoding),
              json.dumps(face_images) if face_images else None,
              datetime.now(IST).isoformat(), student_id))
        conn.commit()
        return True
    finally:
        conn.close()


def get_all_students(active_only=True):
    """Return all students as list of dicts."""
    conn = get_connection()
    try:
        q = "SELECT * FROM students"
        if active_only:
            q += " WHERE is_active = 1"
        q += " ORDER BY name"
        rows = conn.execute(q).fetchall()
        students = []
        for row in rows:
            s = dict(row)
            s['face_encoding'] = json.loads(s['face_encoding']) if s['face_encoding'] else None
            s['face_images']   = json.loads(s['face_images'])   if s['face_images']   else []
            students.append(s)
        return students
    finally:
        conn.close()


def get_student(student_id):
    """Get single student by student_id."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM students WHERE student_id = ?",
                           (student_id,)).fetchone()
        if row:
            s = dict(row)
            s['face_encoding'] = json.loads(s['face_encoding']) if s['face_encoding'] else None
            s['face_images']   = json.loads(s['face_images'])   if s['face_images']   else []
            return s
        return None
    finally:
        conn.close()


def delete_student(student_id):
    """Soft-delete a student."""
    conn = get_connection()
    try:
        conn.execute("UPDATE students SET is_active = 0 WHERE student_id = ?", (student_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
#  ATTENDANCE OPERATIONS
# ──────────────────────────────────────────────────────────────────

def mark_attendance(student_id, confidence=None, method='Face',
                    marked_by='system', notes=''):
    """Mark attendance for today. Returns True if newly marked, False if duplicate."""
    today_str = date.today().isoformat()
    now_str   = datetime.now(IST).strftime('%H:%M:%S')
    conn = get_connection()
    try:
        # Check duplicate
        existing = conn.execute("""
            SELECT id, time_out FROM attendance
            WHERE student_id = ? AND date = ?
        """, (student_id, today_str)).fetchone()

        if existing:
            # Update time_out (check-out)
            conn.execute("""
                UPDATE attendance SET time_out = ?, confidence = ?
                WHERE id = ?
            """, (now_str, confidence, existing['id']))
            conn.commit()
            return 'checkout'

        # Determine status
        now_hour = datetime.now(IST).hour
        status = 'Late' if now_hour >= 10 else 'Present'

        conn.execute("""
            INSERT INTO attendance
              (student_id, date, time_in, status, confidence, method, marked_by, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_id, today_str, now_str, status, confidence, method, marked_by, notes))
        conn.commit()
        log_security_event('attendance_marked', student_id=student_id,
                           details=f"Attendance marked via {method} | confidence: {confidence:.2f}" if confidence else f"Attendance marked via {method}",
                           severity='INFO')
        return 'checkin'
    finally:
        conn.close()


def get_attendance_by_date(date_str=None, class_name=None):
    """Get attendance records for a specific date."""
    if not date_str:
        date_str = date.today().isoformat()
    conn = get_connection()
    try:
        query = """
            SELECT a.*, s.name, s.roll_no, s.class_name, s.section
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.date = ?
        """
        params = [date_str]
        if class_name:
            query += " AND s.class_name = ?"
            params.append(class_name)
        query += " ORDER BY a.time_in"
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def get_attendance_report(start_date, end_date, class_name=None):
    """Get attendance report for date range."""
    conn = get_connection()
    try:
        query = """
            SELECT s.name, s.roll_no, s.class_name, s.section,
                   COUNT(a.id) as present_days,
                   COUNT(CASE WHEN a.status='Late' THEN 1 END) as late_days
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id
                                   AND a.date BETWEEN ? AND ?
                                   AND a.status IN ('Present','Late')
            WHERE s.is_active = 1
        """
        params = [start_date, end_date]
        if class_name:
            query += " AND s.class_name = ?"
            params.append(class_name)
        query += " GROUP BY s.student_id ORDER BY s.name"
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
#  USER / AUTH OPERATIONS
# ──────────────────────────────────────────────────────────────────

def verify_user(username, password):
    """Verify login credentials. Returns user dict or None."""
    conn = get_connection()
    try:
        user = conn.execute("""
            SELECT * FROM users
            WHERE username = ? AND is_active = 1
        """, (username,)).fetchone()

        if not user:
            return None

        user = dict(user)
        now = datetime.now(IST)

        # Check lockout
        if user['locked_until']:
            locked_until = datetime.fromisoformat(user['locked_until'])
            if now < locked_until:
                remaining = int((locked_until - now).total_seconds())
                return {'locked': True, 'remaining': remaining}

        # Verify password
        if bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
            conn.execute("""
                UPDATE users SET login_attempts = 0, last_login = ?
                WHERE username = ?
            """, (now.isoformat(), username))
            conn.commit()
            log_security_event('login', username=username, severity='INFO',
                               details=f"Successful login")
            return user
        else:
            attempts = user['login_attempts'] + 1
            locked_until = None
            if attempts >= 5:
                from datetime import timedelta
                locked_until = (now + timedelta(seconds=300)).isoformat()
                log_security_event('failed_login', username=username, severity='CRITICAL',
                                   details=f"Account locked after {attempts} failed attempts")
            else:
                log_security_event('failed_login', username=username, severity='WARNING',
                                   details=f"Failed attempt {attempts}/5")
            conn.execute("""
                UPDATE users SET login_attempts = ?, locked_until = ?
                WHERE username = ?
            """, (attempts, locked_until, username))
            conn.commit()
            return None
    finally:
        conn.close()


def create_user(username, password, role='teacher', full_name='', email=''):
    """Create a new user."""
    conn = get_connection()
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        conn.execute("""
            INSERT INTO users (username, password_hash, role, full_name, email, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, hashed, role, full_name, email, datetime.now(IST).isoformat()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
#  SECURITY LOG OPERATIONS
# ──────────────────────────────────────────────────────────────────

def log_security_event(event_type, username=None, student_id=None,
                       details='', ip_address='', severity='INFO'):
    """Log a security event."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO security_logs
              (event_type, username, student_id, details, ip_address, timestamp, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_type, username, student_id, details, ip_address,
              datetime.now(IST).isoformat(), severity))
        conn.commit()
    except Exception as e:
        logger.error(f"Security log error: {e}")
    finally:
        conn.close()


def get_security_logs(limit=100, severity=None):
    """Get recent security logs."""
    conn = get_connection()
    try:
        q = "SELECT * FROM security_logs"
        params = []
        if severity:
            q += " WHERE severity = ?"
            params.append(severity)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(q, params).fetchall()]
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
#  EXPORT
# ──────────────────────────────────────────────────────────────────

def export_attendance_csv(date_str=None, class_name=None):
    """Export attendance to CSV and return filepath."""
    import csv, os
    if not date_str:
        date_str = date.today().isoformat()

    records = get_attendance_by_date(date_str, class_name)
    all_students = get_all_students()

    # Build present set
    present_set = {r['student_id'] for r in records}

    export_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'exports')
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, f"attendance_{date_str}.csv")

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Roll No', 'Name', 'Class', 'Section',
                         'Date', 'Time In', 'Time Out', 'Status', 'Confidence %', 'Method'])
        for s in all_students:
            rec = next((r for r in records if r['student_id'] == s['student_id']), None)
            writer.writerow([
                s.get('roll_no', ''),
                s['name'],
                s.get('class_name', ''),
                s.get('section', ''),
                date_str,
                rec['time_in']  if rec else '',
                rec['time_out'] if rec else '',
                rec['status']   if rec else 'Absent',
                f"{rec['confidence']*100:.1f}" if rec and rec.get('confidence') else '',
                rec['method']   if rec else '',
            ])

    return filepath


if __name__ == '__main__':
    init_database()
