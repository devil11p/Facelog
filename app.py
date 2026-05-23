"""
app.py — Smart Attendance AI System (Flask Web App)
Browser-based webcam face recognition attendance system
"""

import cv2
import os
import sys
import json
import base64
import logging
import numpy as np
from datetime import datetime, date
from pathlib import Path
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, send_file, Response)
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.db_manager import (
    init_database, add_student, get_all_students, get_student,
    mark_attendance, get_attendance_by_date, verify_user,
    log_security_event, export_attendance_csv, get_security_logs,
    delete_student, get_attendance_report, create_user
)
from utils.face_engine import FaceRecognitionEngine

# ── Logging ────────────────────────────────────────────────────────
os.makedirs(BASE_DIR / 'logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / 'logs' / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# ── Flask App ──────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'attendance-ai-secret-2024')

login_manager = LoginManager(app)
login_manager.login_view = 'login_page'
login_manager.login_message = 'Please login to access this page.'

init_database()

# ── Face Engine (lazy load) ────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = FaceRecognitionEngine()
    return _engine


# ── User class for Flask-Login ─────────────────────────────────────
class User(UserMixin):
    def __init__(self, username, role, full_name):
        self.id = username
        self.username = username
        self.role = role
        self.full_name = full_name


@login_manager.user_loader
def load_user(username):
    # We keep minimal info in session
    role = session.get('role', 'teacher')
    full_name = session.get('full_name', username)
    return User(username, role, full_name)


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role not in ('admin', 'teacher'):
            return jsonify({'error': 'Access denied'}), 403
        return f(*args, **kwargs)
    return decorated


# ══════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    students = get_all_students()
    today_records = get_attendance_by_date()
    present_today = len(today_records)
    total_students = len(students)
    return render_template('dashboard.html',
                           total_students=total_students,
                           present_today=present_today,
                           user=current_user)


@app.route('/attendance')
@login_required
def attendance_page():
    return render_template('attendance.html', user=current_user)


@app.route('/register')
@login_required
def register_page():
    return render_template('register.html', user=current_user)


@app.route('/students')
@login_required
def students_page():
    students = get_all_students()
    return render_template('students.html', students=students, user=current_user)


@app.route('/reports')
@login_required
def reports_page():
    return render_template('reports.html', user=current_user)


@app.route('/logs')
@login_required
def logs_page():
    logs = get_security_logs(limit=50)
    return render_template('logs.html', logs=logs, user=current_user)


# ══════════════════════════════════════════════════════════════════
#  AUTH API
# ══════════════════════════════════════════════════════════════════

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400

    result = verify_user(username, password)

    if result is None:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    if isinstance(result, dict) and result.get('locked'):
        return jsonify({'success': False, 'error': f"Account locked. Try again later."}), 403

    user = User(result['username'], result['role'], result.get('full_name', username))
    login_user(user)
    session['role'] = result['role']
    session['full_name'] = result.get('full_name', username)

    log_security_event('login', username=username, severity='INFO', details='Web login')
    return jsonify({'success': True, 'role': result['role'], 'name': result.get('full_name', username)})


@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    log_security_event('logout', username=current_user.username, severity='INFO')
    logout_user()
    return jsonify({'success': True})


# ══════════════════════════════════════════════════════════════════
#  FACE RECOGNITION API
# ══════════════════════════════════════════════════════════════════

@app.route('/api/recognize', methods=['POST'])
@login_required
def api_recognize():
    """Receive a base64 frame, run face recognition, return results."""
    try:
        data = request.get_json()
        img_data = data.get('image', '')

        # Decode base64 image
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        engine = get_engine()
        results = engine.recognize_faces(frame, check_liveness=True)

        # Format results for JSON
        output = []
        for r in results:
            box = [int(v) for v in r['box']]
            output.append({
                'box': box,
                'student_id': r.get('student_id'),
                'name': r.get('name', 'Unknown'),
                'confidence': round(r.get('confidence', 0) * 100, 1),
                'is_live': r.get('is_live', True),
                'status': r.get('status', 'unknown')
            })

        return jsonify({'success': True, 'faces': output})
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mark_attendance', methods=['POST'])
@login_required
def api_mark_attendance():
    """Mark attendance for a recognized student."""
    data = request.get_json()
    student_id = data.get('student_id')
    confidence = data.get('confidence', 0) / 100

    if not student_id:
        return jsonify({'error': 'No student_id provided'}), 400

    result = mark_attendance(
        student_id=student_id,
        confidence=confidence,
        method='Face (Web)',
        marked_by=current_user.username
    )

    student = get_student(student_id)
    name = student['name'] if student else student_id

    log_security_event('attendance_marked', username=current_user.username,
                       student_id=student_id, severity='INFO',
                       details=f"{name} | {result} | conf={confidence:.2f}")

    return jsonify({
        'success': True,
        'action': result,
        'name': name,
        'time': datetime.now().strftime('%H:%M:%S')
    })


# ══════════════════════════════════════════════════════════════════
#  REGISTRATION API
# ══════════════════════════════════════════════════════════════════

@app.route('/api/register_student', methods=['POST'])
@login_required
def api_register_student():
    """Register a new student with face images."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        roll_no = data.get('roll_no', '').strip()
        class_name = data.get('class_name', '').strip()
        section = data.get('section', '').strip()
        email = data.get('email', '').strip()
        images_b64 = data.get('images', [])  # list of base64 frames

        if not name or not roll_no:
            return jsonify({'success': False, 'error': 'Name and Roll No required'}), 400

        if len(images_b64) < 3:
            return jsonify({'success': False, 'error': 'At least 3 face images required'}), 400

        # Add student to DB first
        student_id = add_student(
            name=name, roll_no=roll_no,
            class_name=class_name, section=section,
            email=email, created_by=current_user.username
        )
        if not student_id:
            return jsonify({'success': False, 'error': 'Roll number already exists'}), 409

        # Decode frames
        frames = []
        for b64 in images_b64:
            if ',' in b64:
                b64 = b64.split(',')[1]
            img_bytes = base64.b64decode(b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return jsonify({'success': False, 'error': 'Could not decode images'}), 400

        engine = get_engine()
        success, msg, _ = engine.register_student(student_id, name, frames)

        if success:
            engine.reload_cache()
            return jsonify({'success': True, 'student_id': student_id, 'message': msg})
        else:
            delete_student(student_id)
            return jsonify({'success': False, 'error': msg}), 400

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════
#  DATA API
# ══════════════════════════════════════════════════════════════════

@app.route('/api/attendance_today')
@login_required
def api_attendance_today():
    date_str = request.args.get('date', date.today().isoformat())
    records = get_attendance_by_date(date_str)
    students = get_all_students()
    present_ids = {r['student_id'] for r in records}

    result = []
    for s in students:
        rec = next((r for r in records if r['student_id'] == s['student_id']), None)
        result.append({
            'student_id': s['student_id'],
            'name': s['name'],
            'roll_no': s.get('roll_no', ''),
            'class_name': s.get('class_name', ''),
            'status': rec['status'] if rec else 'Absent',
            'time_in': rec['time_in'] if rec else '—',
            'time_out': rec['time_out'] if rec else '—',
            'confidence': f"{rec['confidence']*100:.1f}%" if rec and rec.get('confidence') else '—',
        })
    return jsonify({'success': True, 'records': result, 'date': date_str})


@app.route('/api/students')
@login_required
def api_students():
    students = get_all_students()
    return jsonify({'success': True, 'students': [
        {k: v for k, v in s.items() if k not in ('face_encoding',)}
        for s in students
    ]})


@app.route('/api/delete_student/<student_id>', methods=['DELETE'])
@login_required
def api_delete_student(student_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin only'}), 403
    delete_student(student_id)
    get_engine().reload_cache()
    return jsonify({'success': True})


@app.route('/api/export_csv')
@login_required
def api_export_csv():
    date_str = request.args.get('date', date.today().isoformat())
    filepath = export_attendance_csv(date_str)
    return send_file(filepath, as_attachment=True,
                     download_name=f'attendance_{date_str}.csv')


@app.route('/api/report')
@login_required
def api_report():
    start = request.args.get('start', date.today().isoformat())
    end = request.args.get('end', date.today().isoformat())
    report = get_attendance_report(start, end)
    return jsonify({'success': True, 'report': report})


@app.route('/api/stats')
@login_required
def api_stats():
    students = get_all_students()
    today_records = get_attendance_by_date()
    present_today = len(today_records)
    total = len(students)
    return jsonify({
        'total_students': total,
        'present_today': present_today,
        'absent_today': total - present_today,
        'attendance_pct': round((present_today / total * 100) if total else 0, 1),
        'date': date.today().strftime('%d %B %Y')
    })


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    init_database()
    logger.info("Starting Smart Attendance AI Web App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
