# ⚡ Smart Attendance AI — Web App

Browser-based Face Recognition Attendance System built with Flask + OpenCV + DeepFace.

## Features
- 📷 Live webcam face recognition in browser
- 👥 Student registration with face enrollment
- ✅ One-click attendance marking
- 📊 Reports & CSV export
- 🔐 Secure login with role-based access
- 🛡 Liveness detection (anti-spoofing)

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser
# Go to: http://localhost:5000
# Login: admin / Admin@123
```

## Deploy on Render (Free)

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and deploys!

## Project Structure
```
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── templates/          # HTML pages
│   ├── base.html
│   ├── login.html
│   ├── dashboard.html
│   ├── attendance.html
│   ├── register.html
│   ├── students.html
│   ├── reports.html
│   └── logs.html
├── database/           # DB manager
├── utils/              # Face engine + security
└── models/             # Haar cascade XML
```

## Default Login
- **Username:** `admin`
- **Password:** `Admin@123`
