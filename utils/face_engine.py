"""
utils/face_engine.py
---------------------
Face Recognition Engine
  - Stage 1 : Haar Cascade (fast face detection)
  - Stage 2 : FaceNet-512 deep-learning embeddings
  - Stage 3 : Cosine-similarity matching
  - Stage 4 : Liveness / anti-spoofing check
"""

import cv2
import os
import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
CASCADE_PATH    = str(BASE_DIR / 'models' / 'haarcascade_frontalface_default.xml')
FACES_DIR       = str(BASE_DIR / 'faces')
MODELS_DIR      = str(BASE_DIR / 'models')

os.makedirs(FACES_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.40   # cosine distance — lower = stricter
MIN_FACE_SIZE         = 80     # pixels
REGISTRATION_SAMPLES  = 5      # frames to average during registration
EYE_BLINK_FRAMES      = 3      # liveness: consecutive blink frames


# ══════════════════════════════════════════════════════════════════
#  HAAR CASCADE DETECTOR
# ══════════════════════════════════════════════════════════════════

class HaarFaceDetector:
    """OpenCV Haar Cascade face detector."""

    def __init__(self):
        self.cascade = self._load_cascade()
        self.eye_cascade = self._load_eye_cascade()

    def _load_cascade(self):
        # Try local model first, then OpenCV built-in
        if os.path.exists(CASCADE_PATH):
            c = cv2.CascadeClassifier(CASCADE_PATH)
        else:
            built_in = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            c = cv2.CascadeClassifier(built_in)
            # Save a copy to models/
            import shutil
            os.makedirs(MODELS_DIR, exist_ok=True)
            shutil.copy(built_in, CASCADE_PATH)

        if c.empty():
            raise RuntimeError("❌ Haar cascade file not found or corrupted!")
        logger.info("✅ Haar cascade loaded.")
        return c

    def _load_eye_cascade(self):
        path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        c = cv2.CascadeClassifier(path)
        return c if not c.empty() else None

    def detect(self, frame, scale=1.1, min_neighbors=5):
        """
        Detect faces in frame.
        Returns list of (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improve contrast

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=min_neighbors,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces if len(faces) > 0 else []

    def detect_eyes(self, face_roi_gray):
        """Detect eyes inside a face ROI (used for liveness)."""
        if self.eye_cascade is None:
            return []
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5,
                                                  minSize=(20, 20))
        return eyes

    def crop_face(self, frame, box, padding=20):
        """Crop and return face region with padding."""
        x, y, w, h = box
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        return frame[y1:y2, x1:x2]


# ══════════════════════════════════════════════════════════════════
#  FACENET-512 EMBEDDER (DeepFace wrapper)
# ══════════════════════════════════════════════════════════════════

class FaceNetEmbedder:
    """
    Extract 512-dimensional face embeddings using FaceNet.
    Uses DeepFace library under the hood.
    """

    def __init__(self):
        self.model = None
        self.model_name = 'Facenet512'
        self._load_model()

    def _load_model(self):
        try:
            from deepface import DeepFace
            # Warm up / download model on first run
            self.DeepFace = DeepFace
            logger.info("✅ DeepFace / FaceNet512 ready.")
            print("✅ FaceNet-512 model loaded.")
        except ImportError:
            logger.warning("DeepFace not installed — using OpenCV LBPH fallback.")
            self.DeepFace = None

    def get_embedding(self, face_img):
        """
        Get 512-dim embedding vector from face image.
        Returns numpy array or None on failure.
        """
        if self.DeepFace is None:
            return self._lbph_fallback(face_img)

        try:
            # DeepFace expects BGR or RGB numpy array
            result = self.DeepFace.represent(
                img_path     = face_img,
                model_name   = self.model_name,
                detector_backend = 'skip',   # we already detected with Haar
                enforce_detection = False,
                align        = True,
                normalization = 'Facenet'
            )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")

        return None

    def _lbph_fallback(self, face_img):
        """Fallback: flatten resized face as pseudo-embedding."""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            flat = resized.flatten().astype(np.float32)
            # L2 normalize
            norm = np.linalg.norm(flat)
            return flat / norm if norm > 0 else flat
        except Exception:
            return None

    def cosine_distance(self, emb1, emb2):
        """Compute cosine distance between two embeddings. Lower = more similar."""
        if emb1 is None or emb2 is None:
            return 1.0
        e1 = np.array(emb1, dtype=np.float32)
        e2 = np.array(emb2, dtype=np.float32)
        dot = np.dot(e1, e2)
        norm = np.linalg.norm(e1) * np.linalg.norm(e2)
        if norm == 0:
            return 1.0
        similarity = dot / norm
        return float(1.0 - similarity)   # distance (0 = identical)


# ══════════════════════════════════════════════════════════════════
#  LIVENESS DETECTOR (Anti-Spoofing)
# ══════════════════════════════════════════════════════════════════

class LivenessDetector:
    """
    Basic anti-spoofing using:
    1. Eye blink detection (eyes present → real face)
    2. Texture analysis (Laplacian variance — photos are blurry)
    3. Face motion across frames
    """

    def __init__(self):
        self.history = []   # recent face frame history
        self.blink_counter = 0
        self.EAR_THRESHOLD = 0.25   # Eye Aspect Ratio threshold

    def check_texture(self, face_gray):
        """High Laplacian variance = sharp = real face."""
        lap = cv2.Laplacian(face_gray, cv2.CV_64F)
        variance = lap.var()
        return variance > 80.0, float(variance)   # (is_real, score)

    def check_eyes_present(self, eye_cascade, face_gray):
        """If no eyes detected, likely a photo/mask."""
        if eye_cascade is None:
            return True   # can't check — assume real
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
        return len(eyes) >= 1

    def check_motion(self, face_img):
        """
        Compare current face to recent frames.
        Real faces have micro-movements; printed photos are static.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        self.history.append(gray)

        if len(self.history) > 10:
            self.history.pop(0)

        if len(self.history) < 3:
            return True, 0.0

        # Mean absolute difference between consecutive frames
        diffs = [
            np.mean(np.abs(self.history[i].astype(float) - self.history[i-1].astype(float)))
            for i in range(1, len(self.history))
        ]
        avg_motion = float(np.mean(diffs))
        # Real faces typically have motion > 1.5
        return avg_motion > 1.0, avg_motion

    def is_live(self, face_img, eye_cascade=None):
        """
        Combined liveness check.
        Returns (is_live: bool, score: float, reason: str)
        """
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        texture_ok, texture_score = self.check_texture(face_gray)
        motion_ok, motion_score   = self.check_motion(face_img)
        eyes_ok                   = self.check_eyes_present(eye_cascade, face_gray)

        # Scoring
        score = 0.0
        if texture_ok: score += 0.4
        if motion_ok:  score += 0.4
        if eyes_ok:    score += 0.2

        is_live = score >= 0.6

        reason = (
            f"texture={texture_score:.1f} | "
            f"motion={motion_score:.2f} | "
            f"eyes={'yes' if eyes_ok else 'no'} | "
            f"score={score:.2f}"
        )

        return is_live, score, reason


# ══════════════════════════════════════════════════════════════════
#  FACE RECOGNITION ENGINE (Puts it all together)
# ══════════════════════════════════════════════════════════════════

class FaceRecognitionEngine:
    """
    Main engine combining:
      - Haar Cascade detection
      - FaceNet-512 embeddings
      - Cosine similarity matching
      - Liveness detection
    """

    def __init__(self):
        print("🔄 Initializing Face Recognition Engine...")
        self.detector   = HaarFaceDetector()
        self.embedder   = FaceNetEmbedder()
        self.liveness   = LivenessDetector()
        self.db_cache   = {}   # {student_id: embedding_array}
        self.load_from_db()
        print("✅ Face Recognition Engine ready!\n")

    # ── Database cache ────────────────────────────────────────────

    def load_from_db(self):
        """Load all face encodings from database into memory."""
        from database.db_manager import get_all_students
        students = get_all_students()
        self.db_cache = {}
        for s in students:
            if s['face_encoding']:
                self.db_cache[s['student_id']] = {
                    'name':     s['name'],
                    'encoding': np.array(s['face_encoding'], dtype=np.float32)
                }
        print(f"📂 Loaded {len(self.db_cache)} face encodings from DB.")

    def reload_cache(self):
        self.load_from_db()

    # ── Registration ─────────────────────────────────────────────

    def register_student(self, student_id, name, frames, save_images=True):
        """
        Register a student's face from multiple frames.
        Averages embeddings for robustness.

        Returns: (success: bool, message: str, embedding: list)
        """
        embeddings = []
        saved_paths = []

        for i, frame in enumerate(frames):
            faces = self.detector.detect(frame)
            if len(faces) == 0:
                continue
            if len(faces) > 1:
                return False, "Multiple faces detected! Please ensure only one person.", None

            x, y, w, h = faces[0]
            face_crop = self.detector.crop_face(frame, faces[0])

            # Liveness check during registration
            is_live, live_score, live_reason = self.liveness.is_live(
                face_crop, self.detector.eye_cascade
            )
            if not is_live and i > 2:   # allow first 2 frames to warm up
                return False, f"Liveness check failed: {live_reason}", None

            emb = self.embedder.get_embedding(face_crop)
            if emb is not None:
                embeddings.append(emb)

            # Save face image
            if save_images:
                face_dir = os.path.join(FACES_DIR, student_id)
                os.makedirs(face_dir, exist_ok=True)
                img_path = os.path.join(face_dir, f"face_{i+1}.jpg")
                cv2.imwrite(img_path, face_crop)
                saved_paths.append(img_path)

        if len(embeddings) < 1:
            return False, "No valid face detected. Try again in better lighting.", None

        # Average embedding for robustness
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)   # L2 normalize

        # Save to DB
        from database.db_manager import update_student_encoding
        update_student_encoding(student_id, avg_embedding.tolist(), saved_paths)

        # Update cache
        self.db_cache[student_id] = {
            'name':     name,
            'encoding': avg_embedding
        }

        msg = f"✅ '{name}' registered with {len(embeddings)} sample(s)."
        logger.info(msg)
        return True, msg, avg_embedding.tolist()

    # ── Recognition ──────────────────────────────────────────────

    def recognize_faces(self, frame, check_liveness=True):
        """
        Detect and identify all faces in a frame.

        Returns list of:
          {
            'box':        (x,y,w,h),
            'student_id': str or None,
            'name':       str,
            'confidence': float,   # 0-1, higher = more confident
            'is_live':    bool,
            'live_score': float,
            'status':     'recognized' | 'unknown' | 'spoof'
          }
        """
        results = []
        faces = self.detector.detect(frame)

        if len(faces) == 0:
            return results

        for box in faces:
            x, y, w, h = box
            face_crop = self.detector.crop_face(frame, box)

            if face_crop is None or face_crop.size == 0:
                continue

            # ── Liveness Check ────────────────────────────────
            is_live, live_score, live_reason = True, 1.0, ""
            if check_liveness:
                is_live, live_score, live_reason = self.liveness.is_live(
                    face_crop, self.detector.eye_cascade
                )

            if not is_live:
                results.append({
                    'box': box, 'student_id': None, 'name': '⚠ Spoof Detected',
                    'confidence': 0.0, 'is_live': False,
                    'live_score': live_score, 'status': 'spoof'
                })
                continue

            # ── Get Embedding ─────────────────────────────────
            emb = self.embedder.get_embedding(face_crop)
            if emb is None:
                results.append({
                    'box': box, 'student_id': None, 'name': 'Unknown',
                    'confidence': 0.0, 'is_live': is_live,
                    'live_score': live_score, 'status': 'unknown'
                })
                continue

            # ── Match against DB ──────────────────────────────
            best_id, best_dist = self._find_best_match(emb)

            if best_dist < RECOGNITION_THRESHOLD:
                student_info = self.db_cache.get(best_id, {})
                confidence   = float(1.0 - best_dist)
                results.append({
                    'box':        box,
                    'student_id': best_id,
                    'name':       student_info.get('name', 'Unknown'),
                    'confidence': confidence,
                    'is_live':    is_live,
                    'live_score': live_score,
                    'status':     'recognized'
                })
            else:
                results.append({
                    'box': box, 'student_id': None, 'name': 'Unknown',
                    'confidence': float(1.0 - best_dist),
                    'is_live': is_live, 'live_score': live_score,
                    'status': 'unknown'
                })

        return results

    def _find_best_match(self, query_emb):
        """Find closest match in DB cache."""
        best_id   = None
        best_dist = 1.0

        for sid, data in self.db_cache.items():
            dist = self.embedder.cosine_distance(query_emb, data['encoding'])
            if dist < best_dist:
                best_dist = dist
                best_id   = sid

        return best_id, best_dist

    # ── Drawing ───────────────────────────────────────────────────

    def draw_results(self, frame, results):
        """Draw bounding boxes, names, confidence on frame."""
        for r in results:
            x, y, w, h = [int(v) for v in r['box']]
            status = r['status']

            if status == 'recognized':
                color = (0, 220, 0)       # green
            elif status == 'spoof':
                color = (0, 0, 255)       # red
            else:
                color = (0, 165, 255)     # orange

            # Bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Corner decorations
            corner = 20
            cv2.line(frame, (x, y),         (x+corner, y),       color, 3)
            cv2.line(frame, (x, y),         (x, y+corner),       color, 3)
            cv2.line(frame, (x+w, y),       (x+w-corner, y),     color, 3)
            cv2.line(frame, (x+w, y),       (x+w, y+corner),     color, 3)
            cv2.line(frame, (x, y+h),       (x+corner, y+h),     color, 3)
            cv2.line(frame, (x, y+h),       (x, y+h-corner),     color, 3)
            cv2.line(frame, (x+w, y+h),     (x+w-corner, y+h),   color, 3)
            cv2.line(frame, (x+w, y+h),     (x+w, y+h-corner),   color, 3)

            # Label background
            label = f"{r['name']}  {r['confidence']*100:.1f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y-lh-12), (x+lw+8, y), color, -1)
            cv2.putText(frame, label, (x+4, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            # Liveness indicator
            live_text = f"LIVE {r['live_score']*100:.0f}%" if r['is_live'] else "SPOOF"
            live_color = (0, 220, 0) if r['is_live'] else (0, 0, 255)
            cv2.putText(frame, live_text, (x, y+h+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, live_color, 1)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return frame
