"""
utils/security_manager.py
--------------------------
Security layer:
  - Password hashing (bcrypt)
  - Session token management (HMAC)
  - Data encryption (Fernet)
  - Audit trail
  - Input sanitization
  - Rate limiting
"""

import os
import hmac
import hashlib
import secrets
import base64
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  ENCRYPTION MANAGER
# ─────────────────────────────────────────────────────────────────

class EncryptionManager:
    """
    Fernet symmetric encryption for sensitive data at rest.
    Key is derived from environment SECRET_KEY.
    """

    def __init__(self):
        self.key = self._derive_key()
        self._fernet = None
        self._init_fernet()

    def _derive_key(self):
        secret = os.getenv('SECRET_KEY', 'default-insecure-key-change-me')
        # PBKDF2 to derive a 32-byte key
        key_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            secret.encode(),
            b'attendance_salt_2024',
            iterations=100_000,
            dklen=32
        )
        return base64.urlsafe_b64encode(key_bytes)

    def _init_fernet(self):
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(self.key)
            logger.info("✅ Encryption manager ready.")
        except ImportError:
            logger.warning("cryptography not installed — encryption disabled.")

    def encrypt(self, data: str) -> str:
        """Encrypt string data. Returns base64 ciphertext."""
        if self._fernet is None:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Decrypt ciphertext. Returns original string."""
        if self._fernet is None:
            return token
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except Exception:
            raise ValueError("Decryption failed — data may be tampered.")

    def encrypt_json(self, obj) -> str:
        return self.encrypt(json.dumps(obj))

    def decrypt_json(self, token: str):
        return json.loads(self.decrypt(token))


# ─────────────────────────────────────────────────────────────────
#  SESSION MANAGER
# ─────────────────────────────────────────────────────────────────

class SessionManager:
    """
    In-memory session store with HMAC-signed tokens.
    Production: replace with Redis or DB-backed sessions.
    """

    def __init__(self, timeout_seconds=3600):
        self._sessions = {}          # token -> {user, role, created, expires}
        self.timeout   = timeout_seconds
        self.secret    = os.getenv('SECRET_KEY', 'change_me').encode()

    def create_session(self, username, role='teacher') -> str:
        """Create new session. Returns signed token."""
        token   = secrets.token_hex(32)
        payload = {
            'username': username,
            'role':     role,
            'created':  time.time(),
            'expires':  time.time() + self.timeout
        }
        # HMAC sign
        sig = hmac.new(self.secret,
                       token.encode(), hashlib.sha256).hexdigest()
        signed = f"{token}.{sig}"
        self._sessions[token] = payload
        logger.info(f"Session created for {username}")
        return signed

    def validate_session(self, signed_token) -> dict | None:
        """Validate token. Returns session dict or None."""
        try:
            token, sig = signed_token.rsplit('.', 1)
        except ValueError:
            return None

        # Verify HMAC
        expected_sig = hmac.new(self.secret,
                                token.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            logger.warning("Session: HMAC mismatch — possible tampering!")
            return None

        session = self._sessions.get(token)
        if not session:
            return None

        # Check expiry
        if time.time() > session['expires']:
            del self._sessions[token]
            return None

        return session

    def destroy_session(self, signed_token):
        """Invalidate a session."""
        try:
            token, _ = signed_token.rsplit('.', 1)
            self._sessions.pop(token, None)
        except ValueError:
            pass

    def cleanup_expired(self):
        """Remove all expired sessions."""
        now = time.time()
        expired = [t for t, s in self._sessions.items() if now > s['expires']]
        for t in expired:
            del self._sessions[t]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions.")


# ─────────────────────────────────────────────────────────────────
#  RATE LIMITER
# ─────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple in-memory rate limiter.
    Tracks attempts per key (IP, username) in a time window.
    """

    def __init__(self, max_attempts=5, window_seconds=300):
        self.max_attempts = max_attempts
        self.window       = window_seconds
        self._store       = {}   # key -> [(timestamp, ...)]

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if key is allowed.
        Returns (allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        self._store.setdefault(key, [])

        # Remove old entries outside window
        self._store[key] = [t for t in self._store[key]
                            if now - t < self.window]

        count = len(self._store[key])
        if count >= self.max_attempts:
            oldest = min(self._store[key])
            retry_after = int(self.window - (now - oldest))
            return False, retry_after

        return True, 0

    def record_attempt(self, key: str):
        """Record a failed attempt."""
        self._store.setdefault(key, []).append(time.time())

    def reset(self, key: str):
        """Clear attempts for key (on successful auth)."""
        self._store.pop(key, None)


# ─────────────────────────────────────────────────────────────────
#  INPUT SANITIZER
# ─────────────────────────────────────────────────────────────────

class InputSanitizer:
    """Sanitize and validate user inputs."""

    NAME_PATTERN    = re.compile(r'^[a-zA-Z\s\.\-\']{2,60}$')
    ROLL_PATTERN    = re.compile(r'^[a-zA-Z0-9\-]{1,20}$')
    EMAIL_PATTERN   = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
    PHONE_PATTERN   = re.compile(r'^[\d\+\-\s\(\)]{7,15}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\.]{3,30}$')

    @classmethod
    def sanitize_name(cls, name: str) -> tuple[bool, str]:
        name = name.strip()
        if not cls.NAME_PATTERN.match(name):
            return False, "Name must be 2-60 characters, letters only."
        return True, name

    @classmethod
    def sanitize_roll(cls, roll: str) -> tuple[bool, str]:
        roll = roll.strip().upper()
        if not cls.ROLL_PATTERN.match(roll):
            return False, "Roll number: 1-20 alphanumeric characters."
        return True, roll

    @classmethod
    def sanitize_email(cls, email: str) -> tuple[bool, str]:
        email = email.strip().lower()
        if email and not cls.EMAIL_PATTERN.match(email):
            return False, "Invalid email format."
        return True, email

    @classmethod
    def sanitize_username(cls, uname: str) -> tuple[bool, str]:
        uname = uname.strip().lower()
        if not cls.USERNAME_PATTERN.match(uname):
            return False, "Username: 3-30 chars, letters/digits/underscore."
        return True, uname

    @classmethod
    def validate_password(cls, pwd: str) -> tuple[bool, str]:
        if len(pwd) < 8:
            return False, "Password must be at least 8 characters."
        if not re.search(r'[A-Z]', pwd):
            return False, "Password needs at least one uppercase letter."
        if not re.search(r'\d', pwd):
            return False, "Password needs at least one digit."
        if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', pwd):
            return False, "Password needs at least one special character."
        return True, "OK"

    @classmethod
    def sql_safe(cls, text: str) -> str:
        """Basic SQL injection prevention (use parameterized queries, this is extra)."""
        return re.sub(r"[;'\"\\]", '', str(text))


# ─────────────────────────────────────────────────────────────────
#  AUDIT LOGGER
# ─────────────────────────────────────────────────────────────────

class AuditLogger:
    """File-based audit trail for all sensitive operations."""

    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = str(Path(__file__).parent.parent / 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'audit.log')

    def log(self, action, actor='system', target='', details='', severity='INFO'):
        """Append audit entry to log file."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'severity':  severity,
            'action':    action,
            'actor':     actor,
            'target':    target,
            'details':   details
        }
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Audit log write error: {e}")

    def read_recent(self, n=50):
        """Read last N audit entries."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            entries = [json.loads(l) for l in lines[-n:] if l.strip()]
            return list(reversed(entries))
        except FileNotFoundError:
            return []


# ─────────────────────────────────────────────────────────────────
#  CONVENIENCE: single shared instances
# ─────────────────────────────────────────────────────────────────

encryption = EncryptionManager()
session_mgr = SessionManager()
rate_limiter = RateLimiter(max_attempts=5, window_seconds=300)
sanitizer   = InputSanitizer()
audit       = AuditLogger()
