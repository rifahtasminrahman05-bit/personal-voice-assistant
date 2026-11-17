# -*- coding: utf-8 -*-
"""Encryption manager — opt-in Fernet encryption for personal memory data.

Disabled until a master password unlocks it for the first time. The password
never touches disk; only a salted verification token does.
"""
import base64
import hashlib
import os

from core import config

VERIFY_FILE = os.path.join(config.BASE_DIR, "data", ".crypto_verify")


def is_encryption_enabled():
    return os.path.exists(VERIFY_FILE)


def _derive_key(password, salt):
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return base64.urlsafe_b64encode(key)


class EncryptionManager:
    def __init__(self):
        self._fernet = None

    @property
    def unlocked(self):
        return self._fernet is not None

    def unlock(self, password):
        """Verify the master password and enable encrypt/decrypt. Returns bool."""
        try:
            from cryptography.fernet import Fernet, InvalidToken
        except ImportError:
            print("encryption unavailable: pip install cryptography")
            return False
        if not password:
            return False
        if is_encryption_enabled():
            with open(VERIFY_FILE, "rb") as f:
                blob = f.read()
            salt, token = blob[:16], blob[16:]
            try:
                self._fernet = Fernet(_derive_key(password, salt))
                self._fernet.decrypt(token)
                return True
            except (InvalidToken, Exception):
                self._fernet = None
                return False
        salt = os.urandom(16)
        self._fernet = Fernet(_derive_key(password, salt))
        token = self._fernet.encrypt(b"octo-verify")
        os.makedirs(os.path.dirname(VERIFY_FILE), exist_ok=True)
        with open(VERIFY_FILE, "wb") as f:
            f.write(salt + token)
        return True

    def encrypt(self, plaintext):
        if self._fernet is None:
            raise RuntimeError("encryption manager is locked")
        return self._fernet.encrypt(plaintext.encode())

    def decrypt(self, ciphertext):
        if self._fernet is None:
            raise RuntimeError("encryption manager is locked")
        return self._fernet.decrypt(ciphertext).decode()
