from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os


_ITERATIONS = 100_000
_SALT_SIZE = 16
_ALGORITHM = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password must not be empty")

    salt = os.urandom(_SALT_SIZE)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"{_ALGORITHM}${_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        algorithm, iterations_str, salt_b64, digest_b64 = hashed_password.split("$", maxsplit=3)
        if algorithm != _ALGORITHM:
            return False

        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected_digest = base64.b64decode(digest_b64.encode("ascii"))
        calculated_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(iterations_str),
        )
        return hmac.compare_digest(calculated_digest, expected_digest)
    except (ValueError, binascii.Error):
        return False