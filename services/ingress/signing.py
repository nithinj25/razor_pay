"""Razorpay webhook HMAC. Verified against RAW bytes.

Pitfall #2, and the one that costs the most hours: if you `json.loads()`
the body and re-serialise before hashing, key order and separators shift
and the signature will never match. Nothing in this module accepts a
parsed dict - only bytes.
"""

from __future__ import annotations

import hashlib
import hmac


def sign(raw: bytes | str, secret: str) -> str:
    """HMAC-SHA256 hex digest, the format Razorpay sends."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def verify(raw: bytes | str, signature: str, secret: str) -> bool:
    """Constant-time comparison. Never `==` on a MAC."""
    if not secret or not signature:
        return False
    return hmac.compare_digest(sign(raw, secret), signature)


def verify_any(raw: bytes | str, signature: str, *secrets: str) -> bool:
    """E8 - dual-secret rotation window.

    Razorpay retries a failed delivery for 24h. If the secret is rotated
    mid-stream, those retries still carry signatures made with the old
    secret; rejecting them would drop real events. Both secrets stay
    valid until the retry window has drained.

    Every candidate is checked even after a match so the work is constant
    regardless of which secret succeeded.
    """
    ok = False
    for s in secrets:
        if s and verify(raw, signature, s):
            ok = True
    return ok
