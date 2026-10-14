"""Only the host split rule from the private project (sha1(host) % 100 < 30 = test host)."""
import hashlib


def is_test_host(host: str) -> bool:
    return int(hashlib.sha1((host or "").encode("utf-8")).hexdigest()[:8], 16) % 100 < 30
