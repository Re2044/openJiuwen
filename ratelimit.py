"""Recognizing rate-limit errors (HTTP 429) and how long the proxy asks to wait."""
import random
import re

import config

# "Повторите через 60 с", "retry after 60 seconds", "Retry-After: 60"
_WAIT_PATTERNS = (
    re.compile(r"через\s+(\d+)\s*с", re.IGNORECASE),
    re.compile(r"retry[- ]after:?\s*(\d+)", re.IGNORECASE),
    re.compile(r"try again in\s+(\d+)", re.IGNORECASE),
)


def wait_seconds(error):
    """Seconds to wait before retrying, or None when `error` is not a rate limit."""
    text = str(error)
    if "429" not in text and "RateLimit" not in text and "rate limit" not in text.lower():
        return None
    for pattern in _WAIT_PATTERNS:
        match = pattern.search(text)
        if match:
            seconds = int(match.group(1)) + 1
            break
    else:
        # No time given, e.g. too many requests at once: a short pause is enough.
        seconds = config.RATE_LIMIT_DEFAULT_WAIT_SECONDS
    # Jitter keeps parallel sessions from retrying at the same instant.
    return min(seconds, config.RATE_LIMIT_MAX_WAIT_SECONDS) + random.uniform(0, 2)
