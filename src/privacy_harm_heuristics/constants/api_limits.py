"""Centralized API rate limit defaults and helpers.

Provides documented defaults with environment overrides so collectors/clients
can pace requests close to provider limits without risking bans.
"""

from __future__ import annotations

import os

# -----------------------------
# Reddit
# -----------------------------

# Default per-minute cap used by our client-side limiter. Reddit's OAuth rate
# limits vary; PRAW also respects server-provided headers. Keep a conservative
# default and allow override via environment.
REDDIT_REQUESTS_PER_MIN_DEFAULT = 60
REDDIT_REQUESTS_PER_MIN_ENV = "REDDIT_REQUESTS_PER_MIN"


def get_reddit_requests_per_minute() -> int:
    """Return the configured Reddit per-minute request cap.

    Reads ``REDDIT_REQUESTS_PER_MIN`` from environment; falls back to
    :data:`REDDIT_REQUESTS_PER_MIN_DEFAULT` when unset/invalid.
    """

    raw = os.getenv(REDDIT_REQUESTS_PER_MIN_ENV)
    if not raw:
        return REDDIT_REQUESTS_PER_MIN_DEFAULT
    try:
        val = int(float(raw))
        return max(0, val)
    except (TypeError, ValueError):
        return REDDIT_REQUESTS_PER_MIN_DEFAULT


# -----------------------------
# Wikipedia (placeholder for symmetry; wikipedia lib has internal throttling)
# -----------------------------

WIKIPEDIA_REQUESTS_PER_MIN_DEFAULT = 120  # external pacing when needed
WIKIPEDIA_REQUESTS_PER_MIN_ENV = "WIKIPEDIA_REQUESTS_PER_MIN"


def get_wikipedia_requests_per_minute() -> int:
    raw = os.getenv(WIKIPEDIA_REQUESTS_PER_MIN_ENV)
    if not raw:
        return WIKIPEDIA_REQUESTS_PER_MIN_DEFAULT
    try:
        val = int(float(raw))
        return max(0, val)
    except (TypeError, ValueError):
        return WIKIPEDIA_REQUESTS_PER_MIN_DEFAULT


# -----------------------------
# Mastodon
# -----------------------------

MASTODON_REQUESTS_PER_MIN_DEFAULT = 60
MASTODON_REQUESTS_PER_MIN_ENV = "MASTODON_REQUESTS_PER_MIN"


def get_mastodon_requests_per_minute() -> int:
    """Return configured Mastodon per-minute request cap."""

    raw = os.getenv(MASTODON_REQUESTS_PER_MIN_ENV)
    if not raw:
        return MASTODON_REQUESTS_PER_MIN_DEFAULT
    try:
        val = int(float(raw))
        return max(0, val)
    except (TypeError, ValueError):
        return MASTODON_REQUESTS_PER_MIN_DEFAULT
