"""Date normalization helpers.

Converts heterogeneous incident_date representations (epoch ints, mm/dd/YYYY,
ISO strings) into canonical ISO 8601 date (YYYY-MM-DD) or datetime strings.
We intentionally keep time component out unless provided.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DATE_INPUT_FORMATS = [
    "%m/%d/%Y",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
]


def canonicalize_incident_date(value: Any) -> str | None:
    """Return ISO date (YYYY-MM-DD) from heterogeneous input formats.

    Accepts epoch seconds (int/float), common string formats (``mm/dd/YYYY``,
    ``YYYY-MM-DD``, several ``YYYY-MM-DDTHH:MM:SS[Z]`` variants) and lenient
    slash-separated dates. Invalid / unrecognized inputs return ``None``.

    Args:
        value: Raw input date representation (number, string, etc.).

    Returns:
        Canonical date string or ``None`` when parsing fails.
    """
    if value is None or value == "":
        return None
    # Epoch numeric
    if isinstance(value, (int, float)):
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.date().isoformat()
        except Exception:
            return None
    # Already ISO-like and length 10 (YYYY-MM-DD)
    s = str(value).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    # Try known formats
    for fmt in DATE_INPUT_FORMATS:
        try:
            dt = datetime.strptime(s.replace("Z", ""), fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    # Try splitting mm/dd/YYYY
    if "/" in s and len(s.split("/")) == 3:
        parts = s.split("/")
        if len(parts[2]) == 4:
            try:
                month, day, year = parts
                dt = datetime(int(year), int(month), int(day))
                return dt.date().isoformat()
            except Exception:
                pass
    return None
