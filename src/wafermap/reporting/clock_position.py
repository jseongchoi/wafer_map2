"""Clock-position utilities for wafer-centric reporting."""

from __future__ import annotations

import math


def clock_position_from_xy(x: float, y: float, cx: float, cy: float) -> str:
    """Return the nearest wafer clock position for an image coordinate."""

    dx = x - cx
    dy = y - cy
    if dx == 0 and dy == 0:
        return "center"

    # Image y grows downward, so 12 o'clock is negative dy.
    angle = math.atan2(dx, -dy)
    hour = int(round((angle % (2 * math.pi)) / (2 * math.pi) * 12))
    hour = 12 if hour == 0 else hour
    return f"{hour:02d}:00"
