"""Shared helpers for script command-line interfaces."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse options."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
