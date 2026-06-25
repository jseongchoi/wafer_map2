"""Small file/path helpers shared by report scripts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def relative_path(target: str | Path, base_file: str | Path) -> str:
    """Return a POSIX-style relative path from a report file to a target."""

    return os.path.relpath(Path(target).resolve(), Path(base_file).resolve().parent).replace("\\", "/")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | tuple[str, ...] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def read_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))
