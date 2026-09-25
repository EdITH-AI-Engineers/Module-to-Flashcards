from __future__ import annotations

from pathlib import Path
import re


def safe_path_component(value: object, *, fallback: str) -> str:
    """Return one portable path component without changing display metadata."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return component or fallback


def module_file_label(module_number: object) -> str:
    """Use human-friendly numeric module labels while retaining non-numeric IDs."""
    value = str(module_number).strip()
    if value.isdecimal():
        return str(int(value))
    return safe_path_component(value, fallback="module")


def flashcard_output_path(
    flashcards_root: Path,
    course_code: object,
    module_number: object,
) -> Path:
    course = safe_path_component(course_code, fallback="course")
    module = module_file_label(module_number)
    return Path(flashcards_root) / course / f"{course}_M{module}.csv"
