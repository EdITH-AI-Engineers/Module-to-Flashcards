"""Resolve module PDFs from course folders without trusting upload paths."""
from pathlib import Path, PurePosixPath
import re


def pdf_identity(filename: str, fallback_course: str = "") -> tuple[str, str]:
    path = PurePosixPath(str(filename).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError("PDF uploads must use relative course-folder paths.")
    if path.suffix.casefold() != ".pdf":
        raise ValueError("Expected a module PDF.")
    named = re.fullmatch(r"(.+?)[-_]M(\d+)", path.stem, re.I)
    course = path.parent.name if len(path.parts) > 1 else named[1] if named else fallback_course.strip()
    if not course:
        raise ValueError("Choose a course folder or enter a course code for individual PDFs.")
    module = re.search(r"(?:^|[\s_.-])(?:module|mod|m)[\s_.-]*(\d+)(?=$|[\s_.-])", path.stem, re.I)
    if module is None:
        module = re.match(r"^(\d+)(?=$|[\s_.-])", path.stem)
    if module is None:
        raise ValueError(f"Cannot find the module number in {path.name}. Use a name such as Module 1.pdf or M1.pdf.")
    return course, module[1]


def course_folder_pdfs(root: Path) -> tuple[Path, ...]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Course folder not found: {root}")
    files = []
    for path in root.rglob("*"):
        if path.suffix.casefold() != ".pdf" or not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            continue
        files.append(path)
    if not files:
        raise ValueError("No module PDFs were found in the selected folder.")
    return tuple(sorted(files, key=lambda path: [int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(path.relative_to(root)))]))
