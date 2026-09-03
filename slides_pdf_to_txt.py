from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

from local_qwen import LocalQwenBackend, ensure_model
from pdf_ingestion import PdfExtractionError, extract_pdf_pages
from slide_normalizer import SlideNormalizationError, normalize_document
from structured_module import render_structured_module


def _safe_stem(path: Path) -> str:
    stem = re.sub(r'[<>:"/\\|?*]+', "_", path.stem.strip()).rstrip(" .")
    return stem or "module"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a slide PDF into validated structured UTF-8 text using "
            "local extraction, Tesseract OCR fallback, and local Qwen."
        )
    )
    parser.add_argument("pdf", type=Path, help="source slide PDF")
    parser.add_argument("--course-code", help="exact course code")
    parser.add_argument("--module-number", help="exact module number")
    parser.add_argument("--module-title", help="exact module title")
    parser.add_argument("--output", type=Path, help="structured TXT output path")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="directory containing the Qwen GGUF model (default: models)",
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--ocr-min-chars", type=int, default=40)
    parser.add_argument("--ocr-dpi", type=int, default=200)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    if args.max_tokens < 256:
        raise ValueError("--max-tokens must be at least 256")
    if args.n_ctx < 2048:
        raise ValueError("--n-ctx must be at least 2048")
    if args.ocr_min_chars < 1:
        raise ValueError("--ocr-min-chars must be at least 1")
    if args.ocr_dpi < 72:
        raise ValueError("--ocr-dpi must be at least 72")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def run(args: argparse.Namespace) -> Path:
    _validate_args(args)
    pages = extract_pdf_pages(
        args.pdf,
        min_chars=args.ocr_min_chars,
        dpi=args.ocr_dpi,
    )
    print(f"Extracted {len(pages)} slide(s) locally.", file=sys.stderr, flush=True)

    model_path = ensure_model(args.model_dir)
    backend = LocalQwenBackend(
        model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )
    module = normalize_document(
        backend,
        pages,
        source_file=args.pdf.name,
        course_code=args.course_code,
        module_number=args.module_number,
        module_title=args.module_title,
        attempts=args.attempts,
        max_tokens=args.max_tokens,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    content = render_structured_module(module)
    output = args.output or Path("structured_text") / f"{_safe_stem(args.pdf)}.txt"
    _atomic_write(output, content)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output = run(args)
    except (
        OSError,
        PdfExtractionError,
        SlideNormalizationError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved structured module: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
