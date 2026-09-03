from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Sequence

from graph_input import GraphInputError, extract_graph_facts, load_graph
from structured_module import graph_ready_text, parse_module_metadata


PROJECT_DIR = Path(__file__).resolve().parent


class PipelineRunError(RuntimeError):
    """Raised when an end-to-end stage fails or omits its promised artifact."""


@dataclass(frozen=True)
class PipelinePaths:
    workspace: Path
    structured_text: Path
    graph_dir: Path
    graph_json: Path
    triples_csv: Path
    flashcards: Path


@dataclass(frozen=True)
class StageCommand:
    name: str
    command: tuple[str, ...]
    expected_output: Path


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem.strip()).strip("._-")
    return stem or "module"


def pipeline_paths(pdf: Path, output_root: Path) -> PipelinePaths:
    workspace = Path(output_root) / _safe_stem(Path(pdf))
    graph_dir = workspace / "knowledge_graph"
    return PipelinePaths(
        workspace=workspace,
        structured_text=workspace / "structured_module.txt",
        graph_dir=graph_dir,
        graph_json=graph_dir / "knowledge_graph.json",
        triples_csv=graph_dir / "triples.csv",
        flashcards=workspace / "flashcards.txt",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PDF slides -> structured TXT -> knowledge graph -> validated "
            "flashcards as a resumable local sequence."
        ),
        epilog=(
            "Each per-PDF workspace contains structured_module.txt, "
            "knowledge_graph/knowledge_graph.json, and flashcards.txt. "
            "Valid completed stages are reused when the sequence resumes."
        ),
    )
    parser.add_argument("pdf", type=Path, help="source slide PDF")
    parser.add_argument("--course-code", required=True, help="exact course code")
    parser.add_argument("--module-number", required=True, help="exact module number")
    parser.add_argument("--module-title", help="exact module title")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pipeline_output"),
        help="root for per-PDF stage artifacts (default: pipeline_output)",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--ocr-min-chars", type=int, default=40)
    parser.add_argument("--ocr-dpi", type=int, default=200)
    parser.add_argument(
        "--kg-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="REBEL device (default: auto)",
    )
    parser.add_argument(
        "--skip-final-review",
        action="store_true",
        help="skip Qwen's final flashcard review calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute all stages instead of resuming from valid artifacts",
    )
    return parser.parse_args(argv)


def build_stage_commands(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> tuple[StageCommand, ...]:
    pdf = str(Path(args.pdf).resolve())
    model_dir = str(Path(args.model_dir).resolve())
    stage_one = [
        sys.executable,
        str((PROJECT_DIR / "slides_pdf_to_txt.py").resolve()),
        pdf,
        "--output",
        str(paths.structured_text.resolve()),
        "--course-code",
        args.course_code,
        "--module-number",
        args.module_number,
        "--model-dir",
        model_dir,
        "--attempts",
        str(args.attempts),
        "--seed",
        str(args.seed),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--n-ctx",
        str(args.n_ctx),
        "--ocr-min-chars",
        str(args.ocr_min_chars),
        "--ocr-dpi",
        str(args.ocr_dpi),
    ]
    if args.module_title:
        stage_one.extend(("--module-title", args.module_title))

    stage_two = [
        sys.executable,
        str((PROJECT_DIR / "text-extractor.py").resolve()),
        str(paths.structured_text.resolve()),
        "--output-dir",
        str(paths.graph_dir.resolve()),
        "--device",
        args.kg_device,
    ]
    stage_three = [
        sys.executable,
        str((PROJECT_DIR / "main.py").resolve()),
        str(paths.graph_json.resolve()),
        "--course-code",
        args.course_code,
        "--module-number",
        args.module_number,
        "--output",
        str(paths.flashcards.resolve()),
        "--model-dir",
        model_dir,
        "--max-retries",
        str(args.attempts),
        "--seed",
        str(args.seed),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
        "--n-ctx",
        str(args.n_ctx),
    ]
    if args.skip_final_review:
        stage_three.append("--skip-final-review")

    return (
        StageCommand("pdf-to-text", tuple(stage_one), paths.structured_text),
        StageCommand("knowledge-graph", tuple(stage_two), paths.graph_json),
        StageCommand("flashcards", tuple(stage_three), paths.flashcards),
    )


def _valid_structured_text(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    metadata = parse_module_metadata(text)
    return metadata.get("format_version") == "1" and bool(graph_ready_text(text).strip())


def _valid_graph(path: Path) -> bool:
    try:
        graph = load_graph(path)
        extract_graph_facts(graph)
    except (GraphInputError, OSError, ValueError):
        return False
    return True


def _valid_flashcards(path: Path, module_number: str) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    return (
        f"Module {module_number}.1" in text
        and f"Module {module_number}.2" in text
    )


def run(
    args: argparse.Namespace,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> PipelinePaths:
    source = Path(args.pdf)
    if not source.is_file():
        raise PipelineRunError(f"PDF not found: {source}")
    if source.suffix.casefold() != ".pdf":
        raise PipelineRunError(f"input must be a PDF file: {source}")
    if args.attempts < 1:
        raise PipelineRunError("--attempts must be at least 1")

    paths = pipeline_paths(source, args.output_root)
    paths.workspace.mkdir(parents=True, exist_ok=True)
    commands = build_stage_commands(args, paths)
    validators = {
        "pdf-to-text": _valid_structured_text,
        "knowledge-graph": _valid_graph,
        "flashcards": lambda path: _valid_flashcards(path, args.module_number),
    }
    upstream_recomputed = False
    for stage in commands:
        is_valid = validators[stage.name](stage.expected_output)
        if not args.force and not upstream_recomputed and is_valid:
            print(f"Reusing {stage.name}: {stage.expected_output.resolve()}")
            continue

        print(f"Running {stage.name}...", flush=True)
        try:
            command_runner(
                list(stage.command),
                check=True,
                cwd=str(PROJECT_DIR),
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise PipelineRunError(f"{stage.name} stage failed: {exc}") from exc
        if not validators[stage.name](stage.expected_output):
            raise PipelineRunError(
                f"{stage.name} stage did not create a valid artifact: "
                f"{stage.expected_output}"
            )
        upstream_recomputed = True

    return paths


def main(argv: Sequence[str] | None = None) -> int:
    try:
        paths = run(parse_args(argv))
    except (PipelineRunError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Structured TXT: {paths.structured_text.resolve()}")
    print(f"Knowledge graph: {paths.graph_json.resolve()}")
    print(f"Flashcards: {paths.flashcards.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
