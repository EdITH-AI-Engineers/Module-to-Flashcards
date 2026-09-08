from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from flashcard_csv import render_module, write_module_output
from flashcard_pipeline import FlashcardPipeline, GenerationError, PipelineConfig
from flashcard_types import ChatBackend
from graph_input import (
    GraphInputError,
    extract_graph_facts,
    load_graph,
    resolve_identity,
)
from local_qwen import DEFAULT_N_CTX, LocalQwenBackend, ensure_model


def load_course_corpus(course_dir: Path) -> tuple[list[str], list[str]]:
    path = Path(course_dir) / "course_corpus.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return [], []
        concepts = value.get("concept_names", [])
        questions = value.get("questions", [])
        if not isinstance(concepts, list) or not isinstance(questions, list):
            return [], []
        return (
            [item for item in concepts if isinstance(item, str)],
            [item for item in questions if isinstance(item, str)],
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return [], []


def append_course_corpus(
    course_dir: Path,
    concept_names: Sequence[str],
    questions: Sequence[str],
) -> None:
    course_dir = Path(course_dir)
    existing_concepts, existing_questions = load_course_corpus(course_dir)
    merged_concepts = list(dict.fromkeys((*existing_concepts, *concept_names)))
    merged_questions = list(dict.fromkeys((*existing_questions, *questions)))
    path = course_dir / "course_corpus.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                {
                    "concept_names": merged_concepts,
                    "questions": merged_questions,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate validated assessment CSV from a knowledge graph with a "
            "local Qwen2.5 3B Q5_K_M model."
        )
    )
    parser.add_argument("graph", type=Path, help="knowledge_graph.json path")
    parser.add_argument("--course-code", help="exact course code")
    parser.add_argument("--module-number", help="exact module number")
    parser.add_argument("--output", type=Path, help="output text file")
    parser.add_argument(
        "--course-corpus",
        type=Path,
        help="per-course corpus JSON path for cross-module deduplication",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="download directory for the GGUF model (default: models)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="maximum attempts for each invalid response (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="local inference seed (default: 42)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="layers requested for GPU offload; -1 requests all (default: -1)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=DEFAULT_N_CTX,
        help=f"model context window (default: {DEFAULT_N_CTX})",
    )
    parser.add_argument(
        "--skip-final-review",
        action="store_false",
        dest="final_review",
        help="skip five grounding reviews and the global duplicate review",
    )
    parser.set_defaults(final_review=True)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="load the model and run a tiny JSON response check only",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace, *, backend: ChatBackend | None = None) -> Path | None:
    graph = load_graph(args.graph)
    identity = resolve_identity(graph, args.course_code, args.module_number)
    facts = extract_graph_facts(graph)

    if backend is None:
        try:
            model_path = ensure_model(args.model_dir)
            backend = LocalQwenBackend(
                model_path,
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                seed=args.seed,
            )
        except Exception as exc:
            raise RuntimeError(f"local Qwen setup failed: {exc}") from exc

    if args.smoke_test:
        raw = backend.complete(
            "Return JSON only.",
            'Return exactly this JSON object: {"status":"ok"}',
            max_tokens=32,
        )
        try:
            smoke_value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen smoke test did not return valid JSON") from exc
        if smoke_value != {"status": "ok"}:
            raise RuntimeError("Qwen smoke test returned an unexpected JSON value")
        return None

    output = args.output or Path("flashcards") / f"module_{identity.module_number}.txt"
    course_dir = (
        Path(args.course_corpus).parent
        if getattr(args, "course_corpus", None)
        else output.parent.parent
    )
    prior_concept_names, prior_questions = load_course_corpus(course_dir)
    print(
        f"[flashcard-pipeline] loaded course corpus from "
        f"{(course_dir / 'course_corpus.json').resolve()}: "
        f"{len(prior_concept_names)} concepts, {len(prior_questions)} questions",
        file=sys.stderr,
        flush=True,
    )
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(
            max_retries=args.max_retries,
            final_review=args.final_review,
        ),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    clusters = pipeline.run(
        identity,
        facts,
        prior_concept_names=prior_concept_names,
        prior_questions=prior_questions,
    )
    content = render_module(identity, clusters)
    write_module_output(output, content)
    print(
        f"[flashcard-pipeline] generated module output: {output.resolve()}\n{content}",
        file=sys.stderr,
        flush=True,
    )
    append_course_corpus(
        course_dir,
        [cluster.concept.name for cluster in clusters],
        [card.question for cluster in clusters for card in cluster.cards],
    )
    updated_concepts, updated_questions = load_course_corpus(course_dir)
    print(
        f"[flashcard-pipeline] updated course corpus: "
        f"{len(updated_concepts)} concepts, {len(updated_questions)} questions",
        file=sys.stderr,
        flush=True,
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = run(args)
    except (GraphInputError, GenerationError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if output is None:
        print("Qwen smoke test passed.")
    else:
        print(f"Saved flashcards: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
