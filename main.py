from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from flashcard_csv import render_module, write_module_output
from flashcard_pipeline import FlashcardPipeline, GenerationError, PipelineConfig
from graph_input import (
    GraphInputError,
    extract_graph_facts,
    load_graph,
    resolve_identity,
)
from local_qwen import DEFAULT_N_CTX, LocalQwenBackend, ensure_model


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


def run(args: argparse.Namespace) -> Path | None:
    graph = load_graph(args.graph)
    identity = resolve_identity(graph, args.course_code, args.module_number)
    facts = extract_graph_facts(graph)

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

    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(
            max_retries=args.max_retries,
            final_review=args.final_review,
        ),
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    clusters = pipeline.run(identity, facts)
    content = render_module(identity, clusters)
    output = args.output or Path("flashcards") / f"module_{identity.module_number}.txt"
    write_module_output(output, content)
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
