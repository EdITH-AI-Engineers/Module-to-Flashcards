"""Benchmark Qwen flashcard generation on fixed graph inputs.

Run with ``python -m benchmarks.run --manifest benchmarks/modules.json``.
The manifest's ``modules`` array contains graph paths, course codes, and module
numbers. Sample text and graph contents stay outside the committed benchmark
reports; reports carry SHA-256 hashes so runs can be compared safely.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Sequence

from artifact_paths import flashcard_output_path
from benchmarks.metrics import summarize_metrics
from flashcard_contract import CLUSTERS_PER_MODULE
from flashcard_csv import render_module_parts, write_module_parts
from flashcard_pipeline import FlashcardPipeline, PipelineConfig
from flashcard_types import ModuleIdentity
from graph_input import extract_graph_facts, load_graph
from local_qwen import DEFAULT_N_CTX, LocalQwenBackend, ensure_model
from worker_budget import parse_cluster_workers


@dataclass(frozen=True)
class BenchmarkCase:
    graph: Path
    identity: ModuleIdentity


def load_cases(manifest: Path, count: int) -> tuple[BenchmarkCase, ...]:
    if count < 1:
        raise ValueError("--modules must be at least one")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = data.get("modules") if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) < count:
        raise ValueError(f"benchmark manifest needs at least {count} modules")
    cases = []
    for entry in entries[:count]:
        if not isinstance(entry, dict):
            raise ValueError("every benchmark module must be an object")
        graph = Path(str(entry["graph"]))
        if not graph.is_absolute():
            graph = manifest.parent / graph
        if not graph.is_file():
            raise FileNotFoundError(f"benchmark graph not found: {graph}")
        cases.append(
            BenchmarkCase(
                graph=graph.resolve(),
                identity=ModuleIdentity(
                    str(entry["course_code"]), str(entry["module_number"])
                ),
            )
        )
    return tuple(cases)


def _merge_task_counts(
    target: dict[str, Counter[str]], source: dict[str, dict[str, int]]
) -> None:
    for task, counts in source.items():
        target.setdefault(task, Counter()).update(counts)


def benchmark(
    cases: Sequence[BenchmarkCase],
    *,
    model_dir: Path,
    seed: int = 42,
    n_ctx: int = DEFAULT_N_CTX,
    n_gpu_layers: int = -1,
    workers: int | str = 1,
    max_retries: int = 3,
    final_review: bool = True,
) -> dict[str, object]:
    model_path = ensure_model(model_dir, allow_download=False)
    events: list[dict[str, object]] = []
    event_lock = Lock()
    task_counts: dict[str, Counter[str]] = {}
    module_results: list[dict[str, object]] = []
    prior_by_course: dict[str, tuple[list[str], list[str]]] = {}

    def collect(event: dict[str, object]) -> None:
        with event_lock:
            events.append(dict(event))

    with TemporaryDirectory(prefix="flashcard-benchmark-") as temporary:
        output_root = Path(temporary)
        for index, case in enumerate(cases, start=1):
            graph_bytes = case.graph.read_bytes()
            facts = extract_graph_facts(load_graph(case.graph))
            backend = LocalQwenBackend(
                model_path,
                seed=seed,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
            )
            backend.set_metric_sink(collect)
            actual_workers = (
                backend.auto_cluster_workers if workers == "auto" else int(workers)
            )

            def report_progress(message: str) -> None:
                if message.startswith("Completed cluster "):
                    print(f"Benchmark module {index}: {message.split(':', 1)[0]}", flush=True)
                elif message.startswith("Cluster generation:"):
                    print(f"Benchmark module {index}: {message}", flush=True)

            pipeline = FlashcardPipeline(
                backend,
                PipelineConfig(
                    max_retries=max_retries,
                    final_review=final_review,
                    cluster_workers=actual_workers,
                ),
                progress=report_progress,
            )
            prior_concepts, prior_questions = prior_by_course.setdefault(
                case.identity.course_code, ([], [])
            )
            print(
                f"Benchmark module {index}/{len(cases)}: "
                f"{case.identity.course_code} M{case.identity.module_number}",
                flush=True,
            )
            started = perf_counter()
            try:
                clusters = pipeline.run(
                    case.identity,
                    facts,
                    prior_concept_names=prior_concepts,
                    prior_questions=prior_questions,
                )
                output = flashcard_output_path(
                    output_root / str(index),
                    case.identity.course_code,
                    case.identity.module_number,
                )
                write_module_parts(
                    output,
                    render_module_parts(case.identity, clusters),
                    case.identity,
                )
                prior_concepts.extend(cluster.concept.name for cluster in clusters)
                prior_questions.extend(
                    card.question for cluster in clusters for card in cluster.cards
                )
                result = "passed"
                error = None
                valid_clusters = CLUSTERS_PER_MODULE
            except Exception as exc:
                result = "failed"
                error = type(exc).__name__
                print(
                    f"Benchmark module {index} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                valid_clusters = 0
            finally:
                backend.close()
            elapsed = perf_counter() - started
            _merge_task_counts(task_counts, pipeline.task_metrics)
            module_results.append(
                {
                    "course_code": case.identity.course_code,
                    "module_number": case.identity.module_number,
                    "graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
                    "graph_facts": len(facts),
                    "result": result,
                    "error": error,
                    "valid_clusters": valid_clusters,
                    "duration_seconds": elapsed,
                    "cluster_generation_seconds": pipeline.cluster_generation_seconds,
                    "selected_workers": actual_workers,
                    "actual_workers": getattr(pipeline, "_actual_cluster_workers", None),
                }
            )

    elapsed_total = sum(float(item["duration_seconds"]) for item in module_results)
    valid_total = sum(int(item["valid_clusters"]) for item in module_results)
    cluster_times = [
        float(item["cluster_generation_seconds"])
        for item in module_results
        if item["cluster_generation_seconds"] is not None
    ]
    return {
        "format_version": 1,
        "model_filename": model_path.name,
        "settings": {
            "seed": seed,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "workers": workers,
            "max_retries": max_retries,
            "final_review": final_review,
        },
        "modules": module_results,
        "valid_clusters_per_minute": (
            60 * valid_total / elapsed_total if elapsed_total else None
        ),
        "cluster_generation_per_minute": (
            60 * valid_total / sum(cluster_times) if cluster_times else None
        ),
        "metrics": summarize_metrics(events, task_counts),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/modules.json"))
    parser.add_argument("--modules", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/baseline.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--workers", type=parse_cluster_workers, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--skip-final-review", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_cases(args.manifest, args.modules)
    report = benchmark(
        cases,
        model_dir=args.model_dir,
        seed=args.seed,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        workers=args.workers,
        max_retries=args.max_retries,
        final_review=not args.skip_final_review,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Benchmark report: {args.output.resolve()}")
    return 0 if all(item["result"] == "passed" for item in report["modules"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
