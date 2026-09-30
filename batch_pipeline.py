from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Callable, Sequence

from flashcard_pipeline import GenerationError
from graph_input import GraphInputError
from local_qwen import LocalQwenBackend, ensure_model
import main as flashcard_generator
from pipeline import (
    PipelinePaths,
    _valid_flashcards,
    _valid_graph,
    _valid_checked_graph,
    _valid_structured_text,
    stage_structured_module,
)
import text_extractor
from text_extractor import DEFAULT_MODEL, load_runtime, release_runtime


Loader = Callable[[argparse.Namespace], AbstractContextManager[object]]
Stage = Callable[["BatchItem", object], object]
ProgressCallback = Callable[["BatchProgressEvent"], None]


@dataclass(frozen=True)
class BatchItem:
    filename: str
    args: argparse.Namespace
    paths: PipelinePaths


@dataclass(frozen=True)
class BatchResult:
    outputs: tuple[Path, ...]
    errors: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class BatchProgressEvent:
    """One observable state transition for a module in a batch."""

    filename: str
    module_number: str
    state: str
    stage: str
    progress_percent: int
    message: str
    error: str | None = None


@dataclass(frozen=True)
class BatchDependencies:
    qwen_loader: Loader
    rebel_loader: Loader
    ingest_stage: Stage
    graph_stage: Stage
    flashcard_stage: Stage
    monotonic: Callable[[], float]


@dataclass
class _BatchState:
    item: BatchItem
    needs_ingest: bool
    needs_graph: bool
    needs_flashcards: bool
    remaining_seconds: float | None
    active: bool = True
    error: str | None = None
    stage: str = "waiting"
    progress_percent: int = 0


_STAGE_ERRORS = (
    OSError,
    GraphInputError,
    GenerationError,
    RuntimeError,
    ValueError,
)

_SHARED_CONFIGURATION = (
    ("--model-dir", "model_dir"),
    ("--n-ctx", "n_ctx"),
    ("--n-gpu-layers", "n_gpu_layers"),
    ("--seed", "seed"),
    ("--kg-device", "kg_device"),
    ("--kg-batch-size", "kg_batch_size"),
    ("--kg-num-beams", "kg_num_beams"),
)

_REUSE_MANIFEST_NAME = "batch_reuse_manifest.json"


def _manifest_path(item: BatchItem) -> Path:
    return item.paths.workspace / _REUSE_MANIFEST_NAME


def _source_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with Path(source).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_settings(args: argparse.Namespace) -> dict[str, object]:
    """Return every generation-affecting setting used by staged adapters."""
    return {
        "model_dir": str(Path(args.model_dir).resolve()),
        "attempts": args.attempts,
        "seed": args.seed,
        "n_gpu_layers": args.n_gpu_layers,
        "n_ctx": args.n_ctx,
        "kg_device": args.kg_device,
        "kg_batch_size": args.kg_batch_size,
        "kg_num_beams": args.kg_num_beams,
        "skip_final_review": args.skip_final_review,
        "graph_chunk_tokens": getattr(args, "chunk_tokens", 384),
        "graph_overlap_tokens": getattr(args, "overlap_tokens", 64),
        "graph_max_new_tokens": getattr(args, "max_new_tokens", 192),
        "graph_model": DEFAULT_MODEL,
        "graph_checker": "qwen-conservative-v1",
    }


def _manifest_contents(item: BatchItem) -> dict[str, object]:
    return {
        "source_sha256": _source_sha256(item.args.input),
        "course_code": str(item.args.course_code),
        "module_number": str(item.args.module_number),
        "settings": _manifest_settings(item.args),
    }


def _manifest_matches(item: BatchItem) -> bool:
    try:
        with _manifest_path(item).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest == _manifest_contents(item)
    except (OSError, ValueError, TypeError):
        return False


def _write_manifest(item: BatchItem) -> None:
    path = _manifest_path(item)
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
            json.dump(_manifest_contents(item), handle, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _invalidate(paths: Sequence[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise OSError(f"could not invalidate stale artifact {path}: {exc}") from exc


def _invalidate_for_recomputation(state: _BatchState) -> None:
    paths = state.item.paths
    manifest = _manifest_path(state.item)
    if state.needs_ingest:
        _invalidate(
            (
                paths.unchecked_graph_json,
                paths.unchecked_triples_csv,
                paths.graph_json,
                paths.triples_csv,
                paths.flashcards,
                manifest,
            )
        )
    elif state.needs_graph:
        _invalidate(
            (
                paths.unchecked_graph_json,
                paths.unchecked_triples_csv,
                paths.graph_json,
                paths.triples_csv,
                paths.flashcards,
                manifest,
            )
        )
    elif state.needs_flashcards:
        _invalidate((paths.flashcards, manifest))


def _configuration_value(args: argparse.Namespace, attribute: str):
    value = getattr(args, attribute)
    if attribute == "model_dir":
        return Path(value).resolve()
    return value


def _configuration_error(items: Sequence[BatchItem]) -> str | None:
    if not items:
        return None
    mismatches = []
    for option, attribute in _SHARED_CONFIGURATION:
        expected = _configuration_value(items[0].args, attribute)
        if any(
            _configuration_value(item.args, attribute) != expected
            for item in items[1:]
        ):
            mismatches.append(option)
    if not mismatches:
        return None
    return "batch items must share runtime configuration: " + ", ".join(mismatches)


@contextmanager
def _qwen_loader(args: argparse.Namespace):
    if getattr(args, "portable", False):
        model_path = ensure_model(args.model_dir, allow_download=False)
    else:
        model_path = ensure_model(args.model_dir)
    backend = LocalQwenBackend(
        model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )
    try:
        yield backend
    finally:
        backend.close()


@contextmanager
def _rebel_loader(args: argparse.Namespace):
    if getattr(args, "portable", False):
        runtime = load_runtime(
            args.rebel_model,
            args.kg_device,
            local_files_only=True,
        )
    else:
        runtime = load_runtime(DEFAULT_MODEL, args.kg_device)
    try:
        yield runtime
    finally:
        release_runtime(runtime)


def _ingest_stage(item: BatchItem, _runtime: object) -> Path:
    return stage_structured_module(
        item.args.input,
        item.paths.structured_text,
        course_code=item.args.course_code,
        module_number=item.args.module_number,
    )


def _graph_stage(item: BatchItem, runtime: object) -> tuple[Path, Path]:
    args = argparse.Namespace(**vars(item.args))
    args.input = item.paths.structured_text
    args.output_dir = item.paths.unchecked_graph_dir
    args.model = getattr(item.args, "rebel_model", DEFAULT_MODEL)
    args.chunk_tokens = getattr(args, "chunk_tokens", 384)
    args.overlap_tokens = getattr(args, "overlap_tokens", 64)
    args.batch_size = item.args.kg_batch_size
    args.num_beams = item.args.kg_num_beams
    args.max_new_tokens = getattr(args, "max_new_tokens", 192)
    args.device = item.args.kg_device
    return text_extractor.run(args, runtime=runtime)


def _flashcard_stage(item: BatchItem, backend: object) -> Path | None:
    args = argparse.Namespace(**vars(item.args))
    args.graph = item.paths.graph_json
    args.unchecked_graph = item.paths.unchecked_graph_json
    args.output = item.paths.flashcards
    args.course_corpus = item.paths.flashcards.parent / "course_corpus.json"
    args.max_retries = item.args.attempts
    args.final_review = not item.args.skip_final_review
    args.smoke_test = False
    progress = getattr(item.args, "_progress_callback", None)
    if progress is None:
        return flashcard_generator.run(args, backend=backend)
    return flashcard_generator.run(args, backend=backend, progress=progress)


PRODUCTION_DEPENDENCIES = BatchDependencies(
    qwen_loader=_qwen_loader,
    rebel_loader=_rebel_loader,
    ingest_stage=_ingest_stage,
    graph_stage=_graph_stage,
    flashcard_stage=_flashcard_stage,
    monotonic=time.monotonic,
)


def _make_state(item: BatchItem, timeout_seconds: float | None) -> _BatchState:
    args = item.args
    paths = item.paths
    reusable = _manifest_matches(item)
    needs_ingest = (
        args.force
        or not reusable
        or not _valid_structured_text(paths.structured_text)
    )
    has_graph_source = _valid_graph(paths.unchecked_graph_json) or _valid_checked_graph(
        paths.graph_json
    )
    needs_graph = needs_ingest or args.force or not has_graph_source
    needs_flashcards = (
        needs_graph
        or args.force
        or not _valid_checked_graph(paths.graph_json)
        or not _valid_flashcards(
            paths.flashcards, args.module_number, args.course_code
        )
    )
    if needs_ingest:
        stage, progress_percent = "ingest", 0
    elif needs_graph:
        stage, progress_percent = "graph", 10
    elif needs_flashcards:
        stage, progress_percent = "flashcards", 45
    else:
        stage, progress_percent = "complete", 100
    return _BatchState(
        item=item,
        needs_ingest=needs_ingest,
        needs_graph=needs_graph,
        needs_flashcards=needs_flashcards,
        remaining_seconds=timeout_seconds,
        stage=stage,
        progress_percent=progress_percent,
    )


def _notify_progress(
    callback: ProgressCallback | None,
    state: _BatchState,
    *,
    status: str,
    stage: str,
    progress_percent: int,
    message: str,
    error: str | None = None,
) -> None:
    state.stage = stage
    state.progress_percent = max(0, min(100, int(progress_percent)))
    if callback is None:
        return
    try:
        callback(
            BatchProgressEvent(
                filename=state.item.filename,
                module_number=str(state.item.args.module_number),
                state=status,
                stage=stage,
                progress_percent=state.progress_percent,
                message=message,
                error=error,
            )
        )
    except Exception:
        # Status reporting must never interrupt model work.
        return


def _flashcard_detail_progress(
    callback: ProgressCallback | None,
    state: _BatchState,
    message: str,
) -> None:
    """Publish useful generation detail without exposing full model payloads."""
    if "generated output:\n" in message or re.search(
        r"\] card \d+/\d+ generated:", message
    ):
        return
    percent = max(45, state.progress_percent)
    cluster_match = re.search(
        r"(?:Generating|Regenerating reviewed) cluster (\d+)/(\d+)",
        message,
    )
    if cluster_match:
        position = int(cluster_match.group(1))
        total = max(1, int(cluster_match.group(2)))
        percent = 50 + round(position / total * 45)
    elif message.startswith("Planning "):
        percent = 48
    elif message.startswith(("Grounding review ", "Global duplicate review")):
        percent = 96
    elif message.startswith("100 flashcards generated"):
        percent = 99
    compact_message = " ".join(str(message).split())[:500]
    if compact_message:
        _notify_progress(
            callback,
            state,
            status="processing",
            stage="flashcards",
            progress_percent=percent,
            message=compact_message,
        )


def _run_stage(
    states: list[_BatchState],
    *,
    needs_attribute: str,
    stage_name: str,
    loader: Loader,
    stage: Stage,
    validator: Callable[[BatchItem], bool],
    dependencies: BatchDependencies,
    timeout_seconds: float | None,
    progress: ProgressCallback | None,
) -> bool:
    pending = [
        state
        for state in states
        if state.active and getattr(state, needs_attribute)
    ]
    for state in pending:
        if state.remaining_seconds is not None and state.remaining_seconds <= 0:
            _fail(
                state,
                "module exceeded "
                f"{timeout_seconds:g}-second active-processing timeout",
                progress,
            )
    pending = [state for state in pending if state.active]
    if not pending:
        return True

    stage_start = {"graph": 10, "flashcards": 45}.get(stage_name, 0)
    stage_complete = {"graph": 45, "flashcards": 100}.get(stage_name, 100)
    first = pending[0]
    _notify_progress(
        progress,
        first,
        status="processing",
        stage=stage_name,
        progress_percent=stage_start,
        message=f"Loading resources for {stage_name}...",
    )
    try:
        with loader(pending[0].item.args) as runtime:
            for state in pending:
                _notify_progress(
                    progress,
                    state,
                    status="processing",
                    stage=stage_name,
                    progress_percent=stage_start,
                    message=f"Running {stage_name}...",
                )
                print(
                    f"[{state.item.filename}] Running {stage_name}...",
                    flush=True,
                )
                start = (
                    dependencies.monotonic()
                    if state.remaining_seconds is not None
                    else None
                )
                try:
                    if stage_name == "flashcards" and progress is not None:
                        state.item.args._progress_callback = lambda message, current=state: (
                            _flashcard_detail_progress(progress, current, message)
                        )
                    try:
                        stage_result = stage(state.item, runtime)
                    finally:
                        if hasattr(state.item.args, "_progress_callback"):
                            delattr(state.item.args, "_progress_callback")
                        if start is not None:
                            elapsed = dependencies.monotonic() - start
                            state.remaining_seconds -= elapsed
                    if (
                        state.remaining_seconds is not None
                        and state.remaining_seconds <= 0
                    ):
                        _fail(
                            state,
                            "module exceeded "
                            f"{timeout_seconds:g}-second active-processing timeout",
                            progress,
                        )
                        continue
                    if not validator(state.item):
                        raise RuntimeError(
                            f"{stage_name} stage did not create a valid artifact: "
                            f"{_stage_output(state.item, stage_name)}"
                        )
                    print(
                        f"[{state.item.filename}] {stage_name} complete.",
                        flush=True,
                        )
                    print(
                        f"[{state.item.filename}] {stage_name} generated output: "
                        f"{stage_result!r}",
                        flush=True,
                    )
                    if stage_name == "flashcards":
                        status, next_stage, message = (
                            "completed",
                            "complete",
                            "Module processing complete.",
                        )
                    else:
                        status, next_stage, message = (
                            "queued",
                            "flashcards",
                            "Waiting for flashcard generation.",
                        )
                    _notify_progress(
                        progress,
                        state,
                        status=status,
                        stage=next_stage,
                        progress_percent=stage_complete,
                        message=message,
                    )
                except _STAGE_ERRORS as exc:
                    print(
                        f"[{state.item.filename}] {stage_name} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _fail(state, str(exc), progress)
                    print(
                        f"[{state.item.filename}] {stage_name} failed: {exc}",
                        flush=True,
                    )
    except _STAGE_ERRORS as exc:
        for state in states:
            if state.active and _has_remaining_work(state):
                _fail(state, str(exc), progress)
        return False
    return True


def _fail(
    state: _BatchState,
    error: str,
    progress: ProgressCallback | None = None,
) -> None:
    state.active = False
    state.error = error
    _notify_progress(
        progress,
        state,
        status="failed",
        stage=state.stage,
        progress_percent=state.progress_percent,
        message=str(error),
        error=str(error),
    )


def _has_remaining_work(state: _BatchState) -> bool:
    return state.needs_ingest or state.needs_graph or state.needs_flashcards


def _stage_output(item: BatchItem, stage_name: str) -> Path:
    if stage_name == "ingest":
        return item.paths.structured_text
    if stage_name == "graph":
        return item.paths.unchecked_graph_json
    return item.paths.flashcards


def _result(states: Sequence[_BatchState]) -> BatchResult:
    outputs = tuple(
        state.item.paths.flashcards
        for state in states
        if state.active
        and _valid_checked_graph(state.item.paths.graph_json)
        and _valid_flashcards(
            state.item.paths.flashcards,
            state.item.args.module_number,
            state.item.args.course_code,
        )
    )
    errors = tuple(
        {"file": state.item.filename, "error": state.error}
        for state in states
        if state.error is not None
    )
    return BatchResult(outputs=outputs, errors=errors)


def _write_completed_manifests(
    states: Sequence[_BatchState],
    progress: ProgressCallback | None = None,
) -> None:
    for state in states:
        if not state.active or not _has_remaining_work(state):
            continue
        try:
            _write_manifest(state.item)
        except OSError as exc:
            _fail(state, str(exc), progress)


def run_batch(
    items: Sequence[BatchItem],
    *,
    dependencies: BatchDependencies = PRODUCTION_DEPENDENCIES,
    timeout_seconds: float | None = None,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    items = tuple(items)
    if timeout_seconds is not None:
        if timeout_seconds < 0:
            for item in items:
                state = _BatchState(item, False, False, False, timeout_seconds)
                _fail(state, "module timeout must not be negative", progress)
            return BatchResult(
                outputs=(),
                errors=tuple(
                    {
                        "file": item.filename,
                        "error": "module timeout must not be negative",
                    }
                    for item in items
                ),
            )
        if timeout_seconds == 0:
            timeout_seconds = None
    configuration_error = _configuration_error(items)
    if configuration_error is not None:
        for item in items:
            state = _BatchState(item, False, False, False, timeout_seconds)
            _fail(state, configuration_error, progress)
        return BatchResult(
            outputs=(),
            errors=tuple(
                {"file": item.filename, "error": configuration_error}
                for item in items
            ),
        )

    states = [_make_state(item, timeout_seconds) for item in items]
    for state in states:
        if state.stage == "complete":
            _notify_progress(
                progress,
                state,
                status="completed",
                stage="complete",
                progress_percent=100,
                message="Reused completed module artifacts.",
            )
        else:
            _notify_progress(
                progress,
                state,
                status="queued",
                stage=state.stage,
                progress_percent=state.progress_percent,
                message=f"Waiting for {state.stage}.",
            )
    for state in states:
        try:
            _invalidate_for_recomputation(state)
        except OSError as exc:
            _fail(state, str(exc), progress)
    for state in states:
        if not state.active or not state.needs_ingest:
            continue
        _notify_progress(
            progress,
            state,
            status="processing",
            stage="ingest",
            progress_percent=0,
            message="Staging structured input...",
        )
        print(f"[{state.item.filename}] Staging structured input...", flush=True)
        try:
            stage_result = dependencies.ingest_stage(state.item, None)
            if not _valid_structured_text(state.item.paths.structured_text):
                raise RuntimeError(
                    "ingest stage did not create a valid artifact: "
                    f"{state.item.paths.structured_text}"
                )
            print(
                f"[{state.item.filename}] Structured input staged: "
                f"{stage_result!r}",
                flush=True,
            )
            _notify_progress(
                progress,
                state,
                status="queued",
                stage="graph",
                progress_percent=10,
                message="Waiting for knowledge graph generation.",
            )
        except _STAGE_ERRORS as exc:
            print(
                f"[{state.item.filename}] ingest failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            _fail(state, str(exc), progress)
    if not _run_stage(
        states,
        needs_attribute="needs_graph",
        stage_name="graph",
        loader=dependencies.rebel_loader,
        stage=dependencies.graph_stage,
        validator=lambda item: _valid_graph(item.paths.unchecked_graph_json),
        dependencies=dependencies,
        timeout_seconds=timeout_seconds,
        progress=progress,
    ):
        return _result(states)
    if not _run_stage(
        states,
        needs_attribute="needs_flashcards",
        stage_name="flashcards",
        loader=dependencies.qwen_loader,
        stage=dependencies.flashcard_stage,
        validator=lambda item: _valid_flashcards(
            item.paths.flashcards,
            item.args.module_number,
            item.args.course_code,
        ),
        dependencies=dependencies,
        timeout_seconds=timeout_seconds,
        progress=progress,
    ):
        return _result(states)
    _write_completed_manifests(states, progress)
    return _result(states)
