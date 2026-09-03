from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Sequence

from flashcard_pipeline import GenerationError
from graph_input import GraphInputError
from local_qwen import LocalQwenBackend, ensure_model
import main as flashcard_generator
from pdf_ingestion import PdfExtractionError
from pipeline import (
    PipelinePaths,
    _valid_flashcards,
    _valid_graph,
    _valid_structured_text,
)
from slide_normalizer import SlideNormalizationError
import slides_pdf_to_txt
import text_extractor
from text_extractor import DEFAULT_MODEL, load_runtime, release_runtime


Loader = Callable[[argparse.Namespace], AbstractContextManager[object]]
Stage = Callable[["BatchItem", object], object]


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
class BatchDependencies:
    qwen_loader: Loader
    rebel_loader: Loader
    normalize_stage: Stage
    graph_stage: Stage
    flashcard_stage: Stage
    monotonic: Callable[[], float]


@dataclass
class _BatchState:
    item: BatchItem
    needs_normalize: bool
    needs_graph: bool
    needs_flashcards: bool
    remaining_seconds: float | None
    active: bool = True


_STAGE_ERRORS = (
    OSError,
    PdfExtractionError,
    SlideNormalizationError,
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
    runtime = load_runtime(DEFAULT_MODEL, args.kg_device)
    try:
        yield runtime
    finally:
        release_runtime(runtime)


def _normalize_stage(item: BatchItem, backend: object) -> Path:
    args = argparse.Namespace(**vars(item.args))
    args.output = item.paths.structured_text
    args.max_tokens = getattr(args, "max_tokens", 2048)
    return slides_pdf_to_txt.run(args, backend=backend)


def _graph_stage(item: BatchItem, runtime: object) -> tuple[Path, Path]:
    args = argparse.Namespace(**vars(item.args))
    args.input = item.paths.structured_text
    args.output_dir = item.paths.graph_dir
    args.model = DEFAULT_MODEL
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
    args.output = item.paths.flashcards
    args.max_retries = item.args.attempts
    args.final_review = not item.args.skip_final_review
    args.smoke_test = False
    return flashcard_generator.run(args, backend=backend)


PRODUCTION_DEPENDENCIES = BatchDependencies(
    qwen_loader=_qwen_loader,
    rebel_loader=_rebel_loader,
    normalize_stage=_normalize_stage,
    graph_stage=_graph_stage,
    flashcard_stage=_flashcard_stage,
    monotonic=time.monotonic,
)


def _make_state(item: BatchItem, timeout_seconds: float | None) -> _BatchState:
    args = item.args
    paths = item.paths
    needs_normalize = args.force or not _valid_structured_text(paths.structured_text)
    needs_graph = needs_normalize or args.force or not _valid_graph(paths.graph_json)
    needs_flashcards = (
        needs_graph
        or args.force
        or not _valid_flashcards(paths.flashcards, args.module_number)
    )
    return _BatchState(
        item=item,
        needs_normalize=needs_normalize,
        needs_graph=needs_graph,
        needs_flashcards=needs_flashcards,
        remaining_seconds=timeout_seconds,
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
    errors: list[dict[str, str]],
) -> None:
    pending = [
        state
        for state in states
        if state.active and getattr(state, needs_attribute)
    ]
    for state in pending:
        if state.remaining_seconds is not None and state.remaining_seconds <= 0:
            state.active = False
            errors.append(
                {
                    "pdf": state.item.filename,
                    "error": (
                        "module exceeded "
                        f"{timeout_seconds:g}-second active-processing timeout"
                    ),
                }
            )
    pending = [state for state in pending if state.active]
    if not pending:
        return

    try:
        with loader(pending[0].item.args) as runtime:
            for state in pending:
                start = (
                    dependencies.monotonic()
                    if state.remaining_seconds is not None
                    else None
                )
                try:
                    stage(state.item, runtime)
                    if not validator(state.item):
                        raise RuntimeError(
                            f"{stage_name} stage did not create a valid artifact: "
                            f"{_stage_output(state.item, stage_name)}"
                        )
                except _STAGE_ERRORS as exc:
                    state.active = False
                    errors.append(
                        {"pdf": state.item.filename, "error": str(exc)}
                    )
                finally:
                    if start is not None:
                        elapsed = dependencies.monotonic() - start
                        state.remaining_seconds -= elapsed
    except _STAGE_ERRORS as exc:
        for state in pending:
            if state.active:
                state.active = False
                errors.append({"pdf": state.item.filename, "error": str(exc)})


def _stage_output(item: BatchItem, stage_name: str) -> Path:
    if stage_name == "normalize":
        return item.paths.structured_text
    if stage_name == "graph":
        return item.paths.graph_json
    return item.paths.flashcards


def run_batch(
    items: Sequence[BatchItem],
    *,
    dependencies: BatchDependencies = PRODUCTION_DEPENDENCIES,
    timeout_seconds: float | None = None,
) -> BatchResult:
    items = tuple(items)
    configuration_error = _configuration_error(items)
    if configuration_error is not None:
        return BatchResult(
            outputs=(),
            errors=tuple(
                {"pdf": item.filename, "error": configuration_error}
                for item in items
            ),
        )

    states = [_make_state(item, timeout_seconds) for item in items]
    errors: list[dict[str, str]] = []

    _run_stage(
        states,
        needs_attribute="needs_normalize",
        stage_name="normalize",
        loader=dependencies.qwen_loader,
        stage=dependencies.normalize_stage,
        validator=lambda item: _valid_structured_text(item.paths.structured_text),
        dependencies=dependencies,
        timeout_seconds=timeout_seconds,
        errors=errors,
    )
    _run_stage(
        states,
        needs_attribute="needs_graph",
        stage_name="graph",
        loader=dependencies.rebel_loader,
        stage=dependencies.graph_stage,
        validator=lambda item: _valid_graph(item.paths.graph_json),
        dependencies=dependencies,
        timeout_seconds=timeout_seconds,
        errors=errors,
    )
    _run_stage(
        states,
        needs_attribute="needs_flashcards",
        stage_name="flashcards",
        loader=dependencies.qwen_loader,
        stage=dependencies.flashcard_stage,
        validator=lambda item: _valid_flashcards(
            item.paths.flashcards, item.args.module_number
        ),
        dependencies=dependencies,
        timeout_seconds=timeout_seconds,
        errors=errors,
    )

    outputs = tuple(
        state.item.paths.flashcards
        for state in states
        if state.active
        and _valid_flashcards(
            state.item.paths.flashcards, state.item.args.module_number
        )
    )
    return BatchResult(outputs=outputs, errors=tuple(errors))
