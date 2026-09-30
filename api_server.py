from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import os
import re
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
from typing import Annotated

from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from batch_pipeline import BatchItem, BatchProgressEvent, run_batch
from artifact_paths import module_file_label
from local_qwen import DEFAULT_N_CTX
from pipeline import pipeline_paths
from portable_paths import PortablePaths, build_paths
from structured_module import (
    parse_module_metadata,
    parse_slide_report_metadata,
    validate_slide_report,
)
from version import __version__


PROJECT_DIR = Path(__file__).resolve().parent
_RUNTIME_PATHS = build_paths(PROJECT_DIR, portable=False)
UPLOAD_DIR = _RUNTIME_PATHS.uploads
OUTPUT_ROOT = _RUNTIME_PATHS.outputs
MODEL_DIR = _RUNTIME_PATHS.models
REBEL_MODEL = _RUNTIME_PATHS.rebel_model
PORTABLE_MODE = _RUNTIME_PATHS.portable
QWEN_GPU_LAYERS = -1
KG_DEVICE = "auto"
_REQUEST_LOCK = threading.Lock()
_REQUEST_LOCK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="batch-request-lock"
)


@dataclass
class _ModuleStatus:
    filename: str
    course_code: str
    module_number: str | None = None
    state: str = "queued"
    stage: str = "waiting"
    progress_percent: int = 0
    message: str = "Waiting for an available pipeline slot."
    error: str | None = None
    output: str | None = None


@dataclass
class _RequestStatus:
    request_id: int
    modules: list[_ModuleStatus] = field(default_factory=list)
    finished: bool = False


class PipelineStatusRegistry:
    """Thread-safe, bounded status history for API pipeline work."""

    def __init__(self, *, history_limit: int = 10) -> None:
        self._lock = threading.Lock()
        self._requests: OrderedDict[int, _RequestStatus] = OrderedDict()
        self._next_request_id = 1
        self._history_limit = history_limit

    def enqueue(self, course_code: str, filenames: list[str]) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._requests[request_id] = _RequestStatus(
                request_id=request_id,
                modules=[
                    _ModuleStatus(filename=filename, course_code=course_code)
                    for filename in filenames
                ],
            )
            self._trim_history()
            return request_id

    def update(
        self,
        request_id: int,
        index: int,
        *,
        filename: str | None = None,
        module_number: str | None = None,
        state: str | None = None,
        stage: str | None = None,
        progress_percent: int | None = None,
        message: str | None = None,
        error: str | None = None,
        output: str | None = None,
    ) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or not 0 <= index < len(request.modules):
                return
            module = request.modules[index]
            if filename is not None:
                module.filename = str(filename)
            if module_number is not None:
                module.module_number = str(module_number)
            if state is not None:
                module.state = state
            if stage is not None:
                module.stage = stage
            if progress_percent is not None:
                module.progress_percent = max(0, min(100, int(progress_percent)))
            if message is not None:
                module.message = str(message)[:500]
            if error is not None:
                module.error = str(error)
            if output is not None:
                module.output = str(output)

    def apply_event(
        self,
        request_id: int,
        index: int,
        event: BatchProgressEvent,
    ) -> None:
        self.update(
            request_id,
            index,
            module_number=event.module_number,
            state=event.state,
            stage=event.stage,
            progress_percent=event.progress_percent,
            message=event.message,
            error=event.error,
        )

    def finish(self, request_id: int) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return
            for module in request.modules:
                if module.state in {"queued", "processing"}:
                    module.state = "completed"
                    module.stage = "complete"
                    module.progress_percent = 100
                    module.message = "Module processing complete."
            request.finished = True
            self._trim_history()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            modules = [
                self._module_snapshot(request.request_id, index, module)
                for request in self._requests.values()
                for index, module in enumerate(request.modules)
            ]
        active = next(
            (module for module in modules if module["state"] == "processing"),
            None,
        )
        queued = [module.copy() for module in modules if module["state"] == "queued"]
        for position, module in enumerate(queued, start=1):
            module["position"] = position
        counts = {
            state: sum(module["state"] == state for module in modules)
            for state in ("queued", "processing", "completed", "failed")
        }
        return {
            "status": "processing" if counts["queued"] or counts["processing"] else "idle",
            "active": active,
            "queue": queued,
            "modules": modules,
            "summary": {"total": len(modules), **counts},
        }

    def reset(self) -> None:
        """Clear process-local state; intended for application/test lifecycle use."""
        with self._lock:
            self._requests.clear()
            self._next_request_id = 1

    @staticmethod
    def _module_snapshot(
        request_id: int,
        index: int,
        module: _ModuleStatus,
    ) -> dict[str, object]:
        return {
            "requestId": request_id,
            "index": index,
            "courseCode": module.course_code,
            "filename": module.filename,
            "moduleNumber": module.module_number,
            "state": module.state,
            "stage": module.stage,
            "progressPercent": module.progress_percent,
            "message": module.message,
            "error": module.error,
            "output": module.output,
        }

    def _trim_history(self) -> None:
        completed = [
            request_id
            for request_id, request in self._requests.items()
            if request.finished
        ]
        for request_id in completed[: -self._history_limit]:
            self._requests.pop(request_id, None)


_PIPELINE_STATUS = PipelineStatusRegistry()
app = FastAPI(title="Module to Flashcards local processor", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def configure_api_storage(paths: PortablePaths) -> None:
    global PROJECT_DIR, _RUNTIME_PATHS, UPLOAD_DIR, OUTPUT_ROOT, MODEL_DIR
    global REBEL_MODEL, PORTABLE_MODE
    PROJECT_DIR = paths.root
    _RUNTIME_PATHS = paths
    UPLOAD_DIR = paths.uploads
    OUTPUT_ROOT = paths.outputs
    MODEL_DIR = paths.models
    REBEL_MODEL = paths.rebel_model
    PORTABLE_MODE = paths.portable


def configure_api_runtime(*, n_gpu_layers: int, kg_device: str) -> None:
    global QWEN_GPU_LAYERS, KG_DEVICE
    if kg_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported knowledge-graph device: {kg_device}")
    QWEN_GPU_LAYERS = int(n_gpu_layers)
    KG_DEVICE = kg_device


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("uploaded file must have a filename")
    if not name.casefold().endswith(".txt"):
        raise ValueError("only TXT module reports are accepted")
    return name


def safe_course_component(course_code: str) -> str:
    """Return one safe directory component without changing the public code."""
    value = str(course_code).strip()
    if not value or value in {".", ".."}:
        raise ValueError("course code must not be blank, '.' or '..'")
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not component or component in {".", ".."}:
        raise ValueError("course code does not contain a safe path component")
    return component


def course_upload_directory(course_code: str) -> Path:
    component = safe_course_component(course_code)
    root = UPLOAD_DIR.resolve()
    destination = (root / component).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("course code resolves outside the upload directory") from exc
    return destination


def _release_canceled_lock_acquire(future: Future[bool]) -> None:
    """Release a lock acquired after its waiting request was cancelled."""
    try:
        acquired = future.result()
    except Exception:
        return
    if acquired:
        _REQUEST_LOCK.release()


async def acquire_request_lock() -> None:
    """Acquire the process lock without consuming the default worker pool."""
    acquire_future = _REQUEST_LOCK_EXECUTOR.submit(_REQUEST_LOCK.acquire)
    try:
        acquired = await asyncio.shield(asyncio.wrap_future(acquire_future))
    except asyncio.CancelledError:
        acquire_future.add_done_callback(_release_canceled_lock_acquire)
        raise
    if not acquired:
        raise RuntimeError("could not acquire the batch request lock")


async def save_upload(upload: UploadFile, course_code: str) -> Path:
    filename = safe_filename(upload.filename or "")
    destination_dir = course_upload_directory(course_code)
    destination_dir.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=destination_dir, prefix=f".{filename}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := await upload.read(1024 * 1024):
                temporary.write(chunk)
        staged = temporary_path
        temporary_path = None
        return staged
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def find_module_number(module_text: str) -> str:
    """Find the module number in either supported TXT input format."""
    metadata = parse_module_metadata(module_text)
    module_number = metadata.get("module_number", "").strip()
    if module_number:
        return module_number

    report_number = parse_slide_report_metadata(module_text).get(
        "module_number", ""
    ).strip()
    if report_number:
        return report_number

    fallback = re.search(
        r"(?im)^\s*(?:module_number|module(?:\s+(?:number|no\.?))?)"
        r"\s*[:#-]\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*$",
        module_text,
    )
    if fallback:
        return fallback.group(1)
    raise ValueError("structured module must declare module_number")


def structured_module_number(source: Path, course_code: str) -> str:
    try:
        content = Path(source).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read structured module: {exc}") from exc
    metadata = parse_module_metadata(content)
    if metadata:
        if metadata.get("format_version") != "1":
            raise ValueError("structured module must declare format_version: 1")
        embedded_course = metadata.get("course_code")
        if embedded_course != course_code.strip():
            raise ValueError(
                "structured module course_code does not match the request: "
                f"{embedded_course or '(missing)'}"
            )
        return find_module_number(content)

    report_metadata = validate_slide_report(content, course_code=course_code)
    return report_metadata["module_number"]


def canonical_module_filename(course_code: str, module_number: str) -> str:
    """Build the canonical upload name from payload identity."""
    course = safe_course_component(course_code)
    module = module_file_label(module_number)
    return f"{course}_M{module}.txt"


def publish_module_upload(
    staged_source: Path,
    course_code: str,
    module_number: str,
) -> Path:
    """Atomically rename a validated staged upload to its canonical name."""
    source = Path(staged_source)
    destination = source.with_name(
        canonical_module_filename(course_code, module_number)
    )
    if source != destination:
        source.replace(destination)
    return destination


def pipeline_args(source: Path, course_code: str, module_number: str) -> Namespace:
    try:
        cluster_workers = int(os.environ.get("MODULE_FLASHCARDS_CLUSTER_WORKERS", "1"))
    except ValueError as exc:
        raise ValueError("MODULE_FLASHCARDS_CLUSTER_WORKERS must be 1 to 5") from exc
    if not 1 <= cluster_workers <= 5:
        raise ValueError("MODULE_FLASHCARDS_CLUSTER_WORKERS must be 1 to 5")
    return Namespace(
        input=source,
        course_code=course_code,
        module_number=module_number,
        output_root=OUTPUT_ROOT,
        model_dir=MODEL_DIR,
        rebel_model=REBEL_MODEL,
        portable=PORTABLE_MODE,
        attempts=3,
        seed=42,
        n_gpu_layers=QWEN_GPU_LAYERS,
        n_ctx=DEFAULT_N_CTX,
        cluster_workers=cluster_workers,
        timeout=0,
        kg_device=KG_DEVICE,
        kg_batch_size=1,
        kg_num_beams=1,
        skip_final_review=True,
        force=False,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/status")
def pipeline_status(response: Response) -> dict[str, object]:
    """Return the active module, waiting queue, and recent module states."""
    response.headers["Cache-Control"] = "no-store"
    return _PIPELINE_STATUS.snapshot()


@app.post("/process")
async def process_files(
    course_code: Annotated[
        str,
        Form(
            alias="courseCode",
            description="Course code used to name and group uploaded modules",
        ),
    ],
    files: Annotated[
        list[UploadFile], File(description="TXT module reports")
    ],
) -> dict:
    request_id = _PIPELINE_STATUS.enqueue(
        course_code,
        [upload.filename or "(unnamed)" for upload in files],
    )
    indexed_errors: list[tuple[int, int, dict[str, str]]] = []
    error_order = 0

    def record_error(index: int, filename: str, error: object) -> None:
        nonlocal error_order
        indexed_errors.append(
            (index, error_order, {"file": filename, "error": str(error)})
        )
        error_order += 1
        indexes = range(len(files)) if index < 0 else (index,)
        for module_index in indexes:
            _PIPELINE_STATUS.update(
                request_id,
                module_index,
                state="failed",
                stage="failed",
                message=str(error),
                error=str(error),
            )

    outputs: list[str] = []
    try:
        if not course_code.strip():
            record_error(-1, "(batch)", "Missing course code")
        else:
            try:
                safe_course_component(course_code)
                course_upload_directory(course_code)
            except ValueError as exc:
                record_error(-1, "(batch)", exc)
            else:
                await acquire_request_lock()
                try:
                    items: list[BatchItem] = []
                    item_indexes: dict[str, int] = {}
                    seen_module_filenames: set[str] = set()
                    seen_workspaces: set[str] = set()
                    seen_outputs: set[str] = set()
                    output_root = OUTPUT_ROOT
                    for index, upload in enumerate(files):
                        filename = upload.filename or "(unnamed)"
                        source: Path | None = None
                        _PIPELINE_STATUS.update(
                            request_id,
                            index,
                            state="processing",
                            stage="validating",
                            progress_percent=0,
                            message="Saving and validating module input...",
                        )
                        try:
                            safe_filename(upload.filename or "")
                            source = await save_upload(upload, course_code)
                            module_number = structured_module_number(source, course_code)
                            canonical_filename = canonical_module_filename(
                                course_code,
                                module_number,
                            )
                            canonical_key = canonical_filename.casefold()
                            if canonical_key in seen_module_filenames:
                                raise ValueError(
                                    "module_number collides with an earlier upload: "
                                    f"{module_number}"
                                )
                            seen_module_filenames.add(canonical_key)
                            source = publish_module_upload(
                                source,
                                course_code,
                                module_number,
                            )
                            args = pipeline_args(source, course_code, module_number)
                            paths = pipeline_paths(
                                source,
                                output_root,
                                course_code,
                                args.module_number,
                            )
                            workspace_key = paths.workspace.name.casefold()
                            if workspace_key in seen_workspaces:
                                raise ValueError(
                                    "workspace collides with an earlier upload: "
                                    f"{paths.workspace.name}"
                                )
                            seen_workspaces.add(workspace_key)
                            output_key = str(paths.flashcards.resolve()).casefold()
                            if output_key in seen_outputs:
                                raise ValueError(
                                    "module_number collides with an earlier upload: "
                                    f"{module_number}"
                                )
                            seen_outputs.add(output_key)
                            item = BatchItem(
                                filename,
                                args,
                                paths,
                            )
                            items.append(item)
                            item_indexes[filename] = index
                            _PIPELINE_STATUS.update(
                                request_id,
                                index,
                                filename=canonical_filename,
                                module_number=module_number,
                                state="queued",
                                stage="ingest",
                                progress_percent=0,
                                message="Waiting for module input staging.",
                            )
                        except (OSError, ValueError, RuntimeError) as exc:
                            if source is not None and source.suffix == ".tmp":
                                try:
                                    source.unlink()
                                except OSError:
                                    pass
                            record_error(index, filename, exc)

                    try:
                        def report_progress(event: BatchProgressEvent) -> None:
                            index = item_indexes.get(event.filename)
                            if index is not None:
                                _PIPELINE_STATUS.apply_event(
                                    request_id,
                                    index,
                                    event,
                                )

                        batch_result = await asyncio.to_thread(
                            run_batch,
                            tuple(items),
                            progress=report_progress,
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        for item in items:
                            record_error(item_indexes[item.filename], item.filename, exc)
                    else:
                        for error in batch_result.errors:
                            filename = error["file"]
                            record_error(
                                item_indexes.get(filename, len(files)),
                                filename,
                                error["error"],
                            )
                        outputs = [str(output.resolve()) for output in batch_result.outputs]
                        for item in items:
                            output = str(item.paths.flashcards.resolve())
                            if output in outputs:
                                _PIPELINE_STATUS.update(
                                    request_id,
                                    item_indexes[item.filename],
                                    state="completed",
                                    stage="complete",
                                    progress_percent=100,
                                    message="Module processing complete.",
                                    output=output,
                                )
                finally:
                    _REQUEST_LOCK.release()
    finally:
        for index, upload in enumerate(files):
            filename = upload.filename or "(unnamed)"
            try:
                await upload.close()
            except Exception as exc:
                record_error(index, filename, exc)
        _PIPELINE_STATUS.finish(request_id)

    errors = [
        error
        for _index, _order, error in sorted(indexed_errors, key=lambda item: item[:2])
    ]
    return {
        "outputs": outputs,
        "errors": errors,
        "courseCode": course_code,
        "processUrl": "/process",
    }


@app.post("/process/{course_code}", include_in_schema=False)
async def process_files_legacy(
    course_code: str,
    files: Annotated[
        list[UploadFile], File(description="Structured TXT module files")
    ],
) -> dict:
    """Temporary compatibility route for older extension builds."""
    return await process_files(course_code, files)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
