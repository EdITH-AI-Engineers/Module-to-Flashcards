from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import json
import os
import re
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
from typing import Annotated

from fastapi import FastAPI, File, Form, Response, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from uuid import uuid4
import time
from fastapi.middleware.cors import CORSMiddleware

from batch_pipeline import BatchItem, BatchProgressEvent, run_batch
from artifact_paths import module_file_label
from flashcard_types import CompletionTruncatedError, IncompleteCompletionError
from mistral_backend import DEFAULT_N_CTX, DEFAULT_MODEL, ApiKeyPool, MistralBackend, MistralRequestError
from module_sources import pdf_identity
from pipeline import pipeline_paths
from portable_paths import PortablePaths, build_paths
from saved_graphs import (
    SavedGraphModule, copy_checkpoint, discover_checkpoints, saved_run_root, saved_run_summary,
)
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
MODEL_GPU_LAYERS = -1
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
    outputs: list[str] = field(default_factory=list)


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
        course_code: str | None = None,
        module_number: str | None = None,
        state: str | None = None,
        stage: str | None = None,
        progress_percent: int | None = None,
        message: str | None = None,
        error: str | None = None,
        output: str | None = None,
        outputs: list[str] | None = None,
    ) -> None:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or not 0 <= index < len(request.modules):
                return
            module = request.modules[index]
            if filename is not None:
                module.filename = str(filename)
            if course_code is not None:
                module.course_code = str(course_code)
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
            if outputs is not None:
                module.outputs = list(outputs)

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

    def snapshot(self, request_id: int | None = None) -> dict[str, object]:
        with self._lock:
            modules = [
                self._module_snapshot(request.request_id, index, module)
                for request in self._requests.values()
                if request_id is None or request.request_id == request_id
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
            "outputs": list(module.outputs),
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


@dataclass
class _MistralSession:
    pool: ApiKeyPool
    model: str
    workers: int
    ocr_language: str = "eng"
    ocr_mode: str = "auto"
    lock: threading.Lock = field(default_factory=threading.Lock)
    request_id: int = 0
    last_used: float = field(default_factory=time.monotonic)
    final_status: dict | None = None
    run_number: int = 0


_SESSIONS: dict[str, _MistralSession] = {}
_SESSIONS_LOCK = threading.Lock()


def _get_session(session_id: str) -> _MistralSession:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(404, "Run not found. Create a new run after restarting the server.")
        session.last_used = time.monotonic()
        return session


@app.middleware("http")
async def protect_run_configuration(request: Request, call_next):
    # Credential configuration is available only to pages on this server.
    # Existing extension clients retain the original /process interface.
    if request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Open the app on this server to configure runs."}, status_code=403)
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, exc: RequestValidationError):
    from fastapi.exception_handlers import request_validation_exception_handler
    from fastapi.responses import JSONResponse
    if request.url.path == "/api/sessions":
        return JSONResponse({"detail": "Check API keys, model ID, and worker count (1–20)."}, status_code=422)
    return await request_validation_exception_handler(request, exc)


class SessionSettings(BaseModel):
    apiKeys: list[str] = Field(min_length=1, max_length=100, repr=False)
    model: str = Field(default=DEFAULT_MODEL, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    workers: int = Field(default=1, ge=1, le=20)
    ocrLanguage: str = Field(default="eng", pattern=r"^[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*$", max_length=80)
    ocrMode: str = Field(default="auto", pattern=r"^(auto|always)$")


class ResumeSettings(BaseModel):
    sourceRunId: str = Field(pattern=r"^[a-f0-9]{32}$")


@app.get("/", response_class=HTMLResponse)
def home():
    from web_ui import HTML
    return HTMLResponse(HTML, headers={"Cache-Control": "no-store"})


@app.post("/api/sessions")
def create_session(settings: SessionSettings, response: Response):
    try:
        pool = ApiKeyPool(settings.apiKeys)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    with _SESSIONS_LOCK:
        now = time.monotonic()
        for key, old in list(_SESSIONS.items()):
            if not old.lock.locked() and now - old.last_used > 12 * 3600:
                del _SESSIONS[key]
        if len(_SESSIONS) >= 64:
            raise HTTPException(409, "Close a completed run before creating another (64 run limit).")
        session_id = uuid4().hex
        run_number = max((old.run_number for old in _SESSIONS.values()), default=0) + 1
        _SESSIONS[session_id] = _MistralSession(pool, settings.model, settings.workers,
            ocr_language=settings.ocrLanguage, ocr_mode=settings.ocrMode, run_number=run_number)
    response.headers["Cache-Control"] = "no-store"
    return {"sessionId": session_id, "runNumber": run_number, "keys": pool.snapshot()}


@app.get("/api/sessions")
def list_sessions(response: Response):
    """Rediscover submitted runs after refresh without storing API keys in the browser."""
    response.headers["Cache-Control"] = "no-store"
    with _SESSIONS_LOCK:
        return {"sessions": [
            {"sessionId": session_id, "runNumber": session.run_number,
             "model": session.model, "running": session.lock.locked()}
            for session_id, session in _SESSIONS.items() if session.request_id
        ]}


@app.get("/api/ocr")
def ocr_status():
    from pdf_ingestion import find_tesseract
    try:
        find_tesseract()
        return {"available": True, "message": "Tesseract OCR is ready."}
    except RuntimeError as exc:
        return {"available": False, "message": str(exc)}


@app.post("/api/sessions/{session_id}/check")
async def check_session_model(session_id: str):
    """Check actual model access before the browser uploads any module PDFs."""
    session = _get_session(session_id)
    if not session.lock.acquire(blocking=False):
        raise HTTPException(409, "Wait for this run to finish before checking its model.")
    try:
        backend = MistralBackend(session.pool, model=session.model, timeout=30,
                                 retry_wait_timeout=45, max_retries=1)
        try:
            result = await asyncio.to_thread(backend.complete,
                "Return JSON only.", 'Reply with {"ok":true}.', max_tokens=32,
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"], "additionalProperties": False})
        except MistralRequestError as exc:
            status = exc.status if exc.status in (400, 401, 402, 403, 404, 429) else (
                429 if exc.category == "rate_limited" else 502)
            raise HTTPException(status, session.pool.redact(exc, 1000)) from None
        except (IncompleteCompletionError, CompletionTruncatedError) as exc:
            raise HTTPException(502, session.pool.redact(exc, 1000)) from None
        try:
            if json.loads(result) != {"ok": True}:
                raise ValueError("Unexpected connection check response")
        except (ValueError, TypeError):
            raise HTTPException(502, "Mistral did not return a valid model connection check.") from None
        return {"model": session.model, "message": "Mistral model connection verified."}
    finally:
        session.lock.release()


@app.get("/api/sessions/{session_id}")
def session_status(session_id: str, response: Response):
    session = _get_session(session_id)
    response.headers["Cache-Control"] = "no-store"
    status = session.final_status or _PIPELINE_STATUS.snapshot(session.request_id)
    return {**status, "runNumber": session.run_number, "model": session.model, "keys": session.pool.snapshot(), "apiDiagnostics": session.pool.diagnostics(),
            "running": session.lock.locked()}


@app.delete("/api/sessions/{session_id}")
def close_session(session_id: str):
    with _SESSIONS_LOCK:
        session = _get_session_unlocked(session_id)
        if not session.lock.acquire(blocking=False):
            raise HTTPException(409, "Wait for this run to finish before closing it.")
        del _SESSIONS[session_id]
    return {"closed": True}


def _get_session_unlocked(session_id: str) -> _MistralSession:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Run not found.")
    return session


@app.get("/api/sessions/{session_id}/files/{request_index}/{part}")
def download_csv(session_id: str, request_index: int, part: int):
    session = _get_session(session_id)
    status = session.final_status or _PIPELINE_STATUS.snapshot(session.request_id)
    module = next((item for item in status["modules"] if item["index"] == request_index), None)
    if module is None or module["state"] != "completed" or part not in (1, 2) or len(module["outputs"]) != 2:
        raise HTTPException(404, "Completed CSV not found.")
    path = Path(module["outputs"][part - 1]).resolve()
    root = (OUTPUT_ROOT.parent / "runs" / session_id).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Completed CSV not found.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


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
    global MODEL_GPU_LAYERS, KG_DEVICE
    if kg_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported knowledge-graph device: {kg_device}")
    MODEL_GPU_LAYERS = int(n_gpu_layers)
    KG_DEVICE = kg_device


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("uploaded file must have a filename")
    if Path(name).suffix.casefold() not in {".txt", ".pdf"}:
        raise ValueError("only PDF modules or TXT module reports are accepted")
    return name


def filename_module_identity(filename: str) -> tuple[str, str] | None:
    """Return identity from an exact ``<course>-M<number>.txt`` filename."""
    name = Path(filename).name
    match = re.fullmatch(
        r"(?P<course>[A-Za-z0-9][A-Za-z0-9._-]*)-M(?P<module>[0-9]+)\.(?:txt|pdf)",
        name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group("course"), match.group("module")


def safe_course_component(course_code: str) -> str:
    """Return one safe directory component without changing the public code."""
    value = str(course_code).strip()
    if not value or value in {".", ".."}:
        raise ValueError("course code must not be blank, '.' or '..'")
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not component or component in {".", ".."}:
        raise ValueError("course code does not contain a safe path component")
    return component


def course_upload_directory(course_code: str, upload_root: Path | None = None) -> Path:
    component = safe_course_component(course_code)
    root = (upload_root or UPLOAD_DIR).resolve()
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


async def _run_owned_thread(function, *args, **kwargs):
    """Keep run ownership until an off-thread file operation has finished."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        return await task


async def save_upload(upload: UploadFile, course_code: str, upload_root: Path | None = None) -> Path:
    filename = safe_filename(upload.filename or "")
    destination_dir = course_upload_directory(course_code, upload_root)
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


def canonical_module_filename(course_code: str, module_number: str, extension: str = ".txt") -> str:
    """Build the canonical upload name from payload identity."""
    course = safe_course_component(course_code)
    module = module_file_label(module_number)
    return f"{course}_M{module}{extension}"


def publish_module_upload(
    staged_source: Path,
    course_code: str,
    module_number: str,
    extension: str = ".txt",
) -> Path:
    """Atomically rename a validated staged upload to its canonical name."""
    source = Path(staged_source)
    destination = source.with_name(
        canonical_module_filename(course_code, module_number, extension)
    )
    if source != destination:
        source.replace(destination)
    return destination


def pipeline_args(
    source: Path,
    course_code: str,
    module_number: str,
    *,
    identity_from_filename: bool = False,
) -> Namespace:
    from worker_budget import parse_cluster_workers

    try:
        cluster_workers = parse_cluster_workers(
            os.environ.get("MODULE_FLASHCARDS_CLUSTER_WORKERS", "auto")
        )
    except ValueError as exc:
        raise ValueError(
            "MODULE_FLASHCARDS_CLUSTER_WORKERS must be 'auto' or 1 to 20"
        ) from exc
    try:
        clusters_per_call = int(os.environ.get("MODULE_FLASHCARDS_CLUSTERS_PER_CALL", "1"))
        if clusters_per_call not in (1, 2, 5):
            raise ValueError("unsupported batch size")
    except ValueError as exc:
        raise ValueError(
            "MODULE_FLASHCARDS_CLUSTERS_PER_CALL must be 1, 2, or 5"
        ) from exc
    return Namespace(
        input=source,
        course_code=course_code,
        module_number=module_number,
        identity_from_filename=identity_from_filename,
        output_root=OUTPUT_ROOT,
        model_dir=MODEL_DIR,
        mistral_model=os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL),
        rebel_model=REBEL_MODEL,
        portable=PORTABLE_MODE,
        attempts=3,
        seed=42,
        n_gpu_layers=MODEL_GPU_LAYERS,
        n_ctx=DEFAULT_N_CTX,
        ocr_language="eng",
        ocr_dpi=200,
        ocr_mode="auto",
        cluster_workers=cluster_workers,
        clusters_per_call=clusters_per_call,
        timeout=0,
        kg_device=KG_DEVICE,
        kg_batch_size=1,
        kg_num_beams=1,
        # Batched prompts expose several concepts' facts together, so keep
        # concept-scoped grounding review enabled when batching is requested.
        skip_final_review=clusters_per_call == 1,
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


@app.get("/api/saved-runs")
def list_saved_runs(response: Response):
    """Find durable graph and structured-text checkpoints after a restart."""
    response.headers["Cache-Control"] = "no-store"
    root = OUTPUT_ROOT.parent / "runs"
    runs = []
    if root.is_dir():
        for candidate in root.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", candidate.name):
                continue
            try:
                run_root = saved_run_root(root, candidate.name)
                summary = saved_run_summary(run_root)
            except (OSError, ValueError):
                continue
            if summary is not None:
                with _SESSIONS_LOCK:
                    session = _SESSIONS.get(candidate.name)
                    summary["running"] = bool(session and session.lock.locked())
                runs.append(summary)
    return {"runs": sorted(runs, key=lambda run: run["updatedAt"], reverse=True)}


@app.post("/api/sessions/{session_id}/resume")
async def resume_saved_modules(session_id: str, settings: ResumeSettings):
    """Continue from saved graphs or structured text in an isolated new run."""
    session = _get_session(session_id)
    if settings.sourceRunId == session_id:
        raise HTTPException(409, "Choose a previous run to continue in this new run.")
    if not session.lock.acquire(blocking=False):
        raise HTTPException(409, "This run is already processing.")
    source_session = None
    source_locked = False
    request_id = None
    outputs: list[str] = []
    errors: list[dict[str, object]] = []
    try:
        if session.request_id:
            raise HTTPException(409, "Create a new run to continue saved modules.")
        try:
            source_root = saved_run_root(OUTPUT_ROOT.parent / "runs", settings.sourceRunId)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from None
        with _SESSIONS_LOCK:
            source_session = _SESSIONS.get(settings.sourceRunId)
            if source_session is not None:
                source_locked = source_session.lock.acquire(blocking=False)
                if not source_locked:
                    raise HTTPException(409, "The saved run is still processing. Wait for it to finish.")
        modules, discovery_errors = await _run_owned_thread(discover_checkpoints, source_root)
        if not modules:
            raise HTTPException(422, "This run has no usable saved knowledge graphs or structured modules.")
        destination_root = OUTPUT_ROOT.parent / "runs" / session_id
        request_id = _PIPELINE_STATUS.enqueue("", [
            canonical_module_filename(module.course_code, module.module_number, module.source.suffix)
            for module in modules
        ] + [error["file"] for error in discovery_errors])
        session.request_id = request_id
        session.final_status = None

        def fail(index: int, filename: str, error: object):
            message = session.pool.redact(error, 2000)
            errors.append({"file": filename, "error": message})
            _PIPELINE_STATUS.update(request_id, index, state="failed", stage="failed",
                                    message=message, error=message)

        for index, error in enumerate(discovery_errors, start=len(modules)):
            fail(index, error["file"], error["error"])
        items = []
        for index, module in enumerate(modules):
            from_graph = isinstance(module, SavedGraphModule)
            filename = canonical_module_filename(module.course_code, module.module_number, module.source.suffix)
            try:
                paths = await _run_owned_thread(copy_checkpoint, module, destination_root)
                if from_graph:
                    source = paths.graph_json if module.checked else paths.unchecked_graph_json
                else:
                    source = paths.structured_text
                args = pipeline_args(source, module.course_code, module.module_number,
                    identity_from_filename=not from_graph and module.identity_from_filename)
                args.start_from_graph = from_graph
                args.start_from_structured = not from_graph
                args.output_root = destination_root / "pipeline_output"
                args._mistral_pool = session.pool
                args.mistral_model = session.model
                args.cluster_workers = session.workers
                args.skip_final_review = False
                items.append(BatchItem(filename, args, paths, request_index=index))
                _PIPELINE_STATUS.update(request_id, index, course_code=module.course_code,
                    module_number=module.module_number, stage="flashcards" if from_graph else "graph",
                    progress_percent=45 if from_graph else 10,
                    message="Continuing from saved knowledge graph." if from_graph else
                            "Continuing from saved structured module; skipping ingestion.")
            except (OSError, ValueError, RuntimeError) as exc:
                fail(index, filename, exc)
        if source_locked:
            source_session.lock.release()
            source_locked = False

        def report_progress(event: BatchProgressEvent):
            if event.request_index is not None:
                _PIPELINE_STATUS.apply_event(request_id, event.request_index, event)

        try:
            result = await _run_owned_thread(run_batch, tuple(items), progress=report_progress)
        except (OSError, ValueError, RuntimeError) as exc:
            for item in items:
                fail(item.request_index, item.filename, exc)
        else:
            for error in result.errors:
                index = error.get("uploadIndex")
                if not isinstance(index, int):
                    index = next(item.request_index for item in items if item.filename == error["file"])
                fail(index, str(error["file"]), error["error"])
            outputs = [str(path.resolve()) for path in result.outputs]
            for item in items:
                item_outputs = [str(path.resolve()) for path in item.paths.flashcard_parts]
                if all(path in outputs for path in item_outputs):
                    _PIPELINE_STATUS.update(request_id, item.request_index,
                        state="completed", stage="complete", progress_percent=100,
                        message="Module processing complete.", output=item_outputs[0], outputs=item_outputs)
        return {"outputs": outputs, "errors": errors, "sourceRunId": settings.sourceRunId}
    finally:
        if source_locked:
            source_session.lock.release()
        if request_id is not None:
            _PIPELINE_STATUS.finish(request_id)
            session.final_status = _PIPELINE_STATUS.snapshot(request_id)
        session.last_used = time.monotonic()
        session.lock.release()


async def process_files(
    course_code: Annotated[
        str,
        Form(
            alias="courseCode",
            description="Course code used to name and group uploaded modules",
        ),
    ],
    files: Annotated[
        list[UploadFile], File(description="Module PDFs from course folders, or TXT reports")
    ],
    session_id: str | None = None,
) -> dict:
    session = _get_session(session_id) if session_id is not None else None
    if session is not None and not session.lock.acquire(blocking=False):
        raise HTTPException(409, "This run is already processing. Create another run for separate data.")
    request_id = _PIPELINE_STATUS.enqueue(
        course_code,
        [upload.filename or "(unnamed)" for upload in files],
    )
    if session is not None:
        session.request_id = request_id
        session.final_status = None
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
        has_filename_identity = any(
            filename_module_identity(upload.filename or "") is not None
            or (str(upload.filename).lower().endswith(".pdf")
                and len(str(upload.filename).replace("\\", "/").split("/")) > 1)
            for upload in files
        )
        if not has_filename_identity and not course_code.strip():
            record_error(-1, "(batch)", "Missing course code")
        else:
            try:
                if not has_filename_identity:
                    safe_course_component(course_code)
                    course_upload_directory(course_code)
            except ValueError as exc:
                record_error(-1, "(batch)", exc)
            else:
                if session is None:
                    await acquire_request_lock()
                try:
                    items: list[BatchItem] = []
                    item_indexes: dict[str, int] = {}
                    seen_module_filenames: set[str] = set()
                    seen_workspaces: set[str] = set()
                    seen_outputs: set[str] = set()
                    run_root = OUTPUT_ROOT.parent / "runs" / session_id if session else None
                    output_root = run_root / "pipeline_output" if run_root else OUTPUT_ROOT
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
                            filename_identity = filename_module_identity(
                                upload.filename or ""
                            )
                            is_pdf = str(upload.filename).lower().endswith(".pdf")
                            if is_pdf:
                                filename_identity = pdf_identity(upload.filename or "", course_code)
                            effective_course = (
                                filename_identity[0]
                                if filename_identity is not None
                                else course_code
                            )
                            if not effective_course.strip():
                                raise ValueError("Missing course code")
                            safe_course_component(effective_course)
                            course_upload_directory(effective_course)
                            source = (
                                await save_upload(upload, effective_course, run_root / "uploads")
                                if run_root else await save_upload(upload, effective_course)
                            )
                            module_number = (
                                filename_identity[1]
                                if filename_identity is not None
                                else structured_module_number(source, effective_course)
                            )
                            if is_pdf:
                                from pdf_ingestion import validate_pdf
                                await asyncio.to_thread(validate_pdf, source)
                            canonical_filename = canonical_module_filename(
                                effective_course,
                                module_number,
                                ".pdf" if is_pdf else ".txt",
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
                                effective_course,
                                module_number,
                                ".pdf" if is_pdf else ".txt",
                            )
                            args = pipeline_args(
                                source,
                                effective_course,
                                module_number,
                                identity_from_filename=filename_identity is not None,
                            )
                            if session is not None:
                                args._mistral_pool = session.pool
                                args.mistral_model = session.model
                                args.cluster_workers = session.workers
                                args.ocr_language = session.ocr_language
                                args.ocr_mode = session.ocr_mode
                                args.skip_final_review = False
                            paths = pipeline_paths(
                                source,
                                output_root,
                                effective_course,
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
                                request_index=index,
                            )
                            items.append(item)
                            item_indexes[filename] = index
                            _PIPELINE_STATUS.update(
                                request_id,
                                index,
                                filename=canonical_filename,
                                course_code=effective_course,
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
                            index = (
                                event.request_index
                                if event.request_index is not None
                                else item_indexes.get(event.filename)
                            )
                            if index is not None:
                                _PIPELINE_STATUS.apply_event(
                                    request_id,
                                    index,
                                    event,
                                )

                        batch_task = asyncio.create_task(asyncio.to_thread(
                            run_batch, tuple(items), progress=report_progress))
                        try:
                            batch_result = await asyncio.shield(batch_task)
                        except asyncio.CancelledError:
                            # A disconnected browser must not release ownership
                            # while its worker still writes this run's files.
                            batch_result = await batch_task
                    except (OSError, ValueError, RuntimeError) as exc:
                        for item in items:
                            record_error(item.request_index, item.filename, exc)
                    else:
                        for error in batch_result.errors:
                            filename = str(error["file"])
                            error_index = error.get("uploadIndex")
                            record_error(
                                (
                                    error_index
                                    if isinstance(error_index, int)
                                    else item_indexes.get(filename, len(files))
                                ),
                                filename,
                                error["error"],
                            )
                        outputs = [str(output.resolve()) for output in batch_result.outputs]
                        for item in items:
                            item_outputs = [
                                str(path.resolve()) for path in item.paths.flashcard_parts
                            ]
                            if all(path in outputs for path in item_outputs):
                                _PIPELINE_STATUS.update(
                                    request_id,
                                    item.request_index,
                                    state="completed",
                                    stage="complete",
                                    progress_percent=100,
                                    message="Module processing complete.",
                                    output=item_outputs[0],
                                    outputs=item_outputs,
                                )
                finally:
                    if session is None:
                        _REQUEST_LOCK.release()
    finally:
        for index, upload in enumerate(files):
            filename = upload.filename or "(unnamed)"
            try:
                await upload.close()
            except Exception as exc:
                record_error(index, filename, exc)
        _PIPELINE_STATUS.finish(request_id)
        if session is not None:
            session.final_status = _PIPELINE_STATUS.snapshot(request_id)
            session.last_used = time.monotonic()
            session.lock.release()

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


@app.post("/process")
async def upload_modules(
    files: Annotated[list[UploadFile], File(description="Module PDFs from course folders, or TXT reports")],
    course_code: Annotated[str, Form(alias="courseCode")] = "",
    session_id: str | None = None,
) -> dict:
    return await process_files(course_code, files, session_id)


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
