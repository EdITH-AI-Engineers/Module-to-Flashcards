from __future__ import annotations

import asyncio
import re
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from batch_pipeline import BatchItem, run_batch
from local_qwen import DEFAULT_N_CTX
from pipeline import pipeline_paths
from portable_paths import PortablePaths, build_paths
from version import __version__


PROJECT_DIR = Path(__file__).resolve().parent
_RUNTIME_PATHS = build_paths(PROJECT_DIR, portable=False)
UPLOAD_DIR = _RUNTIME_PATHS.uploads
OUTPUT_ROOT = _RUNTIME_PATHS.outputs
MODEL_DIR = _RUNTIME_PATHS.models
REBEL_MODEL = _RUNTIME_PATHS.rebel_model
PORTABLE_MODE = _RUNTIME_PATHS.portable
_REQUEST_LOCK = threading.Lock()
_REQUEST_LOCK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="batch-request-lock"
)
app = FastAPI(title="Module to Flashcards local processor", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
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


def module_number_from_filename(filename: str) -> str:
    match = re.search(r"(?:^|[-_ ])M(?:odule)?[-_ ]?(\d+)(?:\D|$)", filename, re.IGNORECASE)
    return match.group(1) if match else "1"


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValueError("uploaded file must have a filename")
    if not name.casefold().endswith(".pdf"):
        raise ValueError("only PDF files are accepted")
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
    destination = destination_dir / filename

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=destination_dir, prefix=f".{filename}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := await upload.read(1024 * 1024):
                temporary.write(chunk)
        temporary_path.replace(destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def pipeline_args(pdf: Path, course_code: str, module_number: str) -> Namespace:
    return Namespace(
        pdf=pdf,
        course_code=course_code,
        module_number=module_number,
        module_title=None,
        output_root=OUTPUT_ROOT,
        model_dir=MODEL_DIR,
        rebel_model=REBEL_MODEL,
        portable=PORTABLE_MODE,
        attempts=3,
        seed=42,
        n_gpu_layers=-1,
        n_ctx=DEFAULT_N_CTX,
        ocr_min_chars=40,
        ocr_dpi=200,
        timeout=0,
        kg_device="auto",
        kg_batch_size=1,
        kg_num_beams=1,
        skip_final_review=True,
        force=False,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/process/{course_code}")
async def process_files(
    course_code: str,
    files: Annotated[list[UploadFile], File(description="PDF module files")],
) -> dict:
    indexed_errors: list[tuple[int, int, dict[str, str]]] = []
    error_order = 0

    def record_error(index: int, filename: str, error: object) -> None:
        nonlocal error_order
        indexed_errors.append(
            (index, error_order, {"pdf": filename, "error": str(error)})
        )
        error_order += 1

    outputs: list[str] = []
    try:
        if not course_code.strip():
            record_error(-1, "(batch)", "Missing course code")
        else:
            try:
                course_component = safe_course_component(course_code)
                course_upload_directory(course_code)
            except ValueError as exc:
                record_error(-1, "(batch)", exc)
            else:
                await acquire_request_lock()
                try:
                    items: list[BatchItem] = []
                    item_indexes: dict[str, int] = {}
                    seen_filenames: set[str] = set()
                    seen_workspaces: set[str] = set()
                    output_root = OUTPUT_ROOT / course_component
                    for index, upload in enumerate(files):
                        filename = upload.filename or "(unnamed)"
                        try:
                            filename_component = safe_filename(upload.filename or "")
                            filename_key = filename_component.casefold()
                            workspace_key = pipeline_paths(
                                Path(filename_component), output_root
                            ).workspace.name.casefold()
                            if filename_key in seen_filenames:
                                raise ValueError(
                                    "sanitized filename collides with an earlier "
                                    f"upload: {filename_component}"
                                )
                            if workspace_key in seen_workspaces:
                                raise ValueError(
                                    "workspace collides with an earlier upload: "
                                    f"{pipeline_paths(Path(filename_component), output_root).workspace.name}"
                                )
                            seen_filenames.add(filename_key)
                            seen_workspaces.add(workspace_key)
                            pdf = await save_upload(upload, course_code)
                            args = pipeline_args(
                                pdf, course_code, module_number_from_filename(filename)
                            )
                            item = BatchItem(
                                filename, args, pipeline_paths(pdf, output_root)
                            )
                            items.append(item)
                            item_indexes[filename] = index
                        except (OSError, ValueError, RuntimeError) as exc:
                            record_error(index, filename, exc)

                    try:
                        batch_result = await asyncio.to_thread(run_batch, tuple(items))
                    except (OSError, ValueError, RuntimeError) as exc:
                        for item in items:
                            record_error(item_indexes[item.filename], item.filename, exc)
                    else:
                        for error in batch_result.errors:
                            filename = error["pdf"]
                            record_error(
                                item_indexes.get(filename, len(files)),
                                filename,
                                error["error"],
                            )
                        outputs = [str(output.resolve()) for output in batch_result.outputs]
                finally:
                    _REQUEST_LOCK.release()
    finally:
        for index, upload in enumerate(files):
            filename = upload.filename or "(unnamed)"
            try:
                await upload.close()
            except Exception as exc:
                record_error(index, filename, exc)

    errors = [
        error
        for _index, _order, error in sorted(indexed_errors, key=lambda item: item[:2])
    ]
    return {
        "outputs": outputs,
        "errors": errors,
        "courseCode": course_code,
        "processUrl": f"/process/{quote(course_code, safe='')}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
