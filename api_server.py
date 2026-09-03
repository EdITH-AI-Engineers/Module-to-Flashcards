from __future__ import annotations

import asyncio
import re
import time
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import PipelineRunError, pipeline_paths, run


PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_DIR / "pipeline_uploads"
OUTPUT_ROOT = PROJECT_DIR / "pipeline_output"
MODULE_TIMEOUT_SECONDS = 5 * 60

app = FastAPI(title="Module to Flashcards local processor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


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


async def save_upload(upload: UploadFile, course_code: str) -> Path:
    filename = safe_filename(upload.filename or "")
    destination_dir = UPLOAD_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", course_code)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename

    with NamedTemporaryFile(
        mode="wb", dir=destination_dir, prefix=f".{filename}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        while chunk := await upload.read(1024 * 1024):
            temporary.write(chunk)

    temporary_path.replace(destination)
    return destination


def pipeline_args(pdf: Path, course_code: str, module_number: str) -> Namespace:
    return Namespace(
        pdf=pdf,
        course_code=course_code,
        module_number=module_number,
        module_title=None,
        output_root=OUTPUT_ROOT,
        model_dir=PROJECT_DIR / "models",
        attempts=3,
        seed=42,
        n_gpu_layers=-1,
        n_ctx=32768,
        ocr_min_chars=40,
        ocr_dpi=200,
        timeout=300,
        kg_device="auto",
        skip_final_review=False,
        force=False,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process/{course_code}")
async def process_files(
    course_code: str,
    files: Annotated[list[UploadFile], File(description="PDF module files")],
) -> dict:
    if not course_code.strip():
        return {"outputs": [], "errors": [{"pdf": "(batch)", "error": "Missing course code"}]}

    outputs: list[str] = []
    errors: list[dict[str, str]] = []
    for upload in files:
        filename = upload.filename or "(unnamed)"
        deadline = time.monotonic() + MODULE_TIMEOUT_SECONDS
        try:
            pdf = await asyncio.wait_for(
                save_upload(upload, course_code),
                timeout=MODULE_TIMEOUT_SECONDS,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("module exceeded 300-second timeout")
            paths = run(
                pipeline_args(pdf, course_code, module_number_from_filename(filename)),
                timeout_seconds=remaining,
            )
            outputs.append(str(paths.flashcards.resolve()))
        except asyncio.TimeoutError as exc:
            errors.append({"pdf": filename, "error": "module exceeded 300-second timeout"})
        except (OSError, PipelineRunError, ValueError, RuntimeError) as exc:
            errors.append({"pdf": filename, "error": str(exc)})
        finally:
            await upload.close()

    return {
        "outputs": outputs,
        "errors": errors,
        "courseCode": course_code,
        "processUrl": f"/process/{quote(course_code, safe='')}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)