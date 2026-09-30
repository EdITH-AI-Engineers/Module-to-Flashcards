# Module to Flashcards

This project accepts UTF-8 text module reports, extracts a relationship
graph with REBEL, and uses a local Qwen model to generate validated,
copy-paste-ready assessment CSV files.

Current release: 1.1.0

## Input format

The local server accepts `.txt` slide reports with `Module #:` and `Module Title:`
headers followed by `Slide N:` sections. Each slide may contain `Title:`,
`Content:`, `Image/Diagram Description:`, and `Brief Explanation:`. The report
is staged in its original text form. The earlier `[MODULE]` / `[SLIDE n]`
format is also accepted for existing inputs. PDF files are not accepted, and
the application does not perform image recognition or document conversion.

A minimal example is:

```text
Module #: 9
Module Title: BASIC ELECTRICAL ENGINEERING

---

Slide 1:
{
Title:
Effective value of AC

Content:
Course: BASICEE
The effective value of AC is its root mean square (RMS) value.

Image/Diagram Description:
Not Specified
}

Brief Explanation:
RMS compares the heating effect of AC with DC in the same resistor.

---
```

If `Module #:` is `Not Specified`, a slide titled `MODULE 9` supplies module
number `9`. The course code is provided separately as `courseCode` in the API
or `--course-code` in the command line.

## Quick start with the local server

From the project directory in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python api_server.py
```

The server listens at `http://localhost:8000`. Check it with:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

Send one or more text module reports as multipart form data to:

```text
POST http://localhost:8000/process
```

The multipart payload must contain `courseCode` once and one or more `files`
fields. The server reads the module number from each report, checks any declared
course code against `courseCode`, and stores the upload as
`<courseCode>_M<moduleNumber>.txt` regardless of its original filename.

While processing is running, poll the status endpoint:

```text
GET http://localhost:8000/status
```

It returns the currently active module, the ordered module queue, recent module
states, and summary counts. Each module includes `stage`, `progressPercent`,
`message`, `error`, and its output path when complete. The top-level `status` is
`processing` while any module is active or queued, otherwise it is `idle`.

Uploaded modules are saved under `pipeline_uploads/<course>/`. Intermediate artifacts are stored under
`pipeline_output/<course>/<module-name>/`, and final files are stored under
`flashcards/<course>/<course>_M<module>.csv`.

For each batch, the server stages and validates all pending text files, builds
their knowledge graphs with one REBEL model load, and generates flashcards with
one Qwen model load. Heavy inference is sequential. A failure in one module is
reported without preventing the other modules from completing. Re-sending an
unchanged module reuses valid completed artifacts.

Qwen uses a 12,288-token context window by default. Concept planning no longer
uses a fixed fact-count ceiling: it balances eligible facts across slides and
topics, then keeps the largest set that fits while reserving the complete
3,072-token planning response budget.

## Command-line pipeline

Run the complete sequence with a text module report:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\EDITH-2503-0522-5807-75B3.txt" `
  --course-code BASICEE --module-number 9
```

Useful options include `--output-root`, `--kg-device`, `--kg-batch-size`,
`--kg-num-beams`, `--n-gpu-layers`, `--n-ctx`, `--timeout`, and `--force`.
The timeout is disabled by default. `--force` recomputes the graph and
flashcards instead of resuming from valid artifacts.

Run only the knowledge-graph stage:

```powershell
.\.venv\Scripts\python.exe text-extractor.py "C:\path\to\EDITH-2503-0522-5807-75B3.txt" `
  --course-code BASICEE --module-number 9 --output-dir output
```

Run only flashcard generation:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json `
  --course-code BASICEE --module-number 9
```

Every successful module produces two labeled CSV blocks with 50 rows each:
20 graph-supported concept clusters, five questions per cluster, and exactly
100 questions in total. Partial output is not written when validation fails.

## Portable Windows release

The Windows release is a fully offline, one-directory application for Windows
11 x64. It bundles the Python runtime, Qwen3 8B Q5_K_M model, REBEL model, and
the required CUDA libraries. Keep the complete extracted directory together:

```text
ModuleToFlashcards/
  ModuleToFlashcards.exe
  runtime/
  models/
    Qwen3-8B-Q5_K_M.gguf
    rebel-large/
    manifest.json
  licenses/
  data/
    uploads/
    pipeline_output/
    flashcards/
    temporary/
    logs/
```

Double-click `ModuleToFlashcards.exe`, wait for the localhost readiness
message, and leave the console open while processing. Run a full integrity and
device check without starting the server with:

```powershell
.\ModuleToFlashcards.exe --verify
```

Use another port with `--port 8010`. The browser extension must use the same
port and include `http://localhost:8000/*` (or the chosen port) in its host
permissions.

The launcher reports `GPU acceleration ready` only when PyTorch CUDA and
llama.cpp GPU offload both work. Otherwise, it reports CPU fallback. Startup
failures are recorded under `data/logs/` when that directory is writable.

## Build the portable release

Maintainer builds require Windows x64, Python 3.11 or 3.12, the project and
PyInstaller dependencies, CUDA-enabled PyTorch and `llama-cpp-python` builds,
and internet access while staging the locked Qwen and REBEL models.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-build.txt
.\.venv\Scripts\python.exe -c "import torch, llama_cpp; print(torch.cuda.is_available()); print(llama_cpp.llama_supports_gpu_offload())"
powershell -ExecutionPolicy Bypass -File packaging/build_portable.ps1
```

The build validates `packaging/model-lock.json`, stages the exact locked model
assets, freezes the server, assembles and verifies `dist/ModuleToFlashcards`,
and creates the versioned Windows archive. Use `-SkipAssetPreparation` only
when locked assets are already staged. `-SkipGpuPreflight` is intended only for
configuration checks.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Generated models, portable assets, build output, uploads, pipeline output,
flashcards, temporary data, and logs must remain outside version control.
