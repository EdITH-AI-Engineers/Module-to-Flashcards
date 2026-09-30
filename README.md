# Module to Flashcards

This project accepts structured UTF-8 text modules, extracts a relationship
graph with REBEL, and uses a local Qwen model to generate validated,
copy-paste-ready assessment CSV files.

Current release: 1.1.0

## Input format

The local server accepts `.txt` files that already follow the structured module
contract. Each file must contain a `[MODULE]` block with `format_version: 1`
and at least one readable `[SLIDE n]` block. PDF files are not accepted, and
the application does not perform image recognition or document conversion.

A minimal example is:

```text
[MODULE]
format_version: 1
course_code: CPE0021
module_number: 1
module_title: Processor Architecture
source_file: CPE0021-M1.txt
[/MODULE]

[SLIDE 1]
extraction_method: text
[TITLE]
Processor
[/TITLE]
[CONTENT]
- A processor executes instructions.
[/CONTENT]
[VISUAL_TEXT]
- Not Specified
[/VISUAL_TEXT]
[DEFINITIONS]
Not Specified
[/DEFINITIONS]
[KNOWLEDGE_STATEMENTS]
- A processor contains an arithmetic logic unit.
[/KNOWLEDGE_STATEMENTS]
[BRIEF_EXPLANATION]
The slide describes a processor.
[/BRIEF_EXPLANATION]
[/SLIDE]
```

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

Send one or more structured text modules as multipart form data to:

```text
POST http://localhost:8000/process
```

The multipart payload must contain `courseCode` once and one or more `files`
fields. The server reads `module_number` from each structured module, verifies
the module's embedded course code against `courseCode`, and stores the upload as
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

Run the complete sequence with a structured text module:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\CPE0021-M1.txt" `
  --course-code CPE0021 --module-number 1
```

Useful options include `--output-root`, `--kg-device`, `--kg-batch-size`,
`--kg-num-beams`, `--n-gpu-layers`, `--n-ctx`, `--timeout`, and `--force`.
The timeout is disabled by default. `--force` recomputes the graph and
flashcards instead of resuming from valid artifacts.

Run only the knowledge-graph stage:

```powershell
.\.venv\Scripts\python.exe text-extractor.py structured-module.txt --output-dir output
```

Run only flashcard generation:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json `
  --course-code CPE0021 --module-number 1
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
