# Portable Offline Windows Executable Design

**Status:** Approved in chat; awaiting written-spec review  
**Date:** 2026-09-08  
**Target release:** Windows 11 x64 portable directory for an NVIDIA GeForce RTX 5070 12 GB system

## Context

Module to Flashcards currently runs from Python entry points. The browser extension sends PDF modules to a local FastAPI server, and the server runs a staged pipeline that uses Qwen for slide normalization and flashcard generation, REBEL for relationship extraction, and Tesseract for OCR fallback. Model and package loading currently assumes a Python environment and can contact Hugging Face when files are absent.

The requested deliverable is a portable Windows application that starts by double-clicking an executable, includes every inference model and runtime component needed by the pipeline, and performs generation without network access or a separately installed Python environment. Generated data must be stored in a folder beside the executable.

## Goals

- Produce a portable one-directory Windows distribution whose primary entry point is `ModuleToFlashcards.exe`.
- Bundle the exact Qwen GGUF, a complete local REBEL snapshot, a CUDA-capable Python runtime, and a portable Tesseract installation.
- Start the existing local HTTP service in a visible console on `127.0.0.1:8000` by default.
- Use the RTX 5070 for both llama.cpp and PyTorch inference when CUDA is available.
- Run without network access after the portable directory is built.
- Keep uploads, resumable artifacts, generated flashcards, temporary files, and logs under a `data` folder beside the executable.
- Preserve the existing staged, sequential model-loading behavior so Qwen and REBEL do not need to occupy VRAM simultaneously.
- Retain CPU fallback with a prominent warning when CUDA initialization fails.
- Provide deterministic build and verification commands and automated tests for the packaging-specific code.

## Non-goals

- A single-file self-extracting executable is not part of this release.
- A graphical desktop interface or system-tray application is not part of this release.
- The executable will not download, update, or repair models at runtime.
- Direct `.pptx` ingestion is not added; inputs remain PDFs.
- The Python developer command-line tools are not being replaced. The portable executable exposes the local-server workflow used by the browser extension.
- NVIDIA display drivers are not bundled. A compatible installed NVIDIA driver remains a system prerequisite.
- The large model and runtime binaries will not be committed to Git.

## Target environment and constraints

The acceptance target is Windows 11 x64 with an AMD Ryzen 7 5800X, 16 GB system RAM, and an NVIDIA GeForce RTX 5070 with 12 GB VRAM. Sixteen gigabytes of RAM is treated as the supported minimum for this bundle. The application must continue loading the Qwen and REBEL runtimes in separate stages to remain within the available memory budget.

The current development workspace is CPU-only and is not the target computer. It can validate source behavior and create the packaging configuration, but the final CUDA build and GPU acceptance test must be executed on the RTX 5070 computer. Success may not be claimed solely from a CPU-hosted PyInstaller build.

## Distribution layout

The completed portable directory will have this logical shape:

```text
ModuleToFlashcards/
|-- ModuleToFlashcards.exe
|-- runtime/
|   `-- Python modules, PyTorch/llama.cpp libraries, and required DLLs
|-- models/
|   |-- qwen2.5-3b-instruct-q5_k_m.gguf
|   |-- rebel-large/
|   `-- manifest.json
|-- tesseract/
|   |-- tesseract.exe
|   `-- tessdata/
|       `-- eng.traineddata
|-- licenses/
`-- data/
    |-- uploads/
    |-- pipeline_output/
    |-- temporary/
    `-- logs/
```

PyInstaller will create a console-enabled one-directory build and place collected Python dependencies under `runtime`. Models and Tesseract assets will be copied into the completed directory after the executable build so changing large assets does not require freezing the Python application again.

The repository will contain a small, text-only `packaging/model-lock.json`. It pins the Qwen file digest, the immutable Hugging Face revision for the REBEL snapshot, the expected REBEL file set, and the selected Tesseract release and checksums. Build preparation consumes this lock file rather than downloading a moving `main` revision. The generated `models/manifest.json` records the corresponding installed files for runtime validation.

The portable directory must be extracted to a writable location. The launcher will not silently redirect generated files elsewhere if the directory is read-only; it will report the exact failing path and exit.

## Executable behavior

Running `ModuleToFlashcards.exe` without arguments performs these steps:

1. Resolve the application root from the executable location rather than the process working directory.
2. Create and validate the `data` subdirectories.
3. Configure Hugging Face and Transformers for offline-only operation.
4. Validate the required model, Tesseract, and runtime assets against the manifest.
5. Detect PyTorch CUDA access and llama.cpp GPU-offload support.
6. Print the selected execution devices and relevant warnings in the console.
7. Check whether the configured port is available.
8. Start the FastAPI application on `127.0.0.1:8000`.
9. Keep the console visible for progress, errors, and shutdown instructions.

The executable will support these packaging-oriented options:

- `--port <number>` changes the local listening port.
- `--verify` performs full bundle and model verification, prints a result, and exits without starting the server.
- `--version` prints the application version and exits.

If port 8000 already hosts this application's healthy endpoint, a second invocation reports that the service is already running and exits successfully. If another program owns the port, startup fails with a clear instruction to select another port.

`Ctrl+C` requests a clean server shutdown. Supported Windows console-close events also request shutdown. Abrupt termination remains recoverable because the existing pipeline validates artifacts and resumes only from complete stages.

## Runtime path abstraction

A focused runtime-path module will be the single authority for application paths. It will distinguish source mode from frozen mode and expose explicit locations for:

- application root;
- bundled model root;
- bundled Tesseract executable and language data;
- writable data root;
- uploads, outputs, temporary files, and logs.

The API server and ingestion/model loaders will consume these resolved paths rather than deriving writable locations from `__file__` independently. Tests will inject a temporary application root so no packaging test writes into the repository or user profile.

## Offline model resolution

### Qwen

The bundle will include the exact `qwen2.5-3b-instruct-q5_k_m.gguf` file expected by the application. Source/developer mode may retain the existing explicit download preparation workflow, but frozen mode must never call `hf_hub_download`. Frozen mode resolves the model only from `models/` and fails startup if it is absent or invalid.

The CUDA-enabled `llama-cpp-python` build must report GPU-offload support. The application retains `n_gpu_layers=-1` so all supported layers are offloaded to the RTX 5070. Failure to initialize GPU offload produces a visible warning and retries Qwen with `n_gpu_layers=0` only when CPU fallback is enabled.

### REBEL

The bundle will contain a complete local snapshot of `Babelscape/rebel-large`, including model weights, configuration, tokenizer files, and SentencePiece assets required by `transformers`. `AutoTokenizer` and `AutoModelForSeq2SeqLM` will receive the local directory with `local_files_only=True`; they will not receive the repository identifier in frozen mode.

Device selection remains `cuda` when `torch.cuda.is_available()` is true and falls back to `cpu` with a visible warning otherwise. The Qwen runtime is released before REBEL loads, and REBEL is released before the second Qwen stage, preserving the current batch memory lifecycle.

### Tesseract

The build will copy a portable Tesseract distribution and English language data under `tesseract/`. Frozen mode sets `pytesseract.pytesseract.tesseract_cmd` to the bundled executable and `TESSDATA_PREFIX` to the bundled `tessdata` directory. The program will not search system installation folders before using the bundled copy.

## Offline enforcement

Before importing model libraries, the launcher will configure the process for offline model access, including `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. All frozen-mode Transformers loads also use `local_files_only=True`. Runtime code will not contain a frozen-mode fallback that contacts Hugging Face.

Offline acceptance testing must run with the network disconnected or blocked and begin from a clean user Hugging Face cache. Passing because a model happened to exist in a user cache is not sufficient.

## Model manifest and integrity checks

`models/manifest.json` will identify each required model asset, its relative path, expected byte size, and SHA-256 digest. Tesseract and critical runtime files will also have manifest entries or equivalent build-time checks. Its expected values originate from the checked-in release lock rather than being accepted from whatever files happen to exist in the staging directory.

Normal startup checks existence, expected size, and a cached successful integrity record. A changed size or modification time invalidates that record and triggers a new hash check. `--verify` always recomputes every recorded digest. This avoids hashing several gigabytes on every launch while still detecting missing, replaced, or corrupted assets.

The build fails before producing a release directory if any staged asset is absent or differs from the build manifest.

## Data flow and storage

Uploaded PDFs are written to `data/uploads/<course-code>/`. Per-module structured text, knowledge graphs, flashcards, and resumable manifests are written beneath `data/pipeline_output/` using the existing sanitized course and module names. Temporary request files use `data/temporary/`, and application logs use `data/logs/`.

Paths returned by the API refer to their final absolute locations inside the portable `data` directory. The source PDFs are not edited or deleted. Existing output-validation and resume rules continue to determine whether completed stages may be reused.

Only one heavy batch executes at a time, matching the existing request lock and sequential staging. Packaging will not introduce parallel model loads.

## Error handling and diagnostics

Startup validation failures are printed to the console, written to a timestamped log, and returned through a nonzero process exit code. Messages identify the failed component and its expected path without exposing arbitrary environment data.

Required cases include:

- portable directory or `data` directory is not writable;
- Qwen file is missing or fails integrity verification;
- REBEL snapshot is incomplete or invalid;
- bundled Tesseract or language data is missing;
- CUDA-enabled PyTorch is unavailable;
- llama.cpp lacks GPU-offload support;
- CUDA initialization runs out of VRAM;
- the configured port belongs to another application;
- the FastAPI server cannot start.

CUDA availability failures warn and use CPU fallback. Missing or corrupt models do not fall back to downloads and are fatal. A module-generation failure remains isolated to that module and is returned through the existing batch API error structure.

The console displays the application version, data location, Qwen device, REBEL device, server URL, and log path. It must not dump slide content, generated flashcards, or full user paths beyond paths needed for diagnosis into routine startup logs.

## Build system

The repository will gain a reproducible PowerShell build entry point and a PyInstaller specification. The build has two explicit phases:

1. **Prepare assets:** obtain and stage the locked Qwen file, the immutable pinned REBEL snapshot, the locked portable Tesseract release, license notices, and a target-compatible CUDA-enabled environment. Each asset is verified against `packaging/model-lock.json`. Network use is allowed only during this build/preparation phase.
2. **Build and assemble:** freeze the console launcher, collect package data and native DLLs, copy staged model/Tesseract assets beside the executable, write the manifest, run verification, and produce a versioned portable archive.

The build script removes or replaces only its own named staging and distribution directories after resolving and validating that they are inside the repository. It does not modify source models or user-generated data.

Dependency versions used for a release will be locked separately from the broad developer `requirements.txt`. A release records the Python, PyInstaller, PyTorch, Transformers, llama-cpp-python, CUDA runtime, and model revision versions used to build it.

The final build must be created from a CUDA-enabled environment compatible with the RTX 5070. The build preflight and target acceptance checks must demonstrate both `torch.cuda.is_available()` and `llama_cpp.llama_supports_gpu_offload()` as true before the release is labeled GPU-ready.

## Testing strategy

### Automated source tests

- Source and frozen application-root resolution.
- Creation and selection of every `data` subdirectory.
- Rejection of read-only or invalid application roots.
- Frozen Qwen resolution never invokes a downloader.
- Frozen REBEL loading uses the bundled path and `local_files_only=True`.
- Bundled Tesseract takes precedence in frozen mode.
- Manifest generation, cached verification, forced full verification, and corruption detection.
- Port handling for available, same-application, and foreign-application cases.
- CPU fallback and fatal startup-error reporting.
- Launcher argument parsing for default startup, `--port`, `--verify`, and `--version`.
- Existing API and pipeline tests continue to pass with injected paths.

### Packaging smoke test

- Build the one-directory executable.
- Start it from a path containing spaces.
- Confirm the visible console reports the expected version and data directory.
- Confirm `GET /health` succeeds.
- Confirm `--verify` succeeds using only bundled assets.
- Confirm no external Python installation is required.

### RTX 5070 acceptance test

On the target Ryzen 7 5800X/RTX 5070 machine:

1. Confirm the NVIDIA driver detects the RTX 5070.
2. Confirm PyTorch CUDA and llama.cpp GPU offload are both enabled.
3. Disconnect or block network access and clear/redirect user model caches.
4. Start the executable and confirm Qwen and REBEL select CUDA.
5. Process a representative 20-slide PDF.
6. Verify all expected artifacts are under the adjacent `data` directory.
7. Parse the result and confirm exactly 100 flashcards, split into two valid 50-row blocks.
8. Resume the same PDF and confirm valid completed stages are reused.
9. Record stage and total elapsed times as the first trustworthy GPU benchmark.

## Security and privacy

The server binds only to `127.0.0.1`; the packaging change will not expose it to the LAN. PDF and generated content remain local. Offline inference is enforced for bundled model loads. Existing filename/course sanitization remains in place, and runtime paths are resolved beneath the designated portable data root.

Third-party model, CUDA runtime, Python package, and Tesseract licenses and notices must be reviewed and included before distributing the bundle outside the development team.

## Completion criteria

The feature is complete when:

- the packaging-specific automated tests pass;
- the existing repository test suite passes;
- a one-directory Windows bundle can be built reproducibly;
- the bundle starts through `ModuleToFlashcards.exe` without system Python;
- all model loads succeed with networking disabled and clean external caches;
- generated artifacts and logs remain under the adjacent `data` folder;
- startup and `--verify` detect damaged or missing assets;
- the RTX 5070 acceptance test proves CUDA is active for Qwen and REBEL; and
- one representative module produces exactly 100 validated flashcards in two blocks.
