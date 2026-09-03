# Module to Flashcards

This project turns slide PDFs into structured text, extracts a relationship graph, and uses a local Qwen model to generate validated, copy-paste-ready assessment CSV.

Current release: 1.1.0

The flashcard generator uses the official Hugging Face repository `Qwen/Qwen2.5-3B-Instruct-GGUF` and the exact file `qwen2.5-3b-instruct-q5_k_m.gguf`. The first run downloads it into `models/`; later runs reuse that local Q5_K_M model. Qwen uses an 8K-token context window by default.

The complete local sequence is:

1. PyMuPDF extracts each PDF slide's embedded text.
2. Sparse or image-only slides use local Tesseract OCR.
3. Qwen organizes the extracted page text into a validated structured TXT file.
4. REBEL converts the structured learning content into a knowledge graph.
5. Qwen generates and validates the flashcards from graph relationships.

No PDF or extracted document content is uploaded to Mistral or another API. The selected Qwen model is text-only: Tesseract recovers visible labels from slide images, but Qwen does not perform visual interpretation of diagrams or photographs.

## Git and release guidance

The matching annotated Git release tag for version 1.1.0 should be `v1.1.0`.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or 3.12 recommended for the easiest `llama-cpp-python` installation
- A CUDA-capable GPU with a GPU-enabled `llama-cpp-python` build for acceleration
- Tesseract OCR available on `PATH` or installed in its standard Windows folder
- Several gigabytes of free disk space
- Internet access for initial dependency and model downloads

GPU offload is enabled by default with `n_gpu_layers=-1`, and REBEL selects CUDA automatically when PyTorch can access it. GPU acceleration requires a GPU-enabled `llama-cpp-python` build compatible with the installed GPU runtime. CPU-only inference remains supported but can be slow because a complete module requires concept planning, 20 cluster-generation calls, validation retries, and, in full-quality CLI mode, six optional final review calls. API fast mode skips those optional reviews.

## Quick start with the browser extension

Run these steps from the project directory in Windows PowerShell:

1. Install the project dependencies once:

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

2. Start the local processor and leave it running:

  ```powershell
  .\.venv\Scripts\python.exe api_server.py
  ```

3. Open the Paraverse course page, select the PDFs in the extension panel,
  and click **Process Selected**. The extension sends them to the local
  server at `http://localhost:8000/process/<course-code>`.

Confirm that the server is ready before using the extension:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

The server accepts multiple selected PDFs in one request and writes results under
`pipeline_output/<sanitized-course-code>/<pdf-name>/`. Keep the server terminal open while processing.
There is no default processing timeout: `--timeout 0` means unlimited time.

For a multi-file request, the server stages work to reduce memory pressure and
model reloads: it normalizes all pending PDFs with one Qwen load, builds all
pending knowledge graphs with one REBEL load, then generates flashcards with a
second Qwen load. Heavy inference is sequential, not parallel. If one module
fails in a stage, it is reported in `errors` and does not block the other
modules from continuing. Re-send the same file to resume: valid completed
artifacts are reused, while incomplete downstream stages run again.

## One-time setup on Windows PowerShell

Install the Python dependencies once in a project virtual environment. Do not
run the install command each time you generate flashcards; later runs only use
the already configured environment and reuse the downloaded model.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Keep the virtual environment activated while using the project. If it is not
activated, call the same interpreter explicitly with
`.\.venv\Scripts\python.exe`.

Install Tesseract OCR if it is not already available. On Windows, one common installation is:

```powershell
winget install --exact --id UB-Mannheim.TesseractOCR
```

Verify the installation:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

If the installer does not add Tesseract to `PATH`, the project also checks the standard Windows installation folders automatically. Tesseract is only invoked when a PDF page has too little embedded text. A text-layer PDF can complete stage 1 without OCR.

The default configuration requests GPU acceleration. Install the `llama-cpp-python` wheel matching your GPU backend before running the project. CUDA users should follow the official `llama-cpp-python` installation instructions for their CUDA version; the correct wheel must be installed in `.venv` for `n_gpu_layers=-1` to use the GPU. If a GPU-enabled build is unavailable, the program falls back only when you explicitly run with `--n-gpu-layers 0` and `--kg-device cpu`.

If `llama-cpp-python` tries to compile and fails, install the Visual Studio C++ Build Tools or install an official prebuilt wheel matching the computer's CPU or CUDA environment. Python 3.11 or 3.12 generally has broader native-wheel compatibility than a newly released Python version.

## Connect the browser extension

Start the local API once after setup and leave this terminal open while using
the Paraverse extension:

```powershell
.\.venv\Scripts\python.exe api_server.py
```

The server listens at `http://localhost:8000`. Your extension's
`background.js` sends selected files to:

```text
POST http://localhost:8000/process/<course-code>
```

The endpoint accepts the extension's multipart `files` fields, processes each
PDF locally, and returns `outputs` and `errors`. It uses fast API defaults for
interactive batches: REBEL batch size `1`, one beam, and no optional final
flashcard review. Names such as
`CPE0021-M1.pdf` automatically select module number `1`; files without an
`M<number>` marker use module number `1`.

If your extension has a `manifest.json`, include localhost in its background
host permissions:

```json
"host_permissions": [
  "https://paraverse.feutech.edu.ph/*",
  "http://localhost:8000/*"
]
```

Check the connection before processing files:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

The extension's Local port field can be changed from `8000`, but the server
must then be started with the matching port, for example:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_server:app --host 127.0.0.1 --port 8010
```

## Run the complete sequence

After one-time setup, from the project directory, run the full-quality CLI
configuration. The CLI defaults are REBEL batch size 4 and three beams:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" `
  --course-code CPE0021 --module-number 1 `
  --kg-batch-size 4 --kg-num-beams 3
```

The command creates a per-PDF workspace:

```text
pipeline_output/
  module/
    structured_module.txt
    knowledge_graph/
      knowledge_graph.json
      triples.csv
    flashcards.txt
```

If the command stops, run it again. Valid completed artifacts are reused. Use `--force` to recompute all stages:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" --course-code CPE0021 --module-number 1 --force
```

CPU-only Qwen and REBEL inference can take a long time for a full module. Force CPU execution when needed:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" --course-code CPE0021 --module-number 1 --n-gpu-layers 0 --kg-device cpu
```

The source PDF is never changed or deleted.

The command-line timeout is disabled by default (`--timeout 0`). Set a positive
number of seconds only when you need a limit for one module:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" --course-code CPE0021 --module-number 1 --timeout 600
```

## Run stage 1 only

Create only the structured TXT artifact:

```powershell
.\.venv\Scripts\python.exe slides_pdf_to_txt.py "C:\path\to\module.pdf" --course-code CPE0021 --module-number 1
```

The default destination is `structured_text/<pdf-name>.txt`. Useful options include `--module-title`, `--output`, `--ocr-min-chars`, `--ocr-dpi`, `--n-gpu-layers`, and `--n-ctx`.

The file uses a stable, machine-readable plain-text contract with `[MODULE]`, `[SLIDE n]`, `[CONTENT]`, `[DEFINITIONS]`, `[KNOWLEDGE_STATEMENTS]`, and related sections. The knowledge-graph stage removes these control tags and preserves their meaningful learning content.

## Run the knowledge-graph stage only

The included extractor accepts a UTF-8 text module:

```powershell
.\.venv\Scripts\python.exe text-extractor.py sample.txt --output-dir output
```

If `output/knowledge_graph.json` already exists, skip this extraction step.

## Generate flashcards

Run stage 3 directly:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1
```

The generated file defaults to:

```text
flashcards/module_1.txt
```

It contains two labeled CSV blocks:

- `Module 1.1`: ten complete clusters and 50 rows
- `Module 1.2`: ten complete clusters and 50 rows

Every successful module contains 20 graph-supported concepts, five questions per concept, 20 UUID clusters, and exactly 100 questions. The program stops without writing partial output if it cannot validate those requirements.

## First-run model smoke test

Before generating all 100 questions, download and load the model with a tiny JSON request:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --smoke-test
```

A successful run prints:

```text
Qwen smoke test passed.
```

## Useful options

Choose another output path:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --output flashcards\cpe0021-module-1.txt
```

The CLI keeps its final model-assisted review enabled by default. To skip the
five grounding reviews and global duplicate review when runtime matters:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --skip-final-review
```

Skipping review retains deterministic schema and duplicate checks, but reduces semantic grounding and duplication assurance.

Use CPU only:

```powershell
.\.venv\Scripts\python.exe main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --n-gpu-layers 0
```

See all options:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

## Identity precedence

- Course code: the `--course-code` value is preferred; graph metadata is the fallback.
- Module number: graph metadata is preferred; `--module-number` is the fallback.

Both values are preserved as exact strings. If a required value is unavailable, the program reports only the missing value.

## Tests

The default tests never download or load the model:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Privacy and output behavior

After the first model download, graph processing and assessment generation run locally. The program sends only graph relationship triples to Qwen, excluding stored evidence chunks, slide numbers, filenames, and other provenance. It assembles the complete result in memory and replaces the destination atomically only after final validation succeeds.
