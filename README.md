# Module to Flashcards

This project turns slide PDFs into structured text, extracts a relationship graph, and uses a local Qwen model to generate validated, copy-paste-ready assessment CSV.

The flashcard generator uses the official Hugging Face repository `Qwen/Qwen2.5-3B-Instruct-GGUF` and the exact file `qwen2.5-3b-instruct-q8_0.gguf`. The first run downloads the approximately 3.6 GB model into `models/`. Later runs reuse that file.

The complete local sequence is:

1. PyMuPDF extracts each PDF slide's embedded text.
2. Sparse or image-only slides use local Tesseract OCR.
3. Qwen organizes the extracted page text into a validated structured TXT file.
4. REBEL converts the structured learning content into a knowledge graph.
5. Qwen generates and validates the flashcards from graph relationships.

No PDF or extracted document content is uploaded to Mistral or another API. The selected Qwen model is text-only: Tesseract recovers visible labels from slide images, but Qwen does not perform visual interpretation of diagrams or photographs.

## Requirements

- Windows, macOS, or Linux
- Python 3.11 or 3.12 recommended for the easiest `llama-cpp-python` installation
- Tesseract OCR available on `PATH` or installed in its standard Windows folder
- Several gigabytes of free disk space
- Internet access for initial dependency and model downloads

CPU-only inference works but can be slow because a complete module requires concept planning, 20 cluster-generation calls, validation retries, and six review calls. GPU acceleration requires a `llama-cpp-python` build compatible with the installed GPU runtime.

## Set up on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Tesseract OCR if it is not already available. On Windows, one common installation is:

```powershell
winget install --exact --id UB-Mannheim.TesseractOCR
```

Verify the installation:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

If the installer does not add Tesseract to `PATH`, the project also checks the standard Windows installation folders automatically. Tesseract is only invoked when a PDF page has too little embedded text. A text-layer PDF can complete stage 1 without OCR.

The requirements file uses the project's official CPU-wheel index so Windows does not need to compile `llama-cpp-python` from source. GPU users can replace that package with a CUDA, Vulkan, or other accelerated wheel supported by their hardware.

If `llama-cpp-python` tries to compile and fails, install the Visual Studio C++ Build Tools or install an official prebuilt wheel matching the computer's CPU or CUDA environment. Python 3.11 or 3.12 generally has broader native-wheel compatibility than a newly released Python version.

## Run the complete sequence

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" --course-code CPE0021 --module-number 1
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
python text-extractor.py sample.txt --output-dir output
```

If `output/knowledge_graph.json` already exists, skip this extraction step.

## Generate flashcards

Run stage 3 directly:

```powershell
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1
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
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --smoke-test
```

A successful run prints:

```text
Qwen smoke test passed.
```

## Useful options

Choose another output path:

```powershell
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --output flashcards\cpe0021-module-1.txt
```

Skip the six final model-assisted reviews to reduce runtime:

```powershell
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --skip-final-review
```

Skipping review retains deterministic schema and duplicate checks, but reduces semantic grounding and duplication assurance.

Use CPU only:

```powershell
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --n-gpu-layers 0
```

See all options:

```powershell
python main.py --help
```

## Identity precedence

- Course code: the `--course-code` value is preferred; graph metadata is the fallback.
- Module number: graph metadata is preferred; `--module-number` is the fallback.

Both values are preserved as exact strings. If a required value is unavailable, the program reports only the missing value.

## Tests

The default tests never download or load the model:

```powershell
python -m pytest -q
```

## Privacy and output behavior

After the first model download, graph processing and assessment generation run locally. The program sends only graph relationship triples to Qwen, excluding stored evidence chunks, slide numbers, filenames, and other provenance. It assembles the complete result in memory and replaces the destination atomically only after final validation succeeds.
