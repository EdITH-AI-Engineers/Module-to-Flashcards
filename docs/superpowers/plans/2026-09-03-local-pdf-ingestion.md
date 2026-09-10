# Local PDF Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local PDF-to-structured-TXT stage and make it the first resumable stage of the knowledge-graph-to-flashcards sequence.

**Architecture:** Focused modules define the TXT contract, extract PDF page text with OCR fallback, normalize each page with the existing local Qwen backend, and orchestrate all three programs through subprocess boundaries. Stage artifacts are validated and atomically written so the runner can safely resume.

**Tech Stack:** Python 3.11/3.12, PyMuPDF, Pillow, pytesseract/Tesseract, llama-cpp-python, Qwen2.5-3B-Instruct Q8_0, Transformers/REBEL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-local-pdf-ingestion-design.md`

## Global Constraints

- All inference and document processing is local; no document content is uploaded.
- Reuse `Qwen/Qwen2.5-3B-Instruct-GGUF`, file `qwen2.5-3b-instruct-q8_0.gguf`.
- Preserve the source PDF and existing stage artifacts unless `--force` is supplied.
- Write UTF-8 stage output atomically only after validation.
- Keep plain unstructured TXT support in `text-extractor.py`.
- Tests must not load models, download files, invoke Tesseract, or run REBEL.
- This workspace has no Git repository metadata, so verification checkpoints replace commit steps.

---

### Task 1: Structured module contract

**Files:**
- Create: `structured_module.py`
- Create: `tests/test_structured_module.py`

**Interfaces:**
- Produces: `Definition`, `StructuredSlide`, `StructuredModule`, `render_structured_module(module) -> str`, `parse_module_metadata(text) -> dict[str, str]`, and `graph_ready_text(text) -> str`.
- Consumes: only Python standard-library dataclasses, regular expressions, and JSON-safe text normalization.

- [ ] **Step 1: Write failing rendering and parsing tests**

  Cover the exact `[MODULE]` and `[SLIDE n]` contract, tag neutralization inside values, metadata round-trip, exclusion of structural tags from graph-ready text, and rejection of a module with no slides.

- [ ] **Step 2: Verify the tests fail because `structured_module` does not exist**

  Run `python -m pytest tests/test_structured_module.py -q` and confirm collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable records, validation, renderer, metadata parser, and graph-ready projection**

  Normalize line endings; replace standalone control-tag-shaped lines in field content; require positive slide numbers and at least one non-unreadable content field across the module; render every approved field in the specified order.

- [ ] **Step 4: Verify Task 1**

  Run `python -m pytest tests/test_structured_module.py -q` and require zero failures.

### Task 2: Qwen slide normalization

**Files:**
- Create: `slide_normalizer.py`
- Create: `tests/test_slide_normalizer.py`

**Interfaces:**
- Consumes: an object exposing `complete(system: str, user: str, *, max_tokens: int) -> str`, page source text, extraction method, and optional identity overrides.
- Produces: `normalize_slide(...) -> StructuredSlide`, `normalize_document(...) -> StructuredModule`, and `SlideNormalizationError`.

- [ ] **Step 1: Write failing response-validation tests**

  Cover a valid JSON response, fenced JSON extraction, invalid list/object shapes, outside-control-tag neutralization, explicit identity precedence, grounded retry feedback, and exhaustion after the configured attempts.

- [ ] **Step 2: Verify the tests fail for the missing module**

  Run `python -m pytest tests/test_slide_normalizer.py -q` and confirm collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the strict prompts, JSON parser, field validators, retries, and document assembly**

  Require strings or lists of strings as specified, reject empty title/content combinations, coerce definitions only from `{term, definition}` objects, and select first-slide metadata only when an explicit value was not supplied.

- [ ] **Step 4: Verify Task 2**

  Run `python -m pytest tests/test_slide_normalizer.py -q` and require zero failures.

### Task 3: Local PDF extraction with OCR fallback

**Files:**
- Create: `pdf_ingestion.py`
- Create: `tests/test_pdf_ingestion.py`

**Interfaces:**
- Produces: `ExtractedPage(number: int, text: str, method: str)`, `extract_pdf_pages(path, *, min_chars=40, dpi=200, document_factory=None, ocr=None) -> tuple[ExtractedPage, ...]`, and `PdfExtractionError`.
- Consumes: PyMuPDF at runtime and a lazily imported pytesseract/Pillow OCR adapter only for sparse pages.

- [ ] **Step 1: Write failing extraction-selection tests**

  Use injected fake documents/pages to cover ordered text extraction, OCR fallback below 40 alphanumeric characters, unreadable-page retention, all-pages-unreadable failure, missing file, non-PDF rejection, and proof that the source file still exists.

- [ ] **Step 2: Verify the tests fail for the missing module**

  Run `python -m pytest tests/test_pdf_ingestion.py -q` and confirm collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement lazy dependency adapters and page extraction**

  Use `page.get_text("text", sort=True)`, `page.get_pixmap(dpi=dpi, alpha=False)`, Pillow byte decoding, and `pytesseract.image_to_string`; translate missing binary errors into one focused installation message.

- [ ] **Step 4: Verify Task 3**

  Run `python -m pytest tests/test_pdf_ingestion.py -q` and require zero failures.

### Task 4: Stage-1 command

**Files:**
- Create: `slides_pdf_to_txt.py`
- Create: `tests/test_slides_pdf_to_txt.py`

**Interfaces:**
- Consumes: `extract_pdf_pages`, `ensure_model`, `LocalQwenBackend`, `normalize_document`, and `render_structured_module`.
- Produces: `parse_args(argv=None)`, `run(args) -> Path`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing CLI and orchestration tests**

  Cover argument preservation, default output naming, extraction before model loading, backend configuration, atomic output, invalid-PDF early failure, and no source deletion.

- [ ] **Step 2: Verify the tests fail for the missing command module**

  Run `python -m pytest tests/test_slides_pdf_to_txt.py -q` and confirm collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the stage-1 CLI and orchestration**

  Accept `--course-code`, `--module-number`, `--module-title`, `--output`, `--model-dir`, `--attempts`, `--seed`, `--n-gpu-layers`, `--n-ctx`, `--ocr-min-chars`, and `--ocr-dpi`; print page-level progress and atomically replace the completed TXT.

- [ ] **Step 4: Verify Task 4**

  Run `python -m pytest tests/test_slides_pdf_to_txt.py -q` and require zero failures.

### Task 5: Knowledge-graph structured-input support

**Files:**
- Modify: `text-extractor.py`
- Create: `tests/test_text_extractor_structured.py`

**Interfaces:**
- Consumes: `parse_module_metadata(text)` and `graph_ready_text(text)` from `structured_module.py`.
- Produces: `build_graph(..., module_metadata=None)` with structured metadata copied into graph metadata while retaining its existing default call behavior.

- [ ] **Step 1: Write failing structured-input graph tests**

  Load `text-extractor.py` with `importlib`, then cover graph-ready projection, metadata propagation, and unchanged behavior for a plain TXT source.

- [ ] **Step 2: Verify the new tests fail for absent structured integration**

  Run `python -m pytest tests/test_text_extractor_structured.py -q` and confirm assertions fail because metadata is absent or tags remain.

- [ ] **Step 3: Integrate structured parsing into `main` and graph construction**

  Detect the exact `[MODULE]` marker, project structured input into meaningful content before chunking, and merge only the approved metadata keys into the graph metadata object.

- [ ] **Step 4: Verify Task 5 and legacy extractor tests**

  Run `python -m pytest tests/test_text_extractor_structured.py tests/test_module_validation.py -q` and require zero failures.

### Task 6: Resumable end-to-end runner

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline_runner.py`

**Interfaces:**
- Consumes: three Python command entry points and their filesystem artifacts.
- Produces: `PipelinePaths`, `build_stage_commands(args, paths)`, `run(args, command_runner=subprocess.run) -> PipelinePaths`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing path, command, resume, force, and propagation tests**

  Cover sanitized PDF stem workspaces, exact command arguments, skipping valid existing artifacts, recomputing with `--force`, stage order, and immediate stop when a subprocess fails.

- [ ] **Step 2: Verify the tests fail for the missing runner**

  Run `python -m pytest tests/test_pipeline_runner.py -q` and confirm collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the subprocess runner**

  Invoke every stage with `sys.executable`, use absolute script and artifact paths, validate each expected output after its stage, pass identity and local-model arguments to the relevant commands, and leave successful artifacts available for resumption.

- [ ] **Step 4: Verify Task 6**

  Run `python -m pytest tests/test_pipeline_runner.py -q` and require zero failures.

### Task 7: Dependencies, setup, and full verification

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- Documents the stage-1 and one-command interfaces and declares the Python dependencies needed for extraction and OCR.

- [ ] **Step 1: Add a documentation expectation test**

  Extend `tests/test_pipeline_runner.py` to assert `pipeline.py --help` exposes the PDF positional input, identity fields, stage output root, resume semantics, and `--force`.

- [ ] **Step 2: Verify the expectation fails before documentation and parser updates**

  Run the focused test and confirm one or more expected help strings are absent.

- [ ] **Step 3: Update dependency and user documentation**

  Add `PyMuPDF`, `Pillow`, and `pytesseract`; document the Tesseract executable requirement, PowerShell setup, artifact layout, stage-only and end-to-end examples, CPU runtime warning, and source preservation. Ignore generated `pipeline_output/`, `structured_text/`, and progress/temp artifacts.

- [ ] **Step 4: Run complete automated verification**

  Run `python -m pytest -q` and require zero failures.

- [ ] **Step 5: Run syntax and CLI verification**

  Run `python -m compileall -q .` excluding `.venv` and `models`, then run `python slides_pdf_to_txt.py --help` and `python pipeline.py --help`; require exit code zero for all three commands.

- [ ] **Step 6: Run a dependency-free stage smoke test**

  Use a tiny generated text-layer PDF only if PyMuPDF is installed; otherwise report the missing runtime dependency precisely and retain the fully mocked automated coverage. Do not claim live PDF/model integration without successful evidence.
