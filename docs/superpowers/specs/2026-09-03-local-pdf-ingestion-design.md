# Local PDF Ingestion and End-to-End Pipeline Design

**Status:** Approved 2026-09-03

## Goal

Insert a fully local PDF-to-structured-text stage before the existing knowledge graph and flashcard stages. Replace every Mistral dependency from the supplied converter with local extraction plus the already selected `Qwen2.5-3B-Instruct Q8_0` model.

## Sequence

1. PDF slides to validated structured UTF-8 text.
2. Structured text to a REBEL knowledge graph.
3. Knowledge graph to validated assessment flashcards with local Qwen.

Each stage writes its own artifact so a later failure can resume without repeating a completed earlier stage.

## Stage 1: PDF to structured text

PyMuPDF extracts the ordered text layer from every page. If a page contains fewer than 40 meaningful alphanumeric characters, the page is rendered at 200 DPI and passed to local Tesseract OCR through `pytesseract`. If neither path produces readable text, the page is retained with `[Unreadable Text]`; the document fails only when every page is unreadable.

The installed local Qwen model receives one page at a time. It may repair obvious OCR spacing and character errors and organize the page, but it must not add outside facts or claim to understand images it cannot see. For OCR-derived pages, visible labels are placed in the visual-text field. For text-layer pages, the visual-text field is `Not Specified` unless the extracted source explicitly labels a figure, table, chart, or diagram.

Qwen must return JSON containing:

- slide title;
- ordered content;
- visual or OCR text;
- grounded definitions;
- grounded knowledge statements;
- a short grounded explanation;
- optional module-number and module-title candidates from the first slide.

Every response is parsed and validated before it enters the output document. Invalid JSON or invalid fields are retried with focused validation feedback. Explicit command-line module metadata takes precedence over Qwen candidates.

## Structured TXT contract

The stage-1 artifact is UTF-8 text with a stable version marker and bracketed records:

```text
[MODULE]
format_version: 1
course_code: CPE0021
module_number: 1
module_title: Example Module
source_file: module.pdf
[/MODULE]

[SLIDE 1]
extraction_method: text
[TITLE]
Example title
[/TITLE]
[CONTENT]
- First visible statement.
[/CONTENT]
[VISUAL_TEXT]
Not Specified
[/VISUAL_TEXT]
[DEFINITIONS]
- Term :: Definition
[/DEFINITIONS]
[KNOWLEDGE_STATEMENTS]
- A complete grounded sentence.
[/KNOWLEDGE_STATEMENTS]
[BRIEF_EXPLANATION]
A short explanation based only on the page.
[/BRIEF_EXPLANATION]
[/SLIDE]
```

Field contents are normalized so control tags cannot be injected into the record structure. The knowledge-graph extractor recognizes this contract, reads its metadata, removes structural tags, and sends only meaningful learning content into REBEL.

## Stage 2: structured text to graph

The existing REBEL extractor remains the relationship-extraction engine. It gains structured-module metadata parsing and copies `course_code`, `module_number`, `module_title`, format version, and source file into `knowledge_graph.json`. Plain unstructured TXT input remains supported.

## Stage 3: graph to flashcards

The existing Qwen flashcard generator remains unchanged except for being callable from the end-to-end runner. It receives exact identity values from graph metadata or command-line overrides and preserves its deterministic validation and atomic output behavior.

## Commands

Stage 1 alone:

```powershell
python slides_pdf_to_txt.py module.pdf --course-code CPE0021 --module-number 1
```

All stages:

```powershell
python pipeline.py module.pdf --course-code CPE0021 --module-number 1
```

The runner creates one workspace under `pipeline_output/<pdf-stem>/` containing `structured_module.txt`, `knowledge_graph/knowledge_graph.json`, `knowledge_graph/triples.csv`, and `flashcards.txt`.

By default, an existing valid stage artifact is reused. `--force` recomputes all stages. The source PDF is never modified or deleted.

## Errors and safety

- Missing PyMuPDF, Pillow, pytesseract, Tesseract, PyTorch, Transformers, or SentencePiece produces a focused setup message.
- The PDF is validated before the Qwen model is downloaded or loaded.
- A failed stage returns a nonzero exit code and leaves completed stage artifacts intact.
- Stage outputs are written through temporary sibling files and atomically replaced only after validation.
- No document content is uploaded to a network service. Internet access is needed only for first-time dependency and model downloads.

## Testing

Unit tests cover TXT rendering and metadata parsing, Qwen response validation and retry behavior, PDF text/OCR selection using injected page adapters, source-file preservation, command construction, resume behavior, and failure propagation. Tests do not load large models, invoke Tesseract, download dependencies, or run REBEL.
