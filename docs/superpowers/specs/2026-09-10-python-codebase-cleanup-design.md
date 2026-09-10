# Python Codebase Cleanup and Artifact Reliability Design

- **Status:** Approved in chat; awaiting written-spec review
- **Date:** 2026-09-10
- **Scope:** Behavior-preserving cleanup of the Python pipeline, with targeted fixes for stale normalized/graph artifacts and noisy concept-plan failures

## Context

Module to Flashcards now performs PDF ingestion, slide normalization, knowledge-graph extraction, concept planning, flashcard generation, API serving, and portable-runtime startup. The public workflows work, but several production modules combine orchestration, persistence, schema validation, parsing, retry behavior, and domain transformations. This makes changes harder to review and allows duplicated or overly permissive validation rules to drift apart.

The most important reliability issue is artifact reuse. Existing normalized text still uses `format_version: 1`, and the pipeline accepts any graph with usable relationship edges. When a newer executable is copied over an older portable `data` directory, it can reuse stale normalized text or a sparse graph that lacks the richer lesson facts now needed by concept planning. That can lead to repeated `fact_ids must be a non-empty JSON array` errors even though the current source code supports grounded lesson facts.

The cleanup must preserve the interfaces used by the CLI, FastAPI server, tests, and portable launcher. It is not an opportunity for a broad rewrite or a new storage system.

## Goals

- Preserve the existing CLI commands, FastAPI routes, output locations, and portable double-click workflow.
- Make current-stage artifact requirements explicit and centrally testable.
- Regenerate stale normalized and graph outputs in dependency order instead of silently reusing them.
- Keep readers tolerant enough to inspect legacy graph files outside pipeline resume decisions.
- Separate domain parsing and validation from orchestration and file I/O.
- Reduce repeated helpers and duplicated JSON/path conversion logic.
- Make concept-plan failures concise, actionable, and safe for API responses and console logs.
- Improve maintainability without changing the 20-concept/100-card contract.
- Keep each refactoring step covered by tests and small enough to review independently.

## Non-goals

- Replacing JSON, CSV, or structured text with a database.
- Changing public API endpoints, CLI flags, output filenames, or directory layout.
- Changing the flashcard count, card schema, spelling of existing serialized compatibility fields, or assessment rules.
- Replacing Qwen, REBEL, Tesseract, FastAPI, or the portable packaging system.
- Introducing network services, migrations requiring user interaction, or deletion of uploaded PDFs.
- Reformatting every Python file solely for stylistic uniformity.

## Considered approaches

### 1. Cosmetic cleanup only

Rename helpers, sort imports, and remove obvious duplication without changing module boundaries or artifact rules. This has the lowest immediate risk, but it leaves the stale-artifact defect and the largest mixed-responsibility modules intact.

### 2. Focused modular cleanup with compatibility boundaries

Extract shared artifact and domain responsibilities, add explicit schema-aware resume validation, simplify retry diagnostics, and retain compatibility wrappers at existing import locations. This addresses the observed failures while limiting the change surface. This is the selected approach.

### 3. Full pipeline rewrite

Replace the staged orchestration and formats with a new architecture. This could produce a cleaner theoretical design, but it would carry excessive regression and portable-release risk and would make existing output compatibility difficult to guarantee.

## Compatibility contract

The cleanup treats the following as stable public behavior:

- Existing CLI entry points and arguments continue to work.
- Existing FastAPI routes and response shapes continue to work.
- `ModuleToFlashcards.exe` keeps the same startup and data-directory behavior.
- Current output filenames and course/module directory layout remain unchanged.
- Existing graph files remain loadable by direct graph/flashcard commands when they contain usable legacy edges.
- Existing flashcard output remains parseable with the current card contract.
- Public functions imported by tests or scripts remain available at their current modules, either as implementations or thin forwarding wrappers.

Pipeline resume compatibility is intentionally stricter than manual read compatibility. A legacy artifact can be readable without being eligible for reuse by a current pipeline stage.

## Proposed architecture

### Artifact contracts

A focused artifact-contract module will own current schema identifiers and reuse eligibility. It will expose side-effect-free checks for:

- normalized structured text;
- knowledge-graph JSON;
- rendered flashcard output; and
- any metadata required to prove that an artifact was produced by the current transformation contract.

The normalized-text writer will emit a new current format version. The graph writer will emit an explicit graph schema version in metadata and will include a non-empty top-level `facts` array for a graph eligible for current pipeline reuse. A current graph fact must have a stable non-empty ID and statement; slide/topic provenance remains optional per fact but aggregate fact coverage is recorded in metadata.

Legacy readers remain tolerant. The pipeline resume validator, however, requires the current schema and rich-fact contract. Therefore an old graph with valid REBEL edges may still be opened manually but is marked stale by the pipeline and rebuilt from the normalized module.

### Atomic artifact I/O

Shared JSON helpers will provide:

- recursive conversion of `Path` and other explicitly supported values to JSON-safe primitives;
- UTF-8 reading with clear errors;
- atomic writes through a temporary sibling followed by replacement; and
- consistent formatting for human-inspectable files.

Only persistence code will use these helpers. Domain parsers will accept strings or mappings and return typed values without performing disk writes. Existing specialized CSV and structured-text writers will remain specialized rather than being forced into a generic abstraction.

### Normalized lesson facts

Lesson-fact extraction will be isolated from structured-module rendering. It will own:

- extracting definitions, knowledge statements, and content fallbacks;
- filtering presentation-only noise;
- normalizing whitespace and deduplication keys;
- retaining slide and topic provenance; and
- assigning deterministic fact IDs in document order.

`structured_module.extract_lesson_facts` will remain available as a compatibility entry point. It will delegate to the focused implementation so existing imports do not break.

### Knowledge-graph assembly

Graph assembly will clearly separate three operations:

1. transform model triples into normalized edges;
2. attach normalized lesson facts and provenance; and
3. serialize the complete graph artifact.

Graph facts are the preferred grounding source for concept planning. Legacy edges are a read-time fallback only. Graph metadata will report schema version, fact count, and source/normalization context needed by reuse validation. Duplicate nodes, edges, evidence records, and facts will be removed by deterministic keys while preserving first-seen lesson order.

### Concept planning

Concept planning will be separated from card-cluster generation. It will receive the bounded, slide-balanced fact set and will be responsible for:

- building the concept-plan prompt;
- parsing the strict JSON contract;
- checking unique concept names, known fact IDs, and assessment approaches;
- retrying invalid model output within the configured bound; and
- returning either a valid 20-concept plan or one concise failure.

The model contract remains strict: every concept must explicitly cite a non-empty JSON array of known fact IDs. The application will not invent grounding for a malformed plan, because doing so could attach unsupported facts to generated cards.

Retry feedback will group repeated errors by type and show representative concept numbers. For example, twenty identical missing-array errors become one message identifying all affected concepts. Full rejected model output will not be printed during normal progress reporting. A bounded diagnostic excerpt may be retained for debug logging, with no duplicate copy in the API error.

If the model repeatedly returns malformed plans, the final message will state the number of attempts, the grouped contract violations, and the likely stale-artifact indicator when the graph has no current rich facts. This keeps the error useful without hiding the underlying validation rule.

### Orchestration modules

`pipeline.py`, `batch_pipeline.py`, and `flashcard_pipeline.py` will remain the workflow entry points but become thinner:

- orchestration decides what stage runs and in which order;
- artifact contracts decide whether an output is reusable;
- persistence helpers perform atomic reads/writes;
- domain modules parse and transform content; and
- presentation layers format progress or API errors.

Stage dependency behavior remains:

```text
normalized text -> knowledge graph -> flashcards
```

When a stage is stale, that stage and all downstream outputs are invalidated. Uploaded source PDFs and independent course outputs are never removed by cleanup logic.

## Redundancy policy

Cleanup will target demonstrable duplication rather than reducing line count as an end in itself. A helper should be centralized when it implements the same contract in multiple modules, such as atomic JSON writes, JSON-safe conversion, normalized deduplication keys, or artifact-version checks.

Similar-looking code will remain separate when it enforces different domain rules. For example, card validation and graph validation should not share a vague generic validator, and boundary-level `except Exception` handlers may remain where they deliberately translate unexpected failures into API or launcher errors.

Dead compatibility files or aliases will be removed only after repository-wide reference checks and tests prove they are unused. Generated data, model bundles, and user artifacts are outside source-cleanup scope and remain protected by `.gitignore`.

## Error handling

- Domain parsing raises typed validation errors with structured error lists.
- File I/O errors include the operation and exact relevant path.
- Pipeline orchestration translates stage errors once; nested layers do not repeatedly prepend the same context.
- API boundaries return concise user-facing messages and keep tracebacks in server diagnostics.
- Portable startup errors retain their existing exit-code behavior.
- No error handler silently substitutes ungrounded concepts or flashcards.

## Testing strategy

Implementation will begin with characterization and regression tests before production changes.

Required tests include:

- current normalized text and rich graph artifacts are reusable;
- version-1 normalized text is readable but stale for pipeline reuse;
- a legacy edge-only graph is readable but stale for pipeline reuse;
- stale normalized text invalidates graph, flashcards, and the reuse manifest;
- stale graph invalidates flashcards and the reuse manifest;
- normalized lesson facts preserve slide/topic coverage and deterministic IDs;
- graph duplicate removal is deterministic and does not discard distinct provenance;
- `Path` values serialize safely through the shared JSON boundary;
- concept-plan errors are grouped without weakening fact-ID validation;
- malformed plans still stop after the configured retry count;
- CLI, API, batch resume, and portable path tests retain their existing behavior.

After focused tests pass, the complete repository test suite must pass. The portable directory will then be rebuilt, `--verify` will pass, the health endpoint will return HTTP 200, and the release ZIP checksum will be regenerated because any source change alters the executable bundle.

## Implementation sequence

1. Add characterization tests for legacy/current artifact readability and reuse.
2. Introduce schema constants and artifact-contract validation.
3. Add shared atomic JSON serialization and migrate one persistence boundary at a time.
4. Extract normalized lesson-fact logic behind compatibility wrappers.
5. Separate graph assembly/deduplication from serialization.
6. Extract concept planning and grouped retry diagnostics.
7. Remove only duplication made obsolete by the new focused modules.
8. Run focused and full tests, inspect the final diff, and rebuild the portable release.

Each step should leave the repository passing tests. Changes that alter a public interface or unrelated code will be excluded from this cleanup.

## Completion criteria

The cleanup is complete when:

- all existing public commands, API routes, and portable startup behavior remain compatible;
- stale version-1 normalized files and legacy sparse graphs are regenerated automatically during pipeline resume;
- newly produced graphs contain versioned, non-empty normalized lesson facts with available slide/topic provenance;
- concept-plan errors are concise and no longer repeat the same violation twenty times;
- duplicated artifact I/O and validation rules have one documented owner;
- the full automated test suite passes;
- the rebuilt portable executable passes bundle verification and its health check; and
- a new transfer ZIP and matching SHA-256 checksum are produced.
