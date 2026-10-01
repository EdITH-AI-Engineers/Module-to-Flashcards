# Split Flashcard CSVs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish each validated 100-card module as two distinct standard CSV files of 50 cards, named `<course>_M<module>-1.csv` and `<course>_M<module>-2.csv`.

**Architecture:** Keep the existing base path as the legacy artifact identity and derive the two canonical paths from it. Render both parts only after full-module validation, publish them with a hash receipt written last, and require the complete validated pair for reuse; batch, CLI, and API return both paths. A valid unchanged legacy combined file can be converted before loading Qwen.

**Tech Stack:** Python standard-library CSV, pathlib, hashlib, JSON, pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-01-gpu-auto-workers-and-split-flashcard-csv-design.md` (two-file-output section).

## Global Constraints

- Exact output names: `<course>_M<module>-1.csv` and `<course>_M<module>-2.csv`.
- Every file has the existing 13-column header and exactly 50 rows (10 complete clusters); no `Module N.1` separator line.
- All 100 cards still pass existing module-level validation, including cross-half duplicate checks, before any file is published.
- A missing/mismatched part or interrupted pair replacement is never reusable as a completed module.
- The API `outputs` list includes both paths in part order; the existing singular status `output` remains part 1 and a new status `outputs` list exposes both.
- Keep valid pre-change combined CSVs as legacy backups; do not silently delete them.

## Review Focus

1. `test_pair_receipt_rejects_mixed_generations`: replacing only part 1 while old part 2 remains must not appear complete.
2. `test_pair_parser_rejects_cross_part_question_duplicate`: a near-duplicate across halves must be rejected.
3. `test_custom_output_base_generates_two_expected_names`: `--output custom.csv` must produce `custom-1.csv` and `custom-2.csv`, not `custom.csv`.
4. `test_legacy_migration_rejects_changed_source`: old CSV for another source revision must never be reused.
5. `test_api_reports_both_parts_once_per_module`: one completed module has two output paths, with part 1 retained in legacy `output`.

---

### Task 1: Canonical pair rendering, validation, and publication

**Files:**
- Modify: `artifact_paths.py`, `flashcard_csv.py`
- Test: `tests/test_flashcard_csv.py`, `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `flashcard_part_paths(base: Path) -> tuple[Path, Path]` and `flashcard_receipt_path(base: Path) -> Path` in `artifact_paths.py`; `base` is the existing unsuffixed `.csv` path.
- Produces: `render_module_parts(identity: ModuleIdentity, clusters: Sequence[FlashcardCluster]) -> tuple[str, str]` and `parse_rendered_parts(parts: tuple[str, str], identity: ModuleIdentity, *, validate_course_code: bool = True) -> tuple[tuple[tuple[str, ...], ...], ...]` in `flashcard_csv.py`.
- Produces: `write_module_parts(base: Path, parts: tuple[str, str], identity: ModuleIdentity) -> tuple[Path, Path]` and `valid_written_parts(base: Path, identity: ModuleIdentity, *, validate_course_code: bool = True) -> bool`.
- Receipt: `base.with_suffix(".parts.json")` stores each part's SHA-256 digest and the course/module identity. Write staged UTF-8-BOM CSV files, replace both destinations, then atomically replace the receipt last. A stale receipt hash makes the pair invalid.

- [ ] **Step 1: Write failing tests.** Assert exact `-1.csv`/`-2.csv` paths, header plus 50 rows in each, ten disjoint cluster UUIDs per part, and rejection of cross-part near-duplicate questions; assert Review Focus tests 1–3 and preservation of valid `expalanation` spelling.
- [ ] **Step 2: Run red tests.** ` .\.venv\Scripts\python.exe -m pytest -q tests/test_flashcard_csv.py tests/test_pipeline_runner.py -p no:cacheprovider --basetemp pipeline_temporary/pytest-csv-core-red` must fail only for the new behavior.
- [ ] **Step 3: Implement the four CSV interfaces and two path helpers.** Retain `render_module`/`parse_rendered_module` for legacy migration, but do not call `render_module` for new output.
- [ ] **Step 4: Run green tests and full suite.** After both pass, commit only the core pair files and tests.

### Task 2: Use the pair throughout the live generation path

**Files:**
- Modify: `main.py`, `pipeline.py`, `batch_pipeline.py`, `api_server.py`, `README.md`
- Test: `tests/test_main.py`, `tests/test_pipeline_runner.py`, `tests/test_batch_pipeline.py`, `tests/test_api_server.py`, `tests/batch_helpers.py`

**Interfaces:**
- Consumes: Task 1 path, writer, and validator helpers.
- Produces: `main.run(...) -> tuple[Path, Path] | None`; `PipelinePaths.flashcards` remains the unsuffixed legacy/base path, with `flashcard_parts` and `flashcard_receipt` read-only properties. `BatchResult.outputs` remains a flat tuple of paths, ordered part 1 then part 2 for each successful module.
- The subprocess still receives `--output <base>.csv`; `StageCommand.expected_output` points to part 1, while stage validation checks both parts plus receipt. API status adds `outputs: list[str]` and preserves `output` as part 1.

- [ ] **Step 1: Write failing integration tests.** Assert custom output names and return tuple, CLI prints both paths, pipeline reuse needs both files, batch returns two paths per module, API `/process` and `/status` expose both, and course-corpus updates only after a complete pair is written. Include Review Focus tests 3 and 5.
- [ ] **Step 2: Run red tests.** Run the four named test files; confirm failures reflect the old single-file behavior.
- [ ] **Step 3: Wire every producer/consumer to the pair.** Invalidation removes both parts and receipt; duplicate-upload detection remains per module, not per output path; never write a new combined CSV.
- [ ] **Step 4: Run green tests, full suite, and `git diff --check`.** Commit this integration task before starting migration.

### Task 3: Safe conversion of unchanged legacy combined files

**Files:**
- Modify: `flashcard_csv.py`, `pipeline.py`, `batch_pipeline.py`
- Test: `tests/test_flashcard_csv.py`, `tests/test_pipeline_runner.py`, `tests/test_batch_pipeline.py`

**Interfaces:**
- Produces: `migrate_legacy_module(base: Path, identity: ModuleIdentity) -> tuple[Path, Path] | None`, which parses the existing two labeled blocks, renders two standard CSVs, writes a receipt, and retains the original base file.
- Call migration only after proving the source matches the source used for the legacy artifact: batch compares its reuse-manifest source digest and non-worker generation settings; subprocess runner compares the incoming source bytes with the previously staged source before overwriting it. If provenance cannot be established, regenerate instead of migrating.

- [ ] **Step 1: Write failing migration tests.** Assert valid unchanged combined output converts without Qwen, original file remains, corrupt legacy input regenerates, and changed source cannot migrate (Review Focus 4). Assert an interrupted conversion leaves no reusable pair (Review Focus 1).
- [ ] **Step 2: Run red tests.** Run the three named test files; confirm feature-related failures.
- [ ] **Step 3: Implement conversion before resource loading/invalidation.** Conversion must validate both new files and receipt before it is reported complete; do not update the course corpus during migration.
- [ ] **Step 4: Run green tests, full suite, and `git diff --check`.** Commit only migration files/tests.

## Verification handoff

Run one normal module through CLI and one through the API, inspect both CSVs with a standard CSV reader, and verify that a rerun reuses the pair. Do not report completion based only on isolated rendering tests.
