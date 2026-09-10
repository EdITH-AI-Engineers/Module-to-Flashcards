# Python Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the Python pipeline without breaking its CLI, API, portable workflow, or 20-concept/100-card contract, while preventing stale low-context artifacts from causing concept-plan failures.

**Architecture:** Keep the existing entry-point modules as compatibility facades while moving atomic JSON I/O, artifact contracts, lesson-fact extraction, graph assembly, and concept planning into focused modules. Pipeline resume uses strict current-schema checks, while direct readers remain able to load legacy edge-based graphs.

**Tech Stack:** Python 3.11+, standard library, pytest, FastAPI, PyInstaller, PowerShell

**Spec:** `docs/superpowers/specs/2026-09-10-python-codebase-cleanup-design.md`

## Global Constraints

- Preserve existing CLI entry points, flags, FastAPI routes, output filenames, and directory layout.
- Preserve the portable double-click workflow and adjacent `data` directory.
- Preserve exactly 20 concepts, 5 cards per concept, and 100 cards per module.
- Existing graph files with usable legacy edges remain directly readable but are not reusable by the current staged pipeline.
- Do not invent fact IDs or silently attach unsupported grounding to malformed model output.
- Do not replace JSON, CSV, or structured text with a database.
- Do not add a network dependency or runtime migration prompt.
- Write failing tests before each production change and keep every task independently passing.
- Keep generated bundles, models, pipeline data, and build outputs ignored by Git.

## File Structure

- Create `artifact_io.py`: recursive JSON-safe conversion and atomic JSON writes.
- Create `artifact_contracts.py`: current normalized/graph schema constants and reusable-artifact validators.
- Create `lesson_facts.py`: normalized fact extraction, noise filtering, deterministic IDs, and provenance merging.
- Create `graph_builder.py`: triple/fact deduplication and knowledge-graph assembly.
- Create `generation_retry.py`: bounded model completion retry loop and shared generation error.
- Create `concept_planning.py`: balanced fact selection, concept error grouping, and plan generation.
- Modify `structured_module.py`: render the current schema and retain compatibility wrappers.
- Modify `text_extractor.py`: orchestrate extraction and delegate graph assembly/JSON persistence.
- Modify `pipeline.py`: delegate reuse decisions to artifact contracts.
- Modify `flashcard_pipeline.py`: delegate retry and concept planning while retaining public imports.
- Modify `main.py` and `batch_pipeline.py`: use shared atomic JSON persistence.
- Modify `README.md`: document automatic regeneration of stale pipeline artifacts.
- Add focused tests under `tests/` and update existing compatibility assertions.

---

### Task 1: Shared Atomic JSON I/O

**Files:**
- Create: `artifact_io.py`
- Create: `tests/test_artifact_io.py`
- Modify: `text_extractor.py:1-12,250-258`
- Test: `tests/test_text_extractor_structured.py`

**Interfaces:**
- Consumes: standard-library `json`, `tempfile`, `Path`, and `Mapping`.
- Produces: `to_json_value(value: object) -> object` and `write_json_atomic(path: Path, value: object, *, ensure_ascii: bool = False, indent: int | None = None, sort_keys: bool = False) -> None`.

- [ ] **Step 1: Write failing JSON-safety and atomic-write tests**

```python
import json
from pathlib import Path

from artifact_io import to_json_value, write_json_atomic


def test_to_json_value_converts_nested_paths_and_tuples(tmp_path):
    value = {
        "model": tmp_path / "models" / "rebel-large",
        "slides": (1, 2),
        "nested": [{"output": Path("graph.json")}],
    }

    assert to_json_value(value) == {
        "model": str(tmp_path / "models" / "rebel-large"),
        "slides": [1, 2],
        "nested": [{"output": "graph.json"}],
    }


def test_write_json_atomic_replaces_file_and_leaves_no_temporary_file(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text('{"old": true}', encoding="utf-8")

    write_json_atomic(destination, {"path": Path("new.txt")}, indent=2)

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "path": "new.txt"
    }
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert list(tmp_path.glob(".state.json.*.tmp")) == []
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `python -m pytest tests/test_artifact_io.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'artifact_io'`.

- [ ] **Step 3: Implement JSON-safe conversion and atomic writing**

```python
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import tempfile


def to_json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def write_json_atomic(
    path: Path,
    value: object,
    *,
    ensure_ascii: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                to_json_value(value),
                handle,
                ensure_ascii=ensure_ascii,
                indent=indent,
                sort_keys=sort_keys,
            )
            handle.write("\n")
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
```

In `text_extractor.save_outputs`, replace the direct `json.dump` block with:

```python
write_json_atomic(json_path, graph, ensure_ascii=False, indent=2)
```

Import `write_json_atomic` from `artifact_io` and remove the unused `json` import from `text_extractor.py`.

- [ ] **Step 4: Run focused persistence tests**

Run: `python -m pytest tests/test_artifact_io.py tests/test_text_extractor_structured.py -v`

Expected: PASS, including serialization of a `WindowsPath`-compatible `Path` model value.

- [ ] **Step 5: Commit the atomic I/O boundary**

```powershell
git add artifact_io.py text_extractor.py tests/test_artifact_io.py tests/test_text_extractor_structured.py
git commit -m "refactor: centralize atomic JSON output"
```

---

### Task 2: Versioned Artifact Contracts

**Files:**
- Create: `artifact_contracts.py`
- Create: `tests/test_artifact_contracts.py`
- Modify: `structured_module.py:1-10,129-138`
- Modify: `text_extractor.py:190-249`
- Modify: `tests/test_structured_module.py:24-43`
- Modify: `tests/test_text_extractor_structured.py:68-177`

**Interfaces:**
- Consumes: `parse_module_metadata`, `graph_ready_text`, `load_graph`, `extract_graph_facts`, and `parse_rendered_module` through function-local imports that avoid import cycles.
- Produces: `STRUCTURED_FORMAT_VERSION = "2"`, `KNOWLEDGE_GRAPH_SCHEMA_VERSION = 2`, `structured_text_is_current(text: str) -> bool`, `graph_is_current(graph: Mapping[str, object]) -> bool`, `valid_structured_artifact(path: Path) -> bool`, `valid_graph_artifact(path: Path) -> bool`, and `valid_flashcard_artifact(path: Path, module_number: str, course_code: str | None = None) -> bool`.

- [ ] **Step 1: Write unit tests for readable legacy artifacts and strict current reuse**

```python
import json

from artifact_contracts import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    STRUCTURED_FORMAT_VERSION,
    graph_is_current,
    structured_text_is_current,
    valid_graph_artifact,
)
from graph_input import extract_graph_facts, load_graph
from structured_module import StructuredModule, StructuredSlide, render_structured_module


def current_structured_text():
    return render_structured_module(
        StructuredModule(
            course_code="CPE0021",
            module_number="01",
            module_title="Architecture",
            source_file="module.pdf",
            slides=(
                StructuredSlide(
                    number=1,
                    extraction_method="text",
                    title="Processor",
                    content=("A processor executes instructions.",),
                    visual_text=("Not Specified",),
                    definitions=(),
                    knowledge_statements=("A processor contains an ALU.",),
                    brief_explanation="The slide describes a processor.",
                ),
            ),
        )
    )


def current_graph():
    return {
        "metadata": {
            "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
            "fact_count": 1,
        },
        "nodes": [],
        "edges": [],
        "facts": [
            {
                "id": "f1",
                "statement": "A processor contains an ALU.",
                "slides": [1],
                "topic": "Processor",
            }
        ],
    }


def test_current_structured_text_is_reusable_and_version_one_is_stale():
    current = current_structured_text()
    legacy = current.replace(
        f"format_version: {STRUCTURED_FORMAT_VERSION}",
        "format_version: 1",
        1,
    )

    assert structured_text_is_current(current)
    assert not structured_text_is_current(legacy)


def test_current_rich_graph_is_reusable():
    assert graph_is_current(current_graph())


def test_legacy_edge_graph_is_readable_but_not_reusable(tmp_path):
    legacy = {
        "metadata": {},
        "nodes": [],
        "edges": [
            {
                "id": "e1",
                "subject": "processor",
                "relation": "contains",
                "object": "ALU",
            }
        ],
    }
    path = tmp_path / "knowledge_graph.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    assert extract_graph_facts(load_graph(path))[0].fact_id == "e1"
    assert not valid_graph_artifact(path)
```

- [ ] **Step 2: Run the contract tests and verify the missing module failure**

Run: `python -m pytest tests/test_artifact_contracts.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'artifact_contracts'`.

- [ ] **Step 3: Implement current-schema checks**

Create `artifact_contracts.py` with these rules:

```python
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


STRUCTURED_FORMAT_VERSION = "2"
KNOWLEDGE_GRAPH_SCHEMA_VERSION = 2


def structured_text_is_current(text: str) -> bool:
    from structured_module import graph_ready_text, parse_module_metadata

    metadata = parse_module_metadata(text)
    return (
        metadata.get("format_version") == STRUCTURED_FORMAT_VERSION
        and bool(graph_ready_text(text).strip())
    )


def graph_is_current(graph: Mapping[str, object]) -> bool:
    from graph_input import GraphInputError, extract_graph_facts

    metadata = graph.get("metadata")
    facts = graph.get("facts")
    if not isinstance(metadata, Mapping) or not isinstance(facts, list) or not facts:
        return False
    if metadata.get("schema_version") != KNOWLEDGE_GRAPH_SCHEMA_VERSION:
        return False
    if metadata.get("fact_count") != len(facts):
        return False
    seen_ids: set[str] = set()
    seen_statements: set[str] = set()
    for fact in facts:
        if not isinstance(fact, Mapping):
            return False
        if not isinstance(fact.get("id"), str) or not str(fact["id"]).strip():
            return False
        if not isinstance(fact.get("statement"), str) or not str(fact["statement"]).strip():
            return False
        fact_id = str(fact["id"]).strip()
        statement = " ".join(str(fact["statement"]).casefold().split())
        if fact_id in seen_ids or statement in seen_statements:
            return False
        seen_ids.add(fact_id)
        seen_statements.add(statement)
    try:
        return len(extract_graph_facts(graph)) == len(facts)
    except (GraphInputError, TypeError, ValueError):
        return False


def valid_structured_artifact(path: Path) -> bool:
    try:
        return structured_text_is_current(Path(path).read_text(encoding="utf-8-sig"))
    except OSError:
        return False


def valid_graph_artifact(path: Path) -> bool:
    from graph_input import GraphInputError, load_graph

    try:
        return graph_is_current(load_graph(Path(path)))
    except (GraphInputError, OSError, TypeError, ValueError):
        return False


def valid_flashcard_artifact(
    path: Path,
    module_number: str,
    course_code: str | None = None,
) -> bool:
    from flashcard_csv import parse_rendered_module
    from flashcard_types import ModuleIdentity

    try:
        text = Path(path).read_text(encoding="utf-8-sig")
        parse_rendered_module(
            text,
            ModuleIdentity(str(course_code or ""), str(module_number)),
            validate_course_code=course_code is not None,
        )
    except (OSError, ValueError):
        return False
    return True
```

Change `structured_module.py` to import `STRUCTURED_FORMAT_VERSION` and retain its existing public constant:

```python
from artifact_contracts import STRUCTURED_FORMAT_VERSION

FORMAT_VERSION = STRUCTURED_FORMAT_VERSION
```

Add the graph schema field before module metadata is merged in `text_extractor.build_graph`:

```python
metadata = {
    "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    "source_file": source_file.name,
    "model": str(model_name),
    "node_count": len(nodes),
    "edge_count": len(edges),
    "fact_count": len(lesson_facts),
}
```

Update current-version assertions from `"1"` to `"2"` and assert the graph metadata has schema version `2`.

- [ ] **Step 4: Run focused schema tests**

Run: `python -m pytest tests/test_artifact_contracts.py tests/test_structured_module.py tests/test_text_extractor_structured.py tests/test_graph_input.py -v`

Expected: PASS. Legacy graph loading remains green while current reuse requires rich facts.

- [ ] **Step 5: Commit the artifact contracts**

```powershell
git add artifact_contracts.py structured_module.py text_extractor.py tests/test_artifact_contracts.py tests/test_structured_module.py tests/test_text_extractor_structured.py
git commit -m "fix: version normalized graph artifacts"
```

---

### Task 3: Schema-Aware Pipeline Resume and Invalidation

**Files:**
- Modify: `pipeline.py:1-15,208-247`
- Modify: `tests/test_pipeline_runner.py:20-75,220-320`
- Test: `tests/test_batch_pipeline.py`

**Interfaces:**
- Consumes: all three `valid_*_artifact` functions from `artifact_contracts.py`.
- Produces: compatibility functions `_valid_structured_text`, `_valid_graph`, and `_valid_flashcards` at their existing `pipeline` import locations.

- [ ] **Step 1: Make test artifacts current and add stale-resume regression tests**

Update `graph_content()` in `tests/test_pipeline_runner.py` so routine reuse tests write a current graph:

```python
def graph_content():
    return json.dumps(
        {
            "metadata": {
                "schema_version": 2,
                "course_code": "CPE0021",
                "module_number": "01",
                "fact_count": 1,
            },
            "nodes": [],
            "edges": [],
            "facts": [
                {
                    "id": "f1",
                    "statement": "A processor contains an ALU.",
                    "slides": [1],
                    "topic": "Processor",
                }
            ],
        }
    )
```

Add two regression tests:

```python
def test_version_one_normalized_artifact_recomputes_all_stages(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    for stage in ("pdf-to-text", "knowledge-graph", "flashcards"):
        materialize(stage, paths)
    paths.structured_text.write_text(
        structured_content().replace("format_version: 2", "format_version: 1", 1),
        encoding="utf-8",
    )
    calls = []

    def runner(command, **kwargs):
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert calls == ["pdf-to-text", "knowledge-graph", "flashcards"]


def test_legacy_edge_only_graph_recomputes_graph_and_flashcards(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    for stage in ("pdf-to-text", "knowledge-graph", "flashcards"):
        materialize(stage, paths)
    legacy_graph = {
        "metadata": {"course_code": "CPE0021", "module_number": "01"},
        "nodes": [],
        "edges": [
            {
                "id": "e1",
                "subject": "processor",
                "relation": "contains",
                "object": "ALU",
            }
        ],
    }
    paths.graph_json.write_text(json.dumps(legacy_graph), encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        calls.append(stage)
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert calls == ["knowledge-graph", "flashcards"]
```

- [ ] **Step 2: Run both regressions and verify they fail under permissive reuse**

Run: `python -m pytest tests/test_pipeline_runner.py -k "version_one or legacy_edge_only" -v`

Expected: FAIL because `pipeline.py` still accepts legacy normalized and edge-only graph artifacts.

- [ ] **Step 3: Delegate existing compatibility functions to artifact contracts**

Replace the bodies in `pipeline.py` with:

```python
def _valid_structured_text(path: Path) -> bool:
    return valid_structured_artifact(path)


def _valid_graph(path: Path) -> bool:
    return valid_graph_artifact(path)


def _valid_flashcards(
    path: Path,
    module_number: str,
    course_code: str | None = None,
) -> bool:
    return valid_flashcard_artifact(path, module_number, course_code)
```

Import these delegates from `artifact_contracts` and remove now-unused parsing imports from `pipeline.py`. Keep the compatibility function names because `batch_pipeline.py` and tests import them.

- [ ] **Step 4: Run pipeline and batch resume tests**

Run: `python -m pytest tests/test_pipeline_runner.py tests/test_batch_pipeline.py -v`

Expected: PASS. A stale upstream artifact causes its stage and all downstream stages to rerun.

- [ ] **Step 5: Commit schema-aware resume behavior**

```powershell
git add pipeline.py tests/test_pipeline_runner.py
git commit -m "fix: rebuild stale pipeline artifacts"
```

---

### Task 4: Focused Lesson-Fact Extraction

**Files:**
- Create: `lesson_facts.py`
- Create: `tests/test_lesson_facts.py`
- Modify: `structured_module.py:13-23,193-332`
- Modify: `tests/test_structured_module.py`

**Interfaces:**
- Consumes: normalized structured text plus an optional module title supplied by the compatibility wrapper.
- Produces: `extract_normalized_lesson_facts(text: str, *, module_title: str = "") -> tuple[dict[str, object], ...]`; `structured_module.extract_lesson_facts(text: str)` remains public.

- [ ] **Step 1: Write direct tests for deterministic deduplication and merged provenance**

```python
from lesson_facts import extract_normalized_lesson_facts


def test_duplicate_fact_merges_slide_provenance_in_first_seen_order():
    text = """[SLIDE 1]
[TITLE]
Processor
[/TITLE]
[KNOWLEDGE_STATEMENTS]
- A processor executes instructions in a defined cycle.
[/KNOWLEDGE_STATEMENTS]
[/SLIDE]

[SLIDE 2]
[TITLE]
Instruction Cycle
[/TITLE]
[KNOWLEDGE_STATEMENTS]
- A processor executes instructions in a defined cycle.
[/KNOWLEDGE_STATEMENTS]
[/SLIDE]
"""

    facts = extract_normalized_lesson_facts(text, module_title="Architecture")

    assert facts == (
        {
            "id": "f1",
            "statement": "A processor executes instructions in a defined cycle.",
            "slides": [1, 2],
            "kind": "knowledge_statement",
            "topic": "Processor",
        },
    )


def test_definition_precedes_statement_and_noise_is_excluded():
    text = """[SLIDE 3]
[TITLE]
Instruction Cycle
[/TITLE]
[CONTENT]
- Fetch, decode, and execute are processor stages.
[/CONTENT]
[DEFINITIONS]
- Fetch :: Retrieve an instruction from memory.
[/DEFINITIONS]
[KNOWLEDGE_STATEMENTS]
- The slide introduces the instruction cycle.
- The processor fetches an instruction before decoding it.
[/KNOWLEDGE_STATEMENTS]
[/SLIDE]
"""

    facts = extract_normalized_lesson_facts(text)

    assert [fact["kind"] for fact in facts] == [
        "definition",
        "knowledge_statement",
    ]
    assert all("slide introduces" not in str(fact["statement"]) for fact in facts)
```

- [ ] **Step 2: Run the direct tests and verify the missing module failure**

Run: `python -m pytest tests/test_lesson_facts.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'lesson_facts'`.

- [ ] **Step 3: Move extraction into the focused module and merge duplicate slides**

Implement `lesson_facts.py` with these internal boundaries:

```python
from __future__ import annotations

import re


_SECTION_TAG = re.compile(r"^\[(/?)([A-Z_]+)(?:\s+\d+)?\]$")
_SLIDE_OPEN_TAG = re.compile(r"^\[SLIDE\s+(\d+)\]$", flags=re.IGNORECASE)
_PRESENTATION_NOISE = re.compile(
    r"^(?:(?:this|the|the first|the current|first)\s+(?:slide|page)\b|"
    r"the module title\b)",
    flags=re.IGNORECASE,
)
_EMPTY_VALUES = {"", "not specified", "[unreadable text]"}
_INCLUDED_SECTIONS = {"TITLE", "CONTENT", "DEFINITIONS", "KNOWLEDGE_STATEMENTS"}


def _meaningful(value: str) -> bool:
    return value.strip().casefold() not in _EMPTY_VALUES


def _fact_text(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value).strip()
    if text.startswith("- "):
        text = text[2:].strip()
    if not _meaningful(text) or _PRESENTATION_NOISE.match(text):
        return None
    if len(text) < 12 or len(text.split()) < 3:
        return None
    return text


def _deduplication_key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).strip().casefold()


def _read_slides(text: str) -> list[dict[str, object]]:
    slides: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    section: str | None = None
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        slide_tag = _SLIDE_OPEN_TAG.fullmatch(line)
        if slide_tag:
            current = {
                "number": int(slide_tag.group(1)),
                "TITLE": [],
                "CONTENT": [],
                "DEFINITIONS": [],
                "KNOWLEDGE_STATEMENTS": [],
            }
            slides.append(current)
            section = None
            continue
        if line.casefold() == "[/slide]":
            current = None
            section = None
            continue
        tag = _SECTION_TAG.fullmatch(line)
        if tag:
            closing, name = tag.groups()
            section = None if closing else name
            continue
        if current is None or section not in _INCLUDED_SECTIONS or not line:
            continue
        values = current[section]
        if isinstance(values, list):
            values.append(line)
    return slides


def extract_normalized_lesson_facts(
    text: str,
    *,
    module_title: str = "",
) -> tuple[dict[str, object], ...]:
    facts: list[dict[str, object]] = []
    facts_by_key: dict[str, dict[str, object]] = {}

    def add_fact(statement: str, *, slide_number: int, kind: str, topic: str) -> None:
        cleaned = _fact_text(statement)
        if cleaned is None:
            return
        key = _deduplication_key(cleaned)
        existing = facts_by_key.get(key)
        if existing is not None:
            slides = existing["slides"]
            if isinstance(slides, list) and slide_number not in slides:
                slides.append(slide_number)
            return
        fact = {
            "id": f"f{len(facts) + 1}",
            "statement": cleaned,
            "slides": [slide_number],
            "kind": kind,
            "topic": topic,
        }
        facts.append(fact)
        facts_by_key[key] = fact

    for slide in _read_slides(text):
        slide_number = int(slide["number"])
        title_values = slide["TITLE"]
        title = ""
        if isinstance(title_values, list):
            title = next(
                (
                    cleaned
                    for value in title_values
                    if _meaningful(
                        cleaned := re.sub(r"\s+", " ", str(value)).strip()
                    )
                ),
                "",
            )
        topic = title or (module_title if _meaningful(module_title) else f"Slide {slide_number}")
        before = len(facts)
        definitions = slide["DEFINITIONS"]
        if isinstance(definitions, list):
            for raw_definition in definitions:
                definition = str(raw_definition)
                if definition.startswith("- "):
                    definition = definition[2:].strip()
                if " :: " not in definition:
                    continue
                term, meaning = definition.split(" :: ", 1)
                add_fact(
                    f"{term.strip()}: {meaning.strip()}",
                    slide_number=slide_number,
                    kind="definition",
                    topic=topic,
                )
        statements = slide["KNOWLEDGE_STATEMENTS"]
        if isinstance(statements, list):
            for statement in statements:
                add_fact(
                    str(statement),
                    slide_number=slide_number,
                    kind="knowledge_statement",
                    topic=topic,
                )
        if len(facts) == before:
            content = slide["CONTENT"]
            if isinstance(content, list):
                for statement in content:
                    add_fact(
                        str(statement),
                        slide_number=slide_number,
                        kind="content",
                        topic=topic,
                    )
    return tuple(facts)
```

Retain the public wrapper in `structured_module.py`:

```python
def extract_lesson_facts(text: str) -> tuple[dict[str, object], ...]:
    metadata = parse_module_metadata(text)
    if not metadata:
        return ()
    return extract_normalized_lesson_facts(
        text,
        module_title=metadata.get("module_title", ""),
    )
```

Remove moved extraction-only regular expressions and helpers from `structured_module.py`; retain `_meaningful` because the structured dataclasses use it.

- [ ] **Step 4: Run lesson and structured-module tests**

Run: `python -m pytest tests/test_lesson_facts.py tests/test_structured_module.py tests/test_slide_normalizer.py tests/test_text_extractor_structured.py -v`

Expected: PASS with stable IDs and merged slide provenance.

- [ ] **Step 5: Commit the lesson-fact boundary**

```powershell
git add lesson_facts.py structured_module.py tests/test_lesson_facts.py tests/test_structured_module.py
git commit -m "refactor: isolate normalized lesson facts"
```

---

### Task 5: Deterministic Knowledge-Graph Assembly

**Files:**
- Create: `graph_builder.py`
- Create: `tests/test_graph_builder.py`
- Modify: `text_extractor.py:100-118,190-249`
- Test: `tests/test_text_extractor_structured.py`

**Interfaces:**
- Consumes: extracted triple dictionaries, normalized lesson-fact dictionaries, source path, model identifier, and structured metadata.
- Produces: `canonical(value: str) -> str` and `build_graph(triples: Sequence[Mapping[str, object]], source_file: Path, model_name: str | Path, module_metadata: Mapping[str, str] | None = None, lesson_facts: Sequence[Mapping[str, object]] = ()) -> dict[str, object]`.

- [ ] **Step 1: Write graph deduplication and provenance tests**

```python
from pathlib import Path

from graph_builder import build_graph


def test_build_graph_merges_duplicate_edges_and_evidence():
    triples = [
        {
            "subject": "Processor",
            "relation": "contains",
            "object": "ALU",
            "evidence": [{"chunk_id": 1, "slides": [1], "text": "Processor contains ALU."}],
        },
        {
            "subject": " processor ",
            "relation": "CONTAINS",
            "object": "alu",
            "evidence": [
                {"chunk_id": 1, "slides": [1], "text": "Processor contains ALU."},
                {"chunk_id": 2, "slides": [2], "text": "The ALU is in the processor."},
            ],
        },
    ]

    graph = build_graph(triples, Path("module.txt"), "rebel")

    assert len(graph["edges"]) == 1
    assert len(graph["edges"][0]["evidence"]) == 2
    assert graph["metadata"]["edge_count"] == 1


def test_build_graph_merges_duplicate_fact_slides_without_losing_order():
    facts = [
        {
            "id": "f1",
            "statement": "A processor executes instructions.",
            "slides": [1],
            "kind": "knowledge_statement",
            "topic": "Processor",
        },
        {
            "id": "f9",
            "statement": " A PROCESSOR executes instructions. ",
            "slides": [3],
            "kind": "knowledge_statement",
            "topic": "Execution",
        },
    ]

    graph = build_graph([], Path("module.txt"), "rebel", lesson_facts=facts)

    assert graph["facts"] == [
        {
            "id": "f1",
            "statement": "A processor executes instructions.",
            "slides": [1, 3],
            "kind": "knowledge_statement",
            "topic": "Processor",
        }
    ]
    assert graph["metadata"]["fact_count"] == 1
```

- [ ] **Step 2: Run graph-builder tests and verify the missing module failure**

Run: `python -m pytest tests/test_graph_builder.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'graph_builder'`.

- [ ] **Step 3: Implement deterministic merging and graph assembly**

Create `graph_builder.py` with:

```python
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import re

from artifact_contracts import KNOWLEDGE_GRAPH_SCHEMA_VERSION


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _merge_triples(triples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    evidence_keys: dict[tuple[str, str, str], set[tuple[object, tuple[object, ...], str]]] = {}
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        object_value = str(triple.get("object", "")).strip()
        if not subject or not relation or not object_value:
            continue
        key = (canonical(subject), canonical(relation), canonical(object_value))
        target = merged.setdefault(
            key,
            {
                "subject": subject,
                "relation": relation,
                "object": object_value,
                "evidence": [],
            },
        )
        seen = evidence_keys.setdefault(key, set())
        raw_evidence = triple.get("evidence")
        if not isinstance(raw_evidence, list):
            continue
        target_evidence = target["evidence"]
        if not isinstance(target_evidence, list):
            continue
        for item in raw_evidence:
            if not isinstance(item, Mapping):
                continue
            slides = item.get("slides")
            slide_values = tuple(slides) if isinstance(slides, list) else ()
            evidence_key = (
                item.get("chunk_id"),
                slide_values,
                canonical(str(item.get("text", ""))),
            )
            if evidence_key in seen:
                continue
            seen.add(evidence_key)
            target_evidence.append(dict(item))
    return list(merged.values())


def _merge_facts(facts: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    by_statement: dict[str, dict[str, object]] = {}
    used_ids: set[str] = set()
    for fact in facts:
        statement = re.sub(r"\s+", " ", str(fact.get("statement", ""))).strip()
        if not statement:
            continue
        key = canonical(statement)
        existing = by_statement.get(key)
        raw_slides = fact.get("slides")
        slides = []
        if isinstance(raw_slides, list):
            for value in raw_slides:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if number > 0 and number not in slides:
                    slides.append(number)
        if existing is not None:
            existing_slides = existing["slides"]
            if isinstance(existing_slides, list):
                existing_slides.extend(
                    number for number in slides if number not in existing_slides
                )
            continue
        item = dict(fact)
        fact_id = str(fact.get("id") or "").strip()
        if not fact_id or fact_id in used_ids:
            next_number = len(merged) + 1
            fact_id = f"f{next_number}"
            while fact_id in used_ids:
                next_number += 1
                fact_id = f"f{next_number}"
        item["id"] = fact_id
        item["statement"] = statement
        item["slides"] = slides
        merged.append(item)
        by_statement[key] = item
        used_ids.add(fact_id)
    return merged


def build_graph(
    triples: Sequence[Mapping[str, object]],
    source_file: Path,
    model_name: str | Path,
    module_metadata: Mapping[str, str] | None = None,
    lesson_facts: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    merged_triples = _merge_triples(triples)
    merged_facts = _merge_facts(lesson_facts)
    node_names: dict[str, str] = {}
    degrees: defaultdict[str, int] = defaultdict(int)
    for triple in merged_triples:
        for field in ("subject", "object"):
            name = str(triple[field])
            key = canonical(name)
            node_names.setdefault(key, name)
            degrees[key] += 1
    ordered_keys = sorted(
        node_names,
        key=lambda key: (-degrees[key], node_names[key].casefold()),
    )
    node_ids = {key: f"n{index}" for index, key in enumerate(ordered_keys, start=1)}
    nodes = [
        {"id": node_ids[key], "label": node_names[key], "degree": degrees[key]}
        for key in ordered_keys
    ]
    edges = [
        {
            "id": f"e{index}",
            "source": node_ids[canonical(str(triple["subject"]))],
            "target": node_ids[canonical(str(triple["object"]))],
            **triple,
        }
        for index, triple in enumerate(merged_triples, start=1)
    ]
    metadata: dict[str, object] = {
        "schema_version": KNOWLEDGE_GRAPH_SCHEMA_VERSION,
        "source_file": Path(source_file).name,
        "model": str(model_name),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "fact_count": len(merged_facts),
    }
    allowed_metadata = {
        "format_version",
        "course_code",
        "module_number",
        "module_title",
        "source_file",
    }
    metadata.update(
        {
            key: value
            for key, value in (module_metadata or {}).items()
            if key in allowed_metadata and str(value).strip()
        }
    )
    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
        "facts": merged_facts,
    }
```

Import `build_graph` and `canonical` into `text_extractor.py`, then remove the old definitions. This import preserves `text_extractor.build_graph` and `text_extractor.canonical` for existing users.

- [ ] **Step 4: Run graph extraction and input tests**

Run: `python -m pytest tests/test_graph_builder.py tests/test_text_extractor_structured.py tests/test_graph_input.py -v`

Expected: PASS with one edge/fact for normalized duplicates and merged provenance.

- [ ] **Step 5: Commit graph assembly cleanup**

```powershell
git add graph_builder.py text_extractor.py tests/test_graph_builder.py tests/test_text_extractor_structured.py
git commit -m "refactor: isolate knowledge graph assembly"
```

---

### Task 6: Shared Generation Retry Boundary

**Files:**
- Create: `generation_retry.py`
- Create: `tests/test_generation_retry.py`
- Modify: `flashcard_pipeline.py:1-40,111-153`

**Interfaces:**
- Consumes: `ChatBackend`, a parser callable, prompt builders, and typed validation errors.
- Produces: `GenerationError` and `complete_with_retries(backend, original_prompt, parser, *, max_retries, max_tokens, label, progress, include_rejected_candidate=True, report_candidate=True, error_formatter=None)`; `flashcard_pipeline.GenerationError` remains importable.

- [ ] **Step 1: Write direct retry tests**

```python
from collections import deque

import pytest

from flashcard_validator import ValidationError
from generation_retry import GenerationError, complete_with_retries


class Backend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return self.responses.popleft()


def parser(raw):
    if raw != "valid":
        raise ValidationError(("concept 1 missing fact_ids", "concept 2 missing fact_ids"))
    return raw


def test_retry_uses_formatted_errors_and_can_hide_candidate_output():
    backend = Backend(("secret invalid output", "valid"))
    messages = []

    result = complete_with_retries(
        backend,
        "original",
        parser,
        max_retries=2,
        max_tokens=100,
        label="concept plan",
        progress=messages.append,
        include_rejected_candidate=False,
        report_candidate=False,
        error_formatter=lambda errors: ("concepts 1-2 missing fact_ids",),
    )

    assert result == "valid"
    assert "concepts 1-2 missing fact_ids" in backend.calls[1][1]
    assert all("secret invalid output" not in message for message in messages)


def test_retry_raises_once_after_configured_attempts():
    backend = Backend(("invalid one", "invalid two"))

    with pytest.raises(
        GenerationError,
        match="concept plan failed after 2 attempts",
    ):
        complete_with_retries(
            backend,
            "original",
            parser,
            max_retries=2,
            max_tokens=100,
            label="concept plan",
            progress=lambda message: None,
            report_candidate=False,
        )

    assert len(backend.calls) == 2
```

- [ ] **Step 2: Run the retry tests and verify the missing module failure**

Run: `python -m pytest tests/test_generation_retry.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'generation_retry'`.

- [ ] **Step 3: Extract the existing bounded retry loop**

Create `generation_retry.py` with the existing `_complete_with_retries` behavior and these changes:

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from flashcard_prompt import SYSTEM_PROMPT, build_retry_prompt
from flashcard_types import ChatBackend
from flashcard_validator import InsufficientContentError, ValidationError


class GenerationError(RuntimeError):
    pass


Parsed = TypeVar("Parsed")
ErrorFormatter = Callable[[Sequence[str]], tuple[str, ...]]


def complete_with_retries(
    backend: ChatBackend,
    original_prompt: str,
    parser: Callable[[str], Parsed],
    *,
    max_retries: int,
    max_tokens: int,
    label: str,
    progress: Callable[[str], None],
    include_rejected_candidate: bool = True,
    report_candidate: bool = True,
    error_formatter: ErrorFormatter | None = None,
) -> Parsed:
    prompt = original_prompt
    last_errors: tuple[str, ...] = ()
    for attempt in range(1, max_retries + 1):
        candidate = backend.complete(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
        if report_candidate:
            progress(
                f"[{label}] attempt {attempt}/{max_retries} generated output:\n"
                f"{candidate}"
            )
        try:
            return parser(candidate)
        except InsufficientContentError as exc:
            raise GenerationError(f"more content is required: {exc}") from exc
        except ValidationError as exc:
            raw_errors = exc.errors
            last_errors = (
                error_formatter(raw_errors) if error_formatter is not None else raw_errors
            )
            progress(
                f"[{label}] attempt {attempt}/{max_retries} rejected: "
                + " | ".join(last_errors)
            )
            rejected = candidate if include_rejected_candidate else None
            prompt = build_retry_prompt(original_prompt, rejected, last_errors)
            if attempt < max_retries:
                progress(
                    f"{label}: {' | '.join(last_errors[:3])}. Retrying after "
                    f"validation errors (attempt {attempt + 1}/{max_retries})."
                )
    details = "; ".join(last_errors) if last_errors else "unknown validation error"
    raise GenerationError(f"{label} failed after {max_retries} attempts: {details}")
```

In `flashcard_pipeline.py`, import `GenerationError` and `complete_with_retries`, remove the local exception class, and make the existing private method a compatibility delegate:

```python
def _complete_with_retries(
    self,
    original_prompt,
    parser,
    *,
    max_tokens,
    label,
    include_rejected_candidate=True,
    report_candidate=True,
    error_formatter=None,
):
    return complete_with_retries(
        self.backend,
        original_prompt,
        parser,
        max_retries=self.config.max_retries,
        max_tokens=max_tokens,
        label=label,
        progress=self._progress,
        include_rejected_candidate=include_rejected_candidate,
        report_candidate=report_candidate,
        error_formatter=error_formatter,
    )
```

- [ ] **Step 4: Run retry and flashcard pipeline tests**

Run: `python -m pytest tests/test_generation_retry.py tests/test_flashcard_pipeline.py -v`

Expected: PASS with existing card/review retry behavior preserved.

- [ ] **Step 5: Commit retry extraction**

```powershell
git add generation_retry.py flashcard_pipeline.py tests/test_generation_retry.py
git commit -m "refactor: isolate generation retries"
```

---

### Task 7: Focused Concept Planning and Grouped Errors

**Files:**
- Create: `concept_planning.py`
- Create: `tests/test_concept_planning.py`
- Modify: `flashcard_pipeline.py:35-82,292-316`
- Modify: `tests/test_flashcard_pipeline.py`

**Interfaces:**
- Consumes: `complete_with_retries`, `build_concept_plan_prompt`, `parse_concept_plan`, and grounded `GraphFact` values.
- Produces: `balanced_plan_facts(facts: Sequence[GraphFact], limit: int = 50) -> tuple[GraphFact, ...]`, `summarize_concept_errors(errors: Sequence[str]) -> tuple[str, ...]`, and `plan_concepts(backend, identity, facts, *, prior_concept_names=(), max_retries=3, max_tokens=3072, progress=print) -> tuple[ConceptPlan, ...]`.

- [ ] **Step 1: Write fact-balancing, error-grouping, and retry tests**

```python
import json

import pytest

from concept_planning import (
    balanced_plan_facts,
    plan_concepts,
    summarize_concept_errors,
)
from flashcard_types import GraphFact, ModuleIdentity
from generation_retry import GenerationError


class RepeatingBackend:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return self.response


def test_balanced_plan_facts_covers_all_slides_with_a_fixed_limit():
    facts = tuple(
        GraphFact(
            f"f{index}",
            f"Grounded fact {index} contains enough instructional detail.",
            slides=((index - 1) % 3 + 1,),
        )
        for index in range(1, 61)
    )

    selected = balanced_plan_facts(facts)

    assert len(selected) == 50
    assert {fact.slides[0] for fact in selected} == {1, 2, 3}


def test_summarize_concept_errors_collapses_repeated_positions():
    errors = tuple(
        f"concept {index} fact_ids must be a non-empty JSON array"
        for index in range(1, 21)
    )

    assert summarize_concept_errors(errors) == (
        "concepts 1-20 fact_ids must be a non-empty JSON array",
    )


def test_plan_failure_is_bounded_grouped_and_does_not_log_bulk_candidate():
    invalid = json.dumps(
        {
            "concepts": [
                {
                    "name": f"Concept {index}",
                    "assessment_approaches": [
                        "recall",
                        "comparison",
                        "application",
                        "misconception detection",
                        "reversed reasoning",
                    ],
                }
                for index in range(1, 21)
            ]
        }
    )
    backend = RepeatingBackend(invalid)
    messages = []
    facts = tuple(
        GraphFact(f"f{index}", f"Grounded fact {index} supports assessment.")
        for index in range(1, 21)
    )

    with pytest.raises(
        GenerationError,
        match="concepts 1-20 fact_ids must be a non-empty JSON array",
    ):
        plan_concepts(
            backend,
            ModuleIdentity("CPE0021", "1"),
            facts,
            max_retries=3,
            progress=messages.append,
        )

    assert len(backend.calls) == 3
    assert all(invalid not in message for message in messages)
```

- [ ] **Step 2: Run the concept-planning tests and verify the missing module failure**

Run: `python -m pytest tests/test_concept_planning.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'concept_planning'`.

- [ ] **Step 3: Implement balanced facts, range formatting, and strict planning**

Create `concept_planning.py`:

```python
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
import re

from flashcard_contract import CONCEPTS_PER_MODULE
from flashcard_prompt import build_concept_plan_prompt
from flashcard_types import ChatBackend, ConceptPlan, GraphFact, ModuleIdentity
from flashcard_validator import parse_concept_plan
from generation_retry import complete_with_retries


PLAN_FACT_LIMIT = 50
_CONCEPT_ERROR = re.compile(r"^concept (\d+) (.+)$")


def balanced_plan_facts(
    facts: Sequence[GraphFact],
    limit: int = PLAN_FACT_LIMIT,
) -> tuple[GraphFact, ...]:
    values = tuple(facts)
    if len(values) <= limit:
        return values
    buckets: dict[tuple[str, object], list[GraphFact]] = {}
    for fact in values:
        if fact.slides:
            key: tuple[str, object] = ("slide", fact.slides[0])
        elif fact.topic:
            key = ("topic", fact.topic.casefold())
        else:
            key = ("general", "")
        buckets.setdefault(key, []).append(fact)
    selected: list[GraphFact] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets.values():
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)


def _position_ranges(positions: Sequence[int]) -> str:
    ordered = sorted(set(positions))
    groups: list[str] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = position
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(groups)


def summarize_concept_errors(errors: Sequence[str]) -> tuple[str, ...]:
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    passthrough: list[str] = []
    for error in errors:
        match = _CONCEPT_ERROR.fullmatch(str(error))
        if match is None:
            if error not in passthrough:
                passthrough.append(str(error))
            continue
        position, detail = match.groups()
        grouped.setdefault(detail, []).append(int(position))
    summaries = [
        (
            f"concept {positions[0]} {detail}"
            if len(positions) == 1
            else f"concepts {_position_ranges(positions)} {detail}"
        )
        for detail, positions in grouped.items()
    ]
    return tuple((*summaries, *passthrough))


def plan_concepts(
    backend: ChatBackend,
    identity: ModuleIdentity,
    facts: Sequence[GraphFact],
    *,
    prior_concept_names: Sequence[str] = (),
    max_retries: int = 3,
    max_tokens: int = 3072,
    progress: Callable[[str], None] = print,
) -> tuple[ConceptPlan, ...]:
    selected = balanced_plan_facts(facts)
    progress(
        f"Planning {CONCEPTS_PER_MODULE} concepts from "
        f"{len(selected)} grounded lesson facts..."
    )
    prompt = build_concept_plan_prompt(identity, selected, prior_concept_names)
    return complete_with_retries(
        backend,
        prompt,
        lambda raw: parse_concept_plan(raw, selected),
        max_retries=max_retries,
        max_tokens=max_tokens,
        label="concept plan",
        progress=progress,
        include_rejected_candidate=False,
        report_candidate=False,
        error_formatter=summarize_concept_errors,
    )
```

Replace the inline planning section in `FlashcardPipeline.run` with:

```python
concepts = plan_concepts(
    self.backend,
    identity,
    facts,
    prior_concept_names=prior_concept_names,
    max_retries=self.config.max_retries,
    max_tokens=self.config.plan_max_tokens,
    progress=self._progress,
)
```

Import `balanced_plan_facts as _balanced_plan_facts` so the existing private test/import surface remains stable during this cleanup.

- [ ] **Step 4: Run concept, validator, prompt, and pipeline tests**

Run: `python -m pytest tests/test_concept_planning.py tests/test_flashcard_validator.py tests/test_flashcard_prompt.py tests/test_flashcard_pipeline.py -v`

Expected: PASS. The strict `fact_ids` contract remains enforced, retries remain capped at three by default, and repeated concept errors are grouped.

- [ ] **Step 5: Commit focused concept planning**

```powershell
git add concept_planning.py flashcard_pipeline.py tests/test_concept_planning.py tests/test_flashcard_pipeline.py
git commit -m "refactor: isolate grounded concept planning"
```

---

### Task 8: Remove Remaining Duplicate JSON Writers and Document Migration

**Files:**
- Modify: `main.py:1-70`
- Modify: `batch_pipeline.py:1-12,140-168`
- Modify: `tests/test_main.py`
- Modify: `tests/test_batch_pipeline.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `write_json_atomic` from Task 1.
- Produces: unchanged `course_corpus.json` and `batch_reuse_manifest.json` shapes written through one tested persistence boundary.

- [ ] **Step 1: Add assertions that application state writes leave no temporary siblings**

After existing course-corpus and batch-manifest write calls, add these assertions to their focused tests:

```python
assert list(output_directory.glob(".course_corpus.json.*.tmp")) == []
```

and:

```python
assert list(item.paths.workspace.glob(".batch_reuse_manifest.json.*.tmp")) == []
```

Use the existing test-local directory and `BatchItem` variable names in each test rather than creating a second setup path.

- [ ] **Step 2: Run the focused state-write tests before migration**

Run: `python -m pytest tests/test_main.py tests/test_batch_pipeline.py -v`

Expected: PASS; these characterization assertions protect the cleanup from changing observable file behavior.

- [ ] **Step 3: Replace both handwritten temporary-file blocks**

In `main.append_course_corpus`, write the already merged payload with:

```python
write_json_atomic(
    path,
    {
        "concept_names": merged_concepts,
        "questions": merged_questions,
    },
    ensure_ascii=False,
    indent=2,
)
```

In `batch_pipeline._write_manifest`, use:

```python
write_json_atomic(
    _manifest_path(item),
    _manifest_contents(item),
    sort_keys=True,
)
```

Import `write_json_atomic` in both modules and remove their unused `tempfile` imports. Keep `json` because both modules still read JSON.

Add this README paragraph in the pipeline resume section:

```markdown
Current releases version normalized text and knowledge-graph artifacts. If an
older portable `data` directory contains version-1 normalized text or an
edge-only graph, the pipeline keeps the uploaded PDF and automatically rebuilds
that stage and its downstream flashcards. Direct graph loading remains compatible
with legacy edge-based JSON files.
```

- [ ] **Step 4: Verify state persistence and scan for the removed duplication**

Run: `python -m pytest tests/test_artifact_io.py tests/test_main.py tests/test_batch_pipeline.py -v`

Expected: PASS.

Run: `rg -n "NamedTemporaryFile" main.py batch_pipeline.py text_extractor.py`

Expected: no matches.

- [ ] **Step 5: Commit the persistence cleanup and documentation**

```powershell
git add main.py batch_pipeline.py tests/test_main.py tests/test_batch_pipeline.py README.md
git commit -m "refactor: reuse atomic state persistence"
```

---

### Task 9: Generate the Transfer Checksum with the Archive

**Files:**
- Modify: `packaging/create_archive.py`
- Modify: `tests/test_portable_archive.py`

**Interfaces:**
- Consumes: the completed ZIP path returned by `create_archive`.
- Produces: `write_sha256_file(archive: Path) -> Path`, which writes `<archive>.sha256` as ASCII using the standard `<hash><two spaces><filename>` format.

- [ ] **Step 1: Write a failing checksum-sidecar test**

```python
import hashlib


def test_write_sha256_file_matches_archive_bytes(tmp_path):
    archive_module = _load_archive_module()
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"portable archive bytes")

    checksum_path = archive_module.write_sha256_file(archive)

    expected = hashlib.sha256(b"portable archive bytes").hexdigest().upper()
    assert checksum_path == Path(f"{archive}.sha256")
    assert checksum_path.read_text(encoding="ascii") == (
        f"{expected}  {archive.name}\n"
    )
```

- [ ] **Step 2: Run the archive test and verify the missing function failure**

Run: `python -m pytest tests/test_portable_archive.py::test_write_sha256_file_matches_archive_bytes -v`

Expected: FAIL with `AttributeError` because `write_sha256_file` is not defined.

- [ ] **Step 3: Implement checksum streaming and call it from the archive CLI**

Add to `packaging/create_archive.py`:

```python
import hashlib


def write_sha256_file(archive: Path) -> Path:
    archive_path = Path(archive).resolve()
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum_path = Path(f"{archive_path}.sha256")
    checksum_path.write_text(
        f"{digest.hexdigest().upper()}  {archive_path.name}\n",
        encoding="ascii",
    )
    return checksum_path
```

Update `main` so every CLI-created release receives a matching checksum:

```python
result = create_archive(args.source_dir, args.output)
checksum = write_sha256_file(result)
print(f"Portable archive created at {result}")
print(f"SHA-256 checksum created at {checksum}")
return 0
```

- [ ] **Step 4: Run portable archive tests**

Run: `python -m pytest tests/test_portable_archive.py -v`

Expected: PASS with both ZIP64 archive and checksum behavior covered.

- [ ] **Step 5: Commit automatic checksum creation**

```powershell
git add packaging/create_archive.py tests/test_portable_archive.py
git commit -m "build: generate portable archive checksum"
```

---

### Task 10: Full Verification and Portable Release Refresh

**Files:**
- Verify: all tracked Python and test files
- Generate but do not commit: `dist/ModuleToFlashcards/`
- Generate but do not commit: `dist/ModuleToFlashcards-1.1.0-windows-x64.zip`
- Generate but do not commit: `dist/ModuleToFlashcards-1.1.0-windows-x64.zip.sha256`

**Interfaces:**
- Consumes: the completed source cleanup and existing locked portable assets.
- Produces: test evidence, a verified executable, a fresh transfer ZIP, and its matching SHA-256 checksum.

- [ ] **Step 1: Compile every tracked Python source file**

Run:

```powershell
$trackedPython = git ls-files "*.py"
python -m py_compile $trackedPython
```

Expected: exit code 0 and no syntax-error output.

- [ ] **Step 2: Run the complete automated test suite**

Run: `python -m pytest -q`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 3: Inspect source changes and Git ignore behavior**

Run:

```powershell
git diff --check
git status --short
git check-ignore -v dist/ModuleToFlashcards/ModuleToFlashcards.exe dist/ModuleToFlashcards-1.1.0-windows-x64.zip
```

Expected: `git diff --check` prints nothing; source status contains only intended tracked changes if a final verification edit was required; both generated paths are ignored by `.gitignore`.

- [ ] **Step 4: Build and verify the portable directory and archive**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_portable.ps1 -SkipGpuPreflight -SkipAssetPreparation
```

Expected: the script completes, `ModuleToFlashcards.exe --verify` succeeds inside the build, and a fresh versioned ZIP is produced.

- [ ] **Step 5: Smoke-test the executable health endpoint**

Run:

```powershell
$portableExe = Resolve-Path "dist\ModuleToFlashcards\ModuleToFlashcards.exe"
$portableProcess = Start-Process -FilePath $portableExe -ArgumentList "--port", "8765" -PassThru -WindowStyle Hidden
try {
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod "http://127.0.0.1:8765/health"
        }
        catch {
            $health = $null
        }
    } while ($null -eq $health -and (Get-Date) -lt $deadline)
    if ($null -eq $health -or $health.status -ne "ok") {
        throw "Portable health check did not return status ok"
    }
    $health | ConvertTo-Json -Compress
}
finally {
    Stop-Process -Id $portableProcess.Id -Force -ErrorAction SilentlyContinue
}
```

Expected: compact JSON containing `"status":"ok"` and version `1.1.0`.

- [ ] **Step 6: Verify the archive checksum generated by the build**

Run:

```powershell
$archive = Resolve-Path "dist\ModuleToFlashcards-1.1.0-windows-x64.zip"
$actual = (Get-FileHash $archive -Algorithm SHA256).Hash
$expected = ((Get-Content -LiteralPath "$archive.sha256" -Raw).Trim() -split "\s+")[0]
if ($actual -ne $expected) {
    throw "Portable archive SHA-256 mismatch"
}
Write-Host "Verified SHA-256: $actual"
```

Expected: `Verified SHA-256:` is printed and the command exits without a mismatch. The previous checksum is not reused because the rebuilt archive has new source code.

- [ ] **Step 7: Record the final source checkpoint**

If verification required no source edits, no extra commit is needed. If a verification-only source correction was necessary, rerun Steps 1-6 and commit only that correction:

```powershell
git add README.md artifact_contracts.py artifact_io.py lesson_facts.py graph_builder.py generation_retry.py concept_planning.py structured_module.py text_extractor.py pipeline.py flashcard_pipeline.py main.py batch_pipeline.py tests
git commit -m "test: finalize Python cleanup verification"
```

Do not add `build/`, `dist/`, `models/`, `data/`, or `packaging/assets/` to Git.
