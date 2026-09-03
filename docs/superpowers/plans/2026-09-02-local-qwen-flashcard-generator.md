# Local Qwen Flashcard Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line program that downloads and runs the official Qwen2.5 3B Instruct Q8_0 GGUF locally, generates 100 graph-grounded flashcards, validates them, and saves two copy-paste-ready 50-row CSV blocks.

**Architecture:** Keep model access behind a small backend protocol, plan 20 concepts once, and generate one five-card cluster per model call. Parse JSON internally, enforce structural guarantees and duplicate rules in Python, perform bounded model-assisted reviews, and render CSV only after the full module passes.

**Tech Stack:** Python 3, `llama-cpp-python`, `huggingface-hub`, standard-library `argparse`/`csv`/`json`/`uuid`, and `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-02-local-qwen-flashcard-generator-design.md`

## Global Constraints

- Use Hugging Face repository `Qwen/Qwen2.5-3B-Instruct-GGUF` and exact file `qwen2.5-3b-instruct-q8_0.gguf`.
- Use `llama-cpp-python`; default to a 32,768-token context, temperature `0.2`, and `n_gpu_layers=-1`.
- Treat only graph nodes, attributes, and relationships as factual authority; do not send provenance or evidence-chunk text to generation prompts.
- Course-code precedence is command-line value, then graph metadata; module-number precedence is graph metadata, then command-line value.
- Preserve course codes and module numbers as exact strings.
- Generate exactly 20 concepts, five cards per concept, 20 valid UUID clusters, and 100 cards when sufficient content exists.
- Use only `multiple-choice`, `identification`, and `true-false`; every cluster contains at least one of each and five distinct assessment approaches.
- Render exactly these columns in order: `Type,Question,Correct Option,Wrong Option 1,Wrong Option 2,Wrong Option 3,is_true,expalanation,hint,difficulty,cluster,course code,module number`.
- Split the output into ten complete clusters in `Module <n>.1` and ten complete clusters in `Module <n>.2`.
- Stop rather than invent, pad, or write partial output when source content or model output remains invalid.
- Retry invalid planning or cluster responses at most three times by default.
- Treat stems as near duplicates when normalized token-set Jaccard similarity is at least `0.85` and normalized character-sequence similarity is at least `0.88`.
- Default tests must not download or load the model.
- The current directory is not a Git repository. Do not initialize Git without user authorization; checkpoint steps note the files that would be committed if Git becomes available.

## File Structure

- Create `flashcard_types.py`: immutable domain records and model-backend protocol.
- Create `graph_input.py`: graph loading, identity resolution, and source-fact extraction.
- Create `flashcard_prompt.py`: system prompt plus planning, cluster, retry, and review prompt builders.
- Create `flashcard_validator.py`: strict JSON parsing and plan, card, cluster, module, and duplicate validation.
- Create `local_qwen.py`: exact model download and `llama-cpp-python` completion adapter.
- Create `flashcard_pipeline.py`: planning, generation, retry, review, regeneration, and final validation orchestration.
- Create `flashcard_csv.py`: exact CSV rendering, 50/50 splitting, and atomic output writing.
- Modify `main.py`: CLI composition and user-facing exit behavior.
- Create `requirements.txt`, `.gitignore`, and `README.md`: installation, model storage, commands, and troubleshooting.
- Create focused tests under `tests/`, including a fake backend for all default end-to-end coverage.

---

### Task 1: Domain records and graph input

**Files:**
- Create: `flashcard_types.py`
- Create: `graph_input.py`
- Create: `tests/test_graph_input.py`

**Interfaces:**
- Consumes: Knowledge-graph dictionaries with `metadata`, `nodes`, and `edges`.
- Produces: `ModuleIdentity`, `GraphFact`, `ConceptPlan`, `FlashcardDraft`, `FlashcardCluster`, `ReviewIssue`, and `ChatBackend`.
- Produces: `load_graph(path: Path) -> dict[str, Any]`.
- Produces: `resolve_identity(graph: Mapping[str, Any], course_code: str | None, module_number: str | None) -> ModuleIdentity`.
- Produces: `extract_graph_facts(graph: Mapping[str, Any]) -> tuple[GraphFact, ...]`.

- [ ] **Step 1: Write failing graph-input tests**

```python
# tests/test_graph_input.py
import json
from pathlib import Path

import pytest

from graph_input import GraphInputError, extract_graph_facts, load_graph, resolve_identity


def graph(metadata=None):
    return {
        "metadata": metadata or {},
        "nodes": [{"id": "n1", "label": "Binary", "degree": 2}],
        "edges": [{
            "id": "e1",
            "source": "n1",
            "target": "n2",
            "subject": "binary",
            "relation": "uses",
            "object": "base 2",
            "evidence": [{"chunk_id": 1, "slides": [3], "text": "do not expose"}],
        }],
    }


def test_identity_precedence_preserves_exact_strings():
    identity = resolve_identity(
        graph({"course_code": "GRAPH101", "module_number": "01"}),
        course_code="CLI-202",
        module_number="2",
    )
    assert identity.course_code == "CLI-202"
    assert identity.module_number == "01"


def test_identity_reports_only_missing_value():
    with pytest.raises(GraphInputError, match="course code is required"):
        resolve_identity(graph({"module_number": "1"}), None, None)


def test_extract_facts_excludes_provenance_text():
    facts = extract_graph_facts(graph())
    assert facts[0].statement == "binary | uses | base 2"
    assert "do not expose" not in facts[0].statement


def test_load_graph_rejects_missing_edges(tmp_path: Path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"metadata": {}, "nodes": []}), encoding="utf-8")
    with pytest.raises(GraphInputError, match="edges"):
        load_graph(path)
```

- [ ] **Step 2: Run the tests and confirm the imports fail**

Run: `python -m pytest tests/test_graph_input.py -q`

Expected: collection fails because `graph_input` does not exist.

- [ ] **Step 3: Implement the domain records and graph loader**

```python
# flashcard_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModuleIdentity:
    course_code: str
    module_number: str


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    statement: str


@dataclass(frozen=True)
class ConceptPlan:
    name: str
    fact_ids: tuple[str, ...]
    facts: tuple[str, ...]
    assessment_approaches: tuple[str, ...]


@dataclass(frozen=True)
class FlashcardDraft:
    type: str
    question: str
    correct_option: str
    wrong_option_1: str
    wrong_option_2: str
    wrong_option_3: str
    is_true: int | None
    expalanation: str
    hint: str
    difficulty: int
    assessment_approach: str


@dataclass(frozen=True)
class FlashcardCluster:
    cluster: str
    concept: ConceptPlan
    cards: tuple[FlashcardDraft, ...]


@dataclass(frozen=True)
class ReviewIssue:
    cluster: str
    reasons: tuple[str, ...]


class ChatBackend(Protocol):
    def complete(self, system: str, user: str, *, max_tokens: int) -> str: ...
```

```python
# graph_input.py core behavior
class GraphInputError(ValueError):
    pass


def _metadata_value(metadata, *keys):
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def resolve_identity(graph, course_code, module_number):
    metadata = graph.get("metadata", {})
    graph_course = _metadata_value(metadata, "course_code", "course code")
    graph_module = _metadata_value(metadata, "module_number", "module number")
    selected_course = course_code if course_code not in (None, "") else graph_course
    selected_module = graph_module if graph_module not in (None, "") else module_number
    if selected_course in (None, ""):
        raise GraphInputError("course code is required")
    if selected_module in (None, ""):
        raise GraphInputError("module number is required")
    return ModuleIdentity(str(selected_course), str(selected_module))


def extract_graph_facts(graph):
    facts = []
    for index, edge in enumerate(graph["edges"], start=1):
        fact_id = str(edge.get("id") or f"e{index}")
        parts = [str(edge.get(key, "")).strip() for key in ("subject", "relation", "object")]
        if all(parts):
            facts.append(GraphFact(fact_id, " | ".join(parts)))
    if not facts:
        raise GraphInputError("graph contains no usable relationship facts")
    return tuple(facts)
```

Implement `load_graph` with UTF-8 reading, `json.JSONDecodeError` conversion, root-object checking, and list checks for `nodes` and `edges`.

- [ ] **Step 4: Run the graph-input tests**

Run: `python -m pytest tests/test_graph_input.py -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_types.py`, `graph_input.py`, `tests/test_graph_input.py`.

If Git is available: `git add flashcard_types.py graph_input.py tests/test_graph_input.py && git commit -m "feat: add graph input contracts"`. Otherwise, do not initialize Git; continue after reporting the passing test command.

---

### Task 2: Qwen prompt protocol

**Files:**
- Create: `flashcard_prompt.py`
- Create: `tests/test_flashcard_prompt.py`

**Interfaces:**
- Consumes: `ModuleIdentity`, `GraphFact`, `ConceptPlan`, `FlashcardCluster`, and validation-error strings.
- Produces: `SYSTEM_PROMPT: str`.
- Produces: `build_concept_plan_prompt(identity, facts) -> str`.
- Produces: `build_cluster_prompt(identity, concept) -> str`.
- Produces: `build_retry_prompt(original_prompt, candidate, errors) -> str`.
- Produces: `build_grounding_review_prompt(clusters) -> str`.
- Produces: `build_duplicate_review_prompt(clusters) -> str`.

- [ ] **Step 1: Write failing prompt tests**

```python
# tests/test_flashcard_prompt.py
from flashcard_prompt import (
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_concept_plan_prompt,
    build_retry_prompt,
)
from flashcard_types import ConceptPlan, GraphFact, ModuleIdentity


def test_system_prompt_contains_non_negotiable_rules():
    assert "only the supplied graph facts" in SYSTEM_PROMPT.lower()
    assert "return json only" in SYSTEM_PROMPT.lower()
    assert "multiple-choice" in SYSTEM_PROMPT
    assert "identification" in SYSTEM_PROMPT
    assert "true-false" in SYSTEM_PROMPT
    assert "expalanation" in SYSTEM_PROMPT


def test_plan_prompt_contains_relationships_but_no_provenance():
    prompt = build_concept_plan_prompt(
        ModuleIdentity("CPE0021", "1"),
        (GraphFact("e1", "binary | uses | base 2"),),
    )
    assert "e1: binary | uses | base 2" in prompt
    assert "slide" not in prompt.lower()
    assert '"concepts"' in prompt


def test_cluster_prompt_limits_generation_to_one_concept():
    concept = ConceptPlan(
        "Binary base",
        ("e1",),
        ("binary | uses | base 2",),
        ("recall", "comparison", "application", "misconception detection", "reversed reasoning"),
    )
    prompt = build_cluster_prompt(ModuleIdentity("CPE0021", "1"), concept)
    assert "exactly 5" in prompt
    assert '"cards"' in prompt
    assert "binary | uses | base 2" in prompt


def test_retry_prompt_requests_complete_replacement():
    prompt = build_retry_prompt("ORIGINAL", '{"cards": []}', ["expected 5 cards"])
    assert "complete replacement" in prompt.lower()
    assert "expected 5 cards" in prompt
```

- [ ] **Step 2: Run the prompt tests and confirm failure**

Run: `python -m pytest tests/test_flashcard_prompt.py -q`

Expected: collection fails because `flashcard_prompt` does not exist.

- [ ] **Step 3: Implement the system prompt and builders**

Make `SYSTEM_PROMPT` a single production prompt that incorporates every source, output-quality, type, stem, difficulty, explanation, hint, equation, and no-provenance rule from the approved user prompt. Add these small-model-specific instructions verbatim in meaning:

```python
SYSTEM_PROMPT = """You are a college-level educational assessment generator.
Use only the supplied graph facts as factual authority. Never add outside knowledge or infer unsupported facts.
Return JSON only for every internal request: no Markdown fences, commentary, headings, or CSV.
Never mention a graph, source, module, document, lesson, slide, file, chunk, citation, URL, or source reference in a question, answer, explanation, or hint.
Use only multiple-choice, identification, and true-false.
Difficulty must come from thinking, never confusing wording.
Preserve the required key spelling expalanation.
Before returning JSON, silently check every item against the supplied facts and all requested constraints.
"""
```

The complete constant must also enumerate the exact type-field rules, direct-stem bans, explanation/hint rules, and plain-text equation syntax. Prompt builders must serialize dynamic values with `json.dumps(..., ensure_ascii=False)` instead of string interpolation that can corrupt quotes.

Concept-planning schema:

```json
{"concepts":[{"name":"...","fact_ids":["e1"],"facts":["subject | relation | object"],"assessment_approaches":["recall","comparison","application","misconception detection","reversed reasoning"]}]}
```

Cluster schema:

```json
{"cards":[{"type":"multiple-choice","question":"...","correct_option":"...","wrong_option_1":"...","wrong_option_2":"...","wrong_option_3":"...","is_true":null,"expalanation":"...","hint":"...","difficulty":2,"assessment_approach":"application"}]}
```

Review builders must use JSON input and demand `{"issues":[{"cluster":"<uuid>","reasons":["..."]}]}`. Grounding review receives four clusters with their facts and all card content. Duplicate review receives only concept names, UUIDs, and stems for all clusters.

- [ ] **Step 4: Run the prompt tests**

Run: `python -m pytest tests/test_flashcard_prompt.py -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_prompt.py`, `tests/test_flashcard_prompt.py`.

If Git is available: commit with `feat: add graph-grounded Qwen prompts`. Otherwise, keep the tested files uncommitted without initializing Git.

---

### Task 3: Strict parsing and per-cluster validation

**Files:**
- Create: `flashcard_validator.py`
- Create: `tests/test_flashcard_validator.py`

**Interfaces:**
- Consumes: raw model strings, `ConceptPlan`, and `FlashcardDraft` records.
- Produces: `ValidationError`, whose `.errors` is `tuple[str, ...]`.
- Produces: `parse_concept_plan(raw, known_facts) -> tuple[ConceptPlan, ...]`.
- Produces: `parse_cards(raw) -> tuple[FlashcardDraft, ...]`.
- Produces: `validate_cluster(cards, concept) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing parser and cluster-validator tests**

```python
# tests/test_flashcard_validator.py
import json

import pytest

from flashcard_types import ConceptPlan, FlashcardDraft, GraphFact
from flashcard_validator import ValidationError, parse_cards, parse_concept_plan, validate_cluster


def valid_cards():
    base = dict(
        correct_option="Answer",
        wrong_option_1="Wrong A",
        wrong_option_2="Wrong B",
        wrong_option_3="Wrong C",
        is_true=None,
        expalanation="The supplied relationship supports the answer.",
        hint="Focus on the relationship between the terms.",
        difficulty=2,
    )
    return (
        FlashcardDraft(type="multiple-choice", question="Which relationship is valid?", assessment_approach="comparison", **base),
        FlashcardDraft(type="identification", question="What term completes the relationship?", correct_option="Term", wrong_option_1="", wrong_option_2="", wrong_option_3="", is_true=None, expalanation="The term completes the relationship.", hint="Recall the named term.", difficulty=1, assessment_approach="recall"),
        FlashcardDraft(type="true-false", question="The stated relationship is valid.", correct_option="", wrong_option_1="", wrong_option_2="", wrong_option_3="", is_true=1, expalanation="The relationship is supported.", hint="Check the relationship direction.", difficulty=1, assessment_approach="misconception detection"),
        FlashcardDraft(type="multiple-choice", question="How would the relationship apply in this case?", assessment_approach="application", **base),
        FlashcardDraft(type="true-false", question="Reversing the relationship preserves its meaning.", correct_option="", wrong_option_1="", wrong_option_2="", wrong_option_3="", is_true=0, expalanation="The reversed relationship is not equivalent.", hint="Consider direction.", difficulty=3, assessment_approach="reversed reasoning"),
    )


def test_parser_rejects_markdown_around_json():
    with pytest.raises(ValidationError, match="JSON object only"):
        parse_cards('```json\n{"cards": []}\n```')


def test_concept_plan_requires_twenty_supported_concepts():
    raw = json.dumps({"concepts": [{
        "name": "Binary",
        "fact_ids": ["e1"],
        "facts": ["binary | uses | base 2"],
        "assessment_approaches": ["recall", "comparison", "application", "conditions", "reversed reasoning"],
    }]})
    with pytest.raises(ValidationError, match="exactly 20 concepts"):
        parse_concept_plan(raw, (GraphFact("e1", "binary | uses | base 2"),))


def test_valid_cluster_has_no_errors():
    concept = ConceptPlan("Binary", ("e1",), ("binary | uses | base 2",), tuple(card.assessment_approach for card in valid_cards()))
    assert validate_cluster(valid_cards(), concept) == ()


def test_identification_rejects_sentence_answer():
    cards = list(valid_cards())
    cards[1] = FlashcardDraft(**{**cards[1].__dict__, "correct_option": "It is the term that completes the relationship."})
    errors = validate_cluster(tuple(cards), ConceptPlan("Binary", ("e1",), ("fact",), tuple(card.assessment_approach for card in cards)))
    assert any("concise phrase" in error for error in errors)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `python -m pytest tests/test_flashcard_validator.py -q`

Expected: collection fails because `flashcard_validator` does not exist.

- [ ] **Step 3: Implement strict JSON and validation rules**

`_parse_json_object` must require the stripped response to begin with `{`, end with `}`, parse once with `json.loads`, and produce a dictionary. Do not attempt to scrape JSON from prose.

Implement conversion helpers that require every expected key and reject unknown card types. Convert `is_true` without coercing booleans, strings, or floats: only integer `0`, integer `1`, or JSON `null` are accepted.

Use these constants:

```python
ALLOWED_TYPES = {"multiple-choice", "identification", "true-false"}
ALLOWED_APPROACHES = {
    "recall", "comparison", "classification", "application", "scenario analysis",
    "cause/effect", "misconception detection", "conditions", "consequences", "reversed reasoning",
}
BANNED_FRAMING = (
    "according to", "based on", "the material states", "the following claim",
    "consider this statement", "evaluate this statement",
    "identify the concept associated with",
)
PROVENANCE_TERMS = ("knowledge graph", "source", "module", "document", "lesson", "slide", "file", "chunk", "citation", "url")
```

Validate all type-specific fields, five distinct approaches, question form for multiple-choice and identification, declarative form for true-false, explanations/hints, no embedded newlines, no option labels in stems, difficulty range, concise identification answers (maximum 12 words, no terminal sentence punctuation), and hints that do not contain the normalized full answer.

Validate concept plans by checking exactly 20 entries, unique normalized names, exactly five distinct allowed approaches, non-empty fact lists, known fact IDs, and exact equality between supplied fact strings and the known statements for those IDs. This prevents the model from inventing its own planning evidence.

- [ ] **Step 4: Run per-cluster validation tests**

Run: `python -m pytest tests/test_flashcard_validator.py -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_validator.py`, `tests/test_flashcard_validator.py`.

If Git is available: commit with `feat: validate generated flashcard clusters`. Otherwise, record the passing test command only.

---

### Task 4: Module validation and duplicate detection

**Files:**
- Modify: `flashcard_validator.py`
- Create: `tests/test_module_validation.py`

**Interfaces:**
- Consumes: ordered `FlashcardCluster` values.
- Produces: `normalize_stem(value: str) -> str`.
- Produces: `are_near_duplicates(left: str, right: str) -> bool`.
- Produces: `validate_module(clusters) -> tuple[str, ...]`.
- Produces: `parse_review_issues(raw, known_clusters) -> tuple[ReviewIssue, ...]`.

- [ ] **Step 1: Write failing module-validation tests**

```python
# tests/test_module_validation.py
import json
from dataclasses import replace

import pytest

from flashcard_types import FlashcardCluster
from flashcard_validator import are_near_duplicates, parse_review_issues, validate_module
from tests.test_flashcard_validator import valid_cards


def make_clusters(concepts):
    return tuple(
        FlashcardCluster(
            cluster=f"00000000-0000-4000-8000-{index:012d}",
            concept=concept,
            cards=tuple(replace(card, question=f"{card.question.rstrip('? .')} concept {index}{'?' if card.type != 'true-false' else '.'}") for card in valid_cards()),
        )
        for index, concept in enumerate(concepts, start=1)
    )


def test_near_duplicate_threshold_requires_both_metrics():
    assert are_near_duplicates(
        "Which number system uses base two for representing values?",
        "Which number system uses base 2 to represent values?",
    )
    assert not are_near_duplicates(
        "Which number system uses base two?",
        "How does parity detect transmission errors?",
    )


def test_module_requires_twenty_clusters(concepts):
    errors = validate_module(make_clusters(concepts[:19]))
    assert any("exactly 20 clusters" in error for error in errors)


def test_review_parser_rejects_unknown_cluster():
    raw = json.dumps({"issues": [{"cluster": "not-known", "reasons": ["unsupported claim"]}]})
    with pytest.raises(Exception, match="unknown cluster"):
        parse_review_issues(raw, {"known-id"})
```

Add a local `concepts` pytest fixture with 20 unique `ConceptPlan` objects, or move reusable test factories to `tests/factories.py` and import them from both test modules. Do not make production code depend on test modules.

- [ ] **Step 2: Run the module tests and confirm failure**

Run: `python -m pytest tests/test_module_validation.py -q`

Expected: imports fail for the new validator functions.

- [ ] **Step 3: Implement normalization, similarity, and module invariants**

```python
def normalize_stem(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def are_near_duplicates(left: str, right: str) -> bool:
    a, b = normalize_stem(left), normalize_stem(right)
    if a == b:
        return True
    a_tokens, b_tokens = set(a.split()), set(b.split())
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 1.0
    sequence = difflib.SequenceMatcher(None, a, b).ratio()
    return jaccard >= 0.85 and sequence >= 0.88
```

`validate_module` must validate UUID syntax with `uuid.UUID`, unique UUIDs, 20 clusters, five cards each, 100 cards, each cluster via `validate_cluster`, and all pairwise stems for duplicates. Prefix errors with cluster or card positions so retries can target the right cluster.

`parse_review_issues` must strictly parse the review schema, accept only known UUIDs, require non-empty string reasons, combine multiple entries for one UUID, and preserve first-seen cluster order.

- [ ] **Step 4: Run all validator tests**

Run: `python -m pytest tests/test_flashcard_validator.py tests/test_module_validation.py -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_validator.py`, `tests/test_module_validation.py`, and `tests/factories.py` if created.

If Git is available: commit with `feat: enforce module-wide flashcard invariants`. Otherwise, continue without repository initialization.

---

### Task 5: Exact Hugging Face model download and local inference

**Files:**
- Create: `local_qwen.py`
- Create: `tests/test_local_qwen.py`
- Create: `requirements.txt`
- Create: `.gitignore`

**Interfaces:**
- Consumes: model directory, inference configuration, and prompt strings.
- Produces: `MODEL_REPO`, `MODEL_FILENAME`, `ensure_model(model_dir: Path) -> Path`.
- Produces: `LocalQwenBackend`, implementing `ChatBackend.complete(system, user, *, max_tokens) -> str`.

- [ ] **Step 1: Write failing download and backend tests using monkeypatches**

```python
# tests/test_local_qwen.py
from pathlib import Path

from local_qwen import MODEL_FILENAME, MODEL_REPO, LocalQwenBackend, ensure_model


def test_ensure_model_downloads_exact_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"gguf")
        return str(target)

    monkeypatch.setattr("local_qwen.hf_hub_download", fake_download)
    path = ensure_model(tmp_path)
    assert path.name == MODEL_FILENAME
    assert calls == [{
        "repo_id": MODEL_REPO,
        "filename": MODEL_FILENAME,
        "local_dir": str(tmp_path),
    }]


def test_existing_model_is_reused(tmp_path, monkeypatch):
    path = tmp_path / MODEL_FILENAME
    path.write_bytes(b"gguf")
    monkeypatch.setattr("local_qwen.hf_hub_download", lambda **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert ensure_model(tmp_path) == path


def test_backend_passes_chat_messages_and_returns_content():
    calls = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"cards": []}'}}]}

    backend = LocalQwenBackend.__new__(LocalQwenBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    assert backend.complete("SYSTEM", "USER", max_tokens=512) == '{"cards": []}'
    assert calls[0]["messages"] == [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "USER"},
    ]
```

- [ ] **Step 2: Install only the test dependency and confirm failure**

Run: `python -m pip install pytest`

Run: `python -m pytest tests/test_local_qwen.py -q`

Expected: collection fails because `local_qwen` does not exist. If dependency installation is blocked by network restrictions, rerun only after obtaining the required network approval.

- [ ] **Step 3: Implement lazy runtime imports and exact download**

```python
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-3b-instruct-q8_0.gguf"


def ensure_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / MODEL_FILENAME
    if target.is_file():
        return target
    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(model_dir),
    )
    return Path(downloaded)


class LocalQwenBackend:
    def __init__(self, model_path, *, n_ctx=32768, n_gpu_layers=-1, temperature=0.2, seed=42):
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        self._temperature = temperature
        self._seed = seed

    def complete(self, system, user, *, max_tokens):
        response = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self._temperature,
            seed=self._seed,
            max_tokens=max_tokens,
        )
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Qwen returned empty assistant content")
        return content.strip()
```

Keep `huggingface_hub` import at module scope for monkeypatching, but keep `llama_cpp` import inside initialization so model-free tests can run when the native runtime is absent.

`requirements.txt`:

```text
huggingface-hub
llama-cpp-python
pytest
```

`.gitignore`:

```text
__pycache__/
.pytest_cache/
.venv/
models/
flashcards/
*.pyc
```

- [ ] **Step 4: Run model-adapter tests without downloading a model**

Run: `python -m pytest tests/test_local_qwen.py -q`

Expected: all tests pass and `models/` remains absent.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `local_qwen.py`, `tests/test_local_qwen.py`, `requirements.txt`, `.gitignore`.

If Git is available: commit with `feat: add local Qwen model backend`. Otherwise, continue after recording test evidence.

---

### Task 6: Generation, retry, and review pipeline

**Files:**
- Create: `flashcard_pipeline.py`
- Create: `tests/test_flashcard_pipeline.py`
- Create: `tests/factories.py` if not created in Task 4

**Interfaces:**
- Consumes: `ChatBackend`, `ModuleIdentity`, ordered `GraphFact` values, and prompt/validator functions.
- Produces: `GenerationError`.
- Produces: `PipelineConfig(max_retries=3, plan_max_tokens=6144, cluster_max_tokens=3072, review_max_tokens=3072, final_review=True)`.
- Produces: `FlashcardPipeline.run(identity, facts) -> tuple[FlashcardCluster, ...]`.

- [ ] **Step 1: Write failing fake-backend pipeline tests**

```python
# tests/test_flashcard_pipeline.py
from collections import deque

import pytest

from flashcard_pipeline import FlashcardPipeline, GenerationError, PipelineConfig
from flashcard_types import ModuleIdentity
from tests.factories import plan_json, cluster_json, review_json, graph_facts


class FakeBackend:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def complete(self, system, user, *, max_tokens):
        self.calls.append((system, user, max_tokens))
        return self.responses.popleft()


def test_pipeline_generates_twenty_valid_clusters_without_review():
    backend = FakeBackend([plan_json()] + [cluster_json(index) for index in range(20)])
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=False))
    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())
    assert len(clusters) == 20
    assert len({cluster.cluster for cluster in clusters}) == 20
    assert all(len(cluster.cards) == 5 for cluster in clusters)


def test_invalid_cluster_is_retried_with_validator_feedback():
    responses = [plan_json(), '{"cards": []}', cluster_json(0)]
    responses.extend(cluster_json(index) for index in range(1, 20))
    backend = FakeBackend(responses)
    pipeline = FlashcardPipeline(backend, PipelineConfig(max_retries=3, final_review=False))
    pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())
    assert "expected exactly 5 cards" in backend.calls[2][1]


def test_exhausted_retries_do_not_return_partial_results():
    backend = FakeBackend([plan_json(), '{"cards": []}', '{"cards": []}', '{"cards": []}'])
    pipeline = FlashcardPipeline(backend, PipelineConfig(max_retries=3, final_review=False))
    with pytest.raises(GenerationError, match="concept 1"):
        pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())


def test_final_review_regenerates_flagged_cluster_once():
    generated = [plan_json()] + [cluster_json(index) for index in range(20)]
    reviews = [review_json([]) for _ in range(5)] + [review_json([0])]
    backend = FakeBackend(generated + reviews + [cluster_json(0, revision=1)])
    pipeline = FlashcardPipeline(backend, PipelineConfig(final_review=True))
    clusters = pipeline.run(ModuleIdentity("CPE0021", "1"), graph_facts())
    assert "revision 1" in clusters[0].cards[0].question
```

Test factories must generate 20 truly distinct stems that do not trip the deterministic near-duplicate thresholds. Reviewer fixtures must use the UUIDs assigned during the run; allow `FakeBackend` responses to be callables receiving prior calls when dynamic UUID output is needed.

- [ ] **Step 2: Run pipeline tests and confirm failure**

Run: `python -m pytest tests/test_flashcard_pipeline.py -q`

Expected: collection fails because `flashcard_pipeline` does not exist.

- [ ] **Step 3: Implement bounded orchestration**

```python
@dataclass(frozen=True)
class PipelineConfig:
    max_retries: int = 3
    plan_max_tokens: int = 6144
    cluster_max_tokens: int = 3072
    review_max_tokens: int = 3072
    final_review: bool = True


class FlashcardPipeline:
    def __init__(self, backend: ChatBackend, config: PipelineConfig = PipelineConfig()):
        self.backend = backend
        self.config = config

    def _complete_with_retries(self, original_prompt, parser, *, max_tokens, label):
        prompt = original_prompt
        last_errors = ()
        for _attempt in range(1, self.config.max_retries + 1):
            candidate = self.backend.complete(SYSTEM_PROMPT, prompt, max_tokens=max_tokens)
            try:
                return parser(candidate)
            except ValidationError as exc:
                last_errors = exc.errors
                prompt = build_retry_prompt(original_prompt, candidate, last_errors)
        raise GenerationError(f"{label} failed after {self.config.max_retries} attempts: {'; '.join(last_errors)}")
```

`run` performs these exact phases:

1. Build and retry the concept plan.
2. For each concept in order, generate and validate five cards, then assign `str(uuid.uuid4())` once.
3. Run `validate_module`.
4. When enabled, split clusters into five consecutive groups of four for grounding review.
5. Run one global duplicate review with all clusters.
6. Merge review reasons by cluster UUID.
7. Regenerate each flagged cluster once through the normal bounded-retry helper, preserving its original UUID and concept.
8. Run final deterministic module validation and return the ordered clusters.

Convert an explicit model response `{"insufficient_content": "..."}` into `GenerationError("more content is required: ...")` during planning. Do not accept partial concept plans. Do not catch `KeyboardInterrupt`.

- [ ] **Step 4: Run pipeline and all prior tests**

Run: `python -m pytest tests/test_flashcard_pipeline.py tests/test_graph_input.py tests/test_flashcard_prompt.py tests/test_flashcard_validator.py tests/test_module_validation.py tests/test_local_qwen.py -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_pipeline.py`, `tests/test_flashcard_pipeline.py`, `tests/factories.py`.

If Git is available: commit with `feat: orchestrate validated flashcard generation`. Otherwise, continue without changing repository state beyond the planned files.

---

### Task 7: Exact CSV blocks and atomic output

**Files:**
- Create: `flashcard_csv.py`
- Create: `tests/test_flashcard_csv.py`

**Interfaces:**
- Consumes: `ModuleIdentity` and 20 validated `FlashcardCluster` values.
- Produces: `CSV_COLUMNS: tuple[str, ...]`.
- Produces: `render_module(identity, clusters) -> str`.
- Produces: `write_module_output(path: Path, content: str) -> None`.

- [ ] **Step 1: Write failing CSV tests**

```python
# tests/test_flashcard_csv.py
import csv
import io
from pathlib import Path

from flashcard_csv import CSV_COLUMNS, render_module, write_module_output
from flashcard_types import ModuleIdentity
from tests.factories import valid_clusters


def split_blocks(text):
    first_label, remainder = text.split("\n", 1)
    first_csv, second = remainder.split("\n\nModule 1.2\n", 1)
    return first_label, first_csv, second


def test_render_module_has_two_complete_fifty_row_blocks():
    text = render_module(ModuleIdentity("CPE0021", "1"), valid_clusters())
    first_label, first_csv, second_csv = split_blocks(text)
    assert first_label == "Module 1.1"
    first_rows = list(csv.reader(io.StringIO(first_csv)))
    second_rows = list(csv.reader(io.StringIO(second_csv)))
    assert tuple(first_rows[0]) == CSV_COLUMNS
    assert tuple(second_rows[0]) == CSV_COLUMNS
    assert len(first_rows) == 51
    assert len(second_rows) == 51
    assert len({row[10] for row in first_rows[1:]}) == 10
    assert len({row[10] for row in second_rows[1:]}) == 10


def test_csv_escapes_commas_and_quotes():
    text = render_module(ModuleIdentity("CPE,0021", "1"), valid_clusters(question='Which value is "valid", here?'))
    assert '"CPE,0021"' in text
    assert '""valid""' in text


def test_write_module_output_replaces_target(tmp_path: Path):
    target = tmp_path / "module.txt"
    target.write_text("old", encoding="utf-8")
    write_module_output(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.tmp")) == []
```

- [ ] **Step 2: Run CSV tests and confirm failure**

Run: `python -m pytest tests/test_flashcard_csv.py -q`

Expected: collection fails because `flashcard_csv` does not exist.

- [ ] **Step 3: Implement exact rendering and atomic write**

```python
CSV_COLUMNS = (
    "Type", "Question", "Correct Option", "Wrong Option 1", "Wrong Option 2",
    "Wrong Option 3", "is_true", "expalanation", "hint", "difficulty",
    "cluster", "course code", "module number",
)


def _rows(identity, clusters):
    for cluster in clusters:
        for card in cluster.cards:
            yield (
                card.type,
                card.question,
                card.correct_option,
                card.wrong_option_1,
                card.wrong_option_2,
                card.wrong_option_3,
                "" if card.is_true is None else str(card.is_true),
                card.expalanation,
                card.hint,
                str(card.difficulty),
                cluster.cluster,
                identity.course_code,
                identity.module_number,
            )
```

Before rendering, call `validate_module` and raise `ValueError` with joined errors if anything is invalid. Render each half independently through `csv.writer` with `lineterminator="\n"`. Join exactly as `Module <n>.1\n<csv>\nModule <n>.2\n<csv>` with one blank line between blocks and one final newline.

For atomic output, create a named temporary file in `path.parent`, write UTF-8 with `newline=""`, flush and close it, then replace the target with `Path.replace`. On any exception, delete only that exact temporary file in a `finally` block if it still exists.

- [ ] **Step 4: Run CSV and full model-free tests**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 5: Record the checkpoint**

Checkpoint files: `flashcard_csv.py`, `tests/test_flashcard_csv.py`.

If Git is available: commit with `feat: render validated flashcard CSV blocks`. Otherwise, record test evidence and continue.

---

### Task 8: Runnable CLI, documentation, and smoke path

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all preceding public interfaces.
- Produces: `parse_args(argv=None) -> argparse.Namespace`.
- Produces: `run(args) -> Path`.
- Produces: `main() -> int` and `python main.py ...` entry behavior.

- [ ] **Step 1: Write failing CLI composition tests**

```python
# tests/test_main.py
from pathlib import Path

import pytest

import main


def test_parse_args_preserves_identity_strings():
    args = main.parse_args([
        "output/knowledge_graph.json",
        "--course-code", "CPE0021",
        "--module-number", "01",
        "--skip-final-review",
    ])
    assert args.course_code == "CPE0021"
    assert args.module_number == "01"
    assert args.final_review is False


def test_run_validates_graph_before_model_download(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(main, "ensure_model", lambda path: (_ for _ in ()).throw(AssertionError("model touched")))
    args = main.parse_args([str(missing), "--course-code", "C", "--module-number", "1"])
    with pytest.raises(Exception, match="not found"):
        main.run(args)


def test_run_writes_pipeline_result(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text('{"metadata": {}, "nodes": [], "edges": [{"id":"e1","subject":"a","relation":"is","object":"b"}]}', encoding="utf-8")
    output_path = tmp_path / "cards.txt"
    monkeypatch.setattr(main, "ensure_model", lambda path: tmp_path / "model.gguf")
    monkeypatch.setattr(main, "LocalQwenBackend", lambda *a, **k: object())

    class FakePipeline:
        def __init__(self, backend, config): pass
        def run(self, identity, facts): return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args([str(graph_path), "--course-code", "CPE0021", "--module-number", "1", "--output", str(output_path)])
    assert main.run(args) == output_path
    assert output_path.read_text(encoding="utf-8").startswith("Module 1.1\n")
```

Expose shared fixtures through `tests/conftest.py` rather than importing them as function arguments without fixtures.

- [ ] **Step 2: Run CLI tests and confirm failure**

Run: `python -m pytest tests/test_main.py -q`

Expected: failures because the existing `main.py` has no CLI functions.

- [ ] **Step 3: Implement CLI composition and focused errors**

Define arguments for graph path, `--course-code`, `--module-number`, `--output`, `--model-dir`, `--max-retries`, `--seed`, `--n-gpu-layers`, `--skip-final-review`, and `--smoke-test`.

```python
def run(args):
    graph = load_graph(args.graph)
    identity = resolve_identity(graph, args.course_code, args.module_number)
    facts = extract_graph_facts(graph)
    model_path = ensure_model(args.model_dir)
    backend = LocalQwenBackend(
        model_path,
        n_gpu_layers=args.n_gpu_layers,
        seed=args.seed,
    )
    if args.smoke_test:
        backend.complete(
            "Return JSON only.",
            'Return exactly {"status":"ok"}.',
            max_tokens=32,
        )
        return None
    pipeline = FlashcardPipeline(
        backend,
        PipelineConfig(max_retries=args.max_retries, final_review=args.final_review),
    )
    clusters = pipeline.run(identity, facts)
    content = render_module(identity, clusters)
    output = args.output or Path("flashcards") / f"module_{identity.module_number}.txt"
    write_module_output(output, content)
    return output
```

`main()` catches only expected `GraphInputError`, `GenerationError`, model download/load exceptions, and `ValueError`; it prints `Error: ...` to stderr and returns `1`. On success it prints the absolute output path and returns `0`. The module ends with `raise SystemExit(main())`.

- [ ] **Step 4: Document installation and exact commands**

`README.md` must include:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1
```

Document that the first run downloads approximately 3.6 GB into `models/`, CPU inference can be slow, GPU acceleration requires a compatible `llama-cpp-python` build, output defaults to `flashcards/module_1.txt`, and `--skip-final-review` is faster but reduces quality assurance.

Document the smoke command:

```powershell
python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --smoke-test
```

Document Windows installation troubleshooting without promising a universal wheel: if `llama-cpp-python` attempts a native build, install the Visual Studio C++ Build Tools or use a wheel matching the user's CUDA/CPU environment from the runtime's official installation guidance.

- [ ] **Step 5: Run the complete automated suite and CLI help**

Run: `python -m pytest -q`

Expected: all model-free tests pass without creating `models/`.

Run: `python main.py --help`

Expected: exit code `0`, with every documented option listed.

- [ ] **Step 6: Run non-network failure checks**

Run: `python main.py does-not-exist.json --course-code CPE0021 --module-number 1`

Expected: exit code `1`, a focused missing-file error, and no model download attempt.

- [ ] **Step 7: Optionally install runtime dependencies and run the real-model smoke test**

This step downloads external packages and a multi-gigabyte model, so obtain explicit network/download approval immediately before running it.

Run: `python -m pip install -r requirements.txt`

Run: `python main.py output\knowledge_graph.json --course-code CPE0021 --module-number 1 --smoke-test`

Expected: the exact model file exists at `models/qwen2.5-3b-instruct-q8_0.gguf` and the smoke response is accepted. If the native runtime is unavailable for Python 3.13, create and document a Python 3.11 or 3.12 virtual environment rather than changing the approved runtime architecture.

- [ ] **Step 8: Final verification and checkpoint**

Re-run: `python -m pytest -q`

Re-run: `python main.py --help`

Inspect: one fake-backend rendered module with Python's `csv.reader` to confirm two 51-line CSV sections and exact 13-column rows.

Checkpoint files: `main.py`, `README.md`, and all files from Tasks 1-7.

If Git is available: commit with `feat: deliver local Qwen flashcard generator`. Otherwise, do not initialize Git; report that implementation is complete but uncommitted because the project is not a repository.

