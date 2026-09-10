from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flashcard_types import GraphFact, ModuleIdentity


class GraphInputError(ValueError):
    """Raised when a knowledge graph cannot be used safely."""


def load_graph(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise GraphInputError(f"knowledge graph not found: {path}")

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise GraphInputError(f"knowledge graph is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise GraphInputError(f"could not read knowledge graph: {exc}") from exc

    if not isinstance(value, dict):
        raise GraphInputError("knowledge graph root must be a JSON object")
    if "nodes" not in value or not isinstance(value["nodes"], list):
        raise GraphInputError("knowledge graph must contain a nodes list")
    if "edges" not in value or not isinstance(value["edges"], list):
        raise GraphInputError("knowledge graph must contain an edges list")
    if "metadata" in value and not isinstance(value["metadata"], dict):
        raise GraphInputError("knowledge graph metadata must be an object")
    return value


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _provided(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value)


def resolve_identity(
    graph: Mapping[str, Any],
    course_code: str | None,
    module_number: str | None,
) -> ModuleIdentity:
    metadata_value = graph.get("metadata", {})
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    graph_course = _metadata_value(metadata, "course_code", "course code")
    graph_module = _metadata_value(metadata, "module_number", "module number")

    selected_course = _provided(course_code) or graph_course
    selected_module = graph_module or _provided(module_number)

    if selected_course is None:
        raise GraphInputError("course code is required")
    if selected_module is None:
        raise GraphInputError("module number is required")
    return ModuleIdentity(selected_course, selected_module)


def extract_graph_facts(graph: Mapping[str, Any]) -> tuple[GraphFact, ...]:
    lesson_values = graph.get("facts")
    lesson_facts: list[GraphFact] = []
    used_lesson_ids: set[str] = set()
    used_statements: set[str] = set()
    if isinstance(lesson_values, list):
        for index, item in enumerate(lesson_values, start=1):
            if not isinstance(item, Mapping):
                continue
            statement = str(item.get("statement", "")).strip()
            statement_key = " ".join(statement.casefold().split())
            if not statement or statement_key in used_statements:
                continue
            fact_id = str(item.get("id") or f"f{index}").strip()
            if not fact_id:
                fact_id = f"f{index}"
            if fact_id in used_lesson_ids:
                fact_id = f"{fact_id}#{index}"
            slide_values = item.get("slides")
            slides: list[int] = []
            if isinstance(slide_values, list):
                for value in slide_values:
                    try:
                        number = int(value)
                    except (TypeError, ValueError):
                        continue
                    if number > 0 and number not in slides:
                        slides.append(number)
            topic_value = item.get("topic")
            topic = str(topic_value).strip() if topic_value is not None else None
            used_lesson_ids.add(fact_id)
            used_statements.add(statement_key)
            lesson_facts.append(
                GraphFact(
                    fact_id=fact_id,
                    statement=statement,
                    slides=tuple(sorted(slides)),
                    topic=topic or None,
                )
            )
    if lesson_facts:
        return tuple(lesson_facts)

    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise GraphInputError("knowledge graph must contain an edges list")

    facts: list[GraphFact] = []
    used_ids: set[str] = set()
    for index, edge in enumerate(edges, start=1):
        if not isinstance(edge, Mapping):
            continue
        parts = [str(edge.get(key, "")).strip() for key in ("subject", "relation", "object")]
        if not all(parts):
            continue
        fact_id = str(edge.get("id") or f"e{index}")
        if fact_id in used_ids:
            fact_id = f"{fact_id}#{index}"
        used_ids.add(fact_id)
        slide_numbers: set[int] = set()
        evidence = edge.get("evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, Mapping) or not isinstance(item.get("slides"), list):
                    continue
                for value in item["slides"]:
                    try:
                        number = int(value)
                    except (TypeError, ValueError):
                        continue
                    if number > 0:
                        slide_numbers.add(number)
        facts.append(
            GraphFact(
                fact_id=fact_id,
                statement=" | ".join(parts),
                slides=tuple(sorted(slide_numbers)),
            )
        )

    if not facts:
        raise GraphInputError("graph contains no usable relationship facts")
    return tuple(facts)
