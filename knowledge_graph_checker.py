from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable, Mapping, Sequence

from flashcard_types import ChatBackend


class KnowledgeGraphCheckError(RuntimeError):
    """Raised when Qwen cannot return a safe graph-pruning decision."""


CHECKER_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "remove_ids": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["remove_ids"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You are a conservative knowledge-graph quality checker.
Review candidate items from one educational module and identify only items that
are clearly irrelevant or unusable for learning. This policy must work across
computer science, engineering, mathematics, sciences, humanities, business,
and multimedia arts.

Keep substantive definitions, principles, processes, formulas, constraints,
comparisons, examples, techniques, historical facts, and domain-specific
claims. Do not remove an item merely because it is specialized, unfamiliar,
short, or belongs to a creative rather than technical discipline.

Remove only clear presentation or extraction noise: course/module identifiers,
standalone page or section labels with no instructional claim, author/contact or
copyright lines, navigation directions, generic statements that merely say the
module introduces/covers/discusses a topic, corrupted fragments, or material
plainly unrelated to the module. When uncertain, keep the item.

Return JSON only. Copy IDs exactly. Never rewrite facts and never invent IDs."""


def _candidate_rows(
    items: Sequence[object],
    *,
    kind: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    positions: dict[str, int] = {}
    for index, value in enumerate(items, start=1):
        if not isinstance(value, Mapping):
            continue
        base = str(value.get("id") or f"{kind}-{index}").strip() or f"{kind}-{index}"
        reference = base
        suffix = 2
        while reference in positions:
            reference = f"{base}#{suffix}"
            suffix += 1
        positions[reference] = index - 1
        if kind == "fact":
            row = {
                "id": reference,
                "statement": str(value.get("statement", "")).strip(),
                "topic": str(value.get("topic", "")).strip(),
                "kind": str(value.get("kind", "")).strip(),
            }
        else:
            row = {
                "id": reference,
                "subject": str(value.get("subject", "")).strip(),
                "relation": str(value.get("relation", "")).strip(),
                "object": str(value.get("object", "")).strip(),
            }
        rows.append(row)
    return rows, positions


def _review_batch(
    backend: ChatBackend,
    rows: Sequence[Mapping[str, object]],
    *,
    item_kind: str,
    metadata: Mapping[str, object],
    max_retries: int,
) -> set[str]:
    allowed_ids = {str(row["id"]) for row in rows}
    context = {
        key: metadata.get(key)
        for key in ("course_code", "module_number", "module_title", "source_file")
        if metadata.get(key) not in (None, "")
    }
    user = (
        f"Module context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"Candidate {item_kind} items:\n"
        f"{json.dumps(list(rows), ensure_ascii=False)}\n\n"
        "Return {\"remove_ids\": [...]} containing only IDs that clearly meet "
        "the removal policy. Return an empty list when every item is useful."
    )
    last_error = "invalid checker response"
    for attempt in range(1, max_retries + 1):
        try:
            raw = backend.complete(
                _SYSTEM_PROMPT,
                user,
                max_tokens=384,
                schema=CHECKER_SCHEMA,
            )
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {"remove_ids"}:
                raise ValueError("response must contain only remove_ids")
            remove_ids = value["remove_ids"]
            if not isinstance(remove_ids, list) or any(
                not isinstance(item, str) for item in remove_ids
            ):
                raise ValueError("remove_ids must be a list of strings")
            duplicates = len(remove_ids) != len(set(remove_ids))
            unknown = sorted(set(remove_ids) - allowed_ids)
            if duplicates:
                raise ValueError("remove_ids contains duplicate IDs")
            if unknown:
                raise ValueError(f"remove_ids contains unknown IDs: {', '.join(unknown)}")
            return set(remove_ids)
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
    raise KnowledgeGraphCheckError(
        f"Qwen {item_kind} review failed after {max_retries} attempts: {last_error}"
    )


def check_knowledge_graph(
    graph: Mapping[str, object],
    backend: ChatBackend,
    *,
    max_retries: int = 3,
    batch_size: int = 24,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Use Qwen to prune graph noise without rewriting grounded content."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    checked = deepcopy(dict(graph))
    metadata_value = checked.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}

    removed_positions: dict[str, set[int]] = {"facts": set(), "edges": set()}
    for collection_name, item_kind in (("facts", "fact"), ("edges", "relationship")):
        values = checked.get(collection_name, [])
        if not isinstance(values, list):
            raise KnowledgeGraphCheckError(
                f"knowledge graph {collection_name} must be a list"
            )
        rows, positions = _candidate_rows(values, kind="fact" if collection_name == "facts" else "edge")
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            if progress is not None:
                number = offset // batch_size + 1
                total = max(1, (len(rows) + batch_size - 1) // batch_size)
                progress(f"Knowledge graph {item_kind} review {number}/{total}...")
            removed = _review_batch(
                backend,
                batch,
                item_kind=item_kind,
                metadata=metadata,
                max_retries=max_retries,
            )
            removed_positions[collection_name].update(positions[item] for item in removed)

        checked[collection_name] = [
            value
            for index, value in enumerate(values)
            if index not in removed_positions[collection_name]
        ]

    original_facts = graph.get("facts", [])
    if isinstance(original_facts, list) and original_facts and not checked["facts"]:
        raise KnowledgeGraphCheckError(
            "Qwen marked every lesson fact irrelevant; refusing to save an empty graph"
        )
    original_edges = graph.get("edges", [])
    if not original_facts and isinstance(original_edges, list) and original_edges and not checked["edges"]:
        raise KnowledgeGraphCheckError(
            "Qwen marked every relationship irrelevant; refusing to save an empty graph"
        )

    referenced_nodes = {
        str(edge.get(key))
        for edge in checked["edges"]
        if isinstance(edge, Mapping)
        for key in ("source", "target")
        if edge.get(key) is not None
    }
    nodes = checked.get("nodes", [])
    if not isinstance(nodes, list):
        raise KnowledgeGraphCheckError("knowledge graph nodes must be a list")
    checked["nodes"] = [
        node
        for node in nodes
        if isinstance(node, Mapping) and str(node.get("id")) in referenced_nodes
    ]

    metadata.update(
        {
            "node_count": len(checked["nodes"]),
            "edge_count": len(checked["edges"]),
            "fact_count": len(checked["facts"]),
            "graph_checker": {
                "model": "Qwen3-8B-Q5_K_M",
                "status": "checked",
                "removed_facts": len(removed_positions["facts"]),
                "removed_edges": len(removed_positions["edges"]),
            },
        }
    )
    checked["metadata"] = metadata
    return checked


def is_checked_graph(graph: Mapping[str, object]) -> bool:
    metadata = graph.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    checker = metadata.get("graph_checker")
    return isinstance(checker, Mapping) and checker.get("status") == "checked"
