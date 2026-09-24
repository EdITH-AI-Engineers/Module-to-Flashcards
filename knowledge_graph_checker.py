from __future__ import annotations

from copy import deepcopy
import json
import time
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

_SYSTEM_PROMPT = """You are a University Professor and a conservative knowledge-graph quality checker.
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

Remove any instance of mentioning the metadata of the module itself, such as the course code, module number, module title, or source file name. Do not remove any item that contains a substantive claim about the subject matter. Ensure that you do not rewrite or invent any facts, and do not change any IDs.

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


# The reply is only a short list of IDs. With short aliases 24 rows need well
# under 150 tokens, so hitting the cap means the model is running on (thinking
# or repeating), not that it needs more room. Keep the cap small so a bad call
# fails in seconds instead of minutes on a local model, allow one modest
# increase for a genuine long reply, and never split/multiply calls.
_BASE_MAX_TOKENS = 128
_TOKENS_PER_ROW = 6
_MAX_TOKENS_CAP = 768


def _initial_max_tokens(row_count: int) -> int:
    return min(_MAX_TOKENS_CAP, _BASE_MAX_TOKENS + _TOKENS_PER_ROW * row_count)


def _looks_truncated(exc: Exception) -> bool:
    return isinstance(exc, json.JSONDecodeError) or "truncat" in str(exc).lower()


def _review_batch(
    backend: ChatBackend,
    rows: Sequence[Mapping[str, object]],
    *,
    item_kind: str,
    metadata: Mapping[str, object],
    max_retries: int,
    progress: Callable[[str], None] | None = None,
) -> set[str]:
    # Send short aliases (F1, F2, ...) instead of the graph's own IDs so the
    # model has far fewer tokens to copy back. They are mapped back below.
    prefix = item_kind[:1].upper()
    alias_to_id: dict[str, str] = {}
    alias_rows: list[dict[str, object]] = []
    for number, row in enumerate(rows, start=1):
        alias = f"{prefix}{number}"
        alias_to_id[alias] = str(row["id"])
        alias_rows.append({**row, "id": alias})
    allowed_ids = set(alias_to_id)
    context = {
        key: metadata.get(key)
        for key in ("course_code", "module_number", "module_title", "source_file")
        if metadata.get(key) not in (None, "")
    }
    user = (
        f"Module context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"Candidate {item_kind} items:\n"
        f"{json.dumps(alias_rows, ensure_ascii=False)}\n\n"
        'Return {"remove_ids": [...]} containing only IDs that clearly meet '
        "the removal policy. Return an empty list when every item is useful. "
        "Output the JSON object only, with no explanation or reasoning.\n"
        "/no_think"
    )
    max_tokens = _initial_max_tokens(len(rows))
    last_error = "invalid checker response"
    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        if progress is not None:
            progress(
                f"  {item_kind} batch attempt {attempt}/{max_retries} "
                f"(max_tokens={max_tokens})..."
            )
        try:
            raw = backend.complete(
                _SYSTEM_PROMPT,
                user,
                max_tokens=max_tokens,
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
                raise ValueError(
                    f"remove_ids contains unknown IDs: {', '.join(unknown)}"
                )
            return {alias_to_id[item] for item in remove_ids}
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if progress is not None:
                progress(
                    f"  attempt {attempt} failed after "
                    f"{time.monotonic() - started:.0f}s: {last_error}"
                )
            if _looks_truncated(exc):
                max_tokens = min(_MAX_TOKENS_CAP, max_tokens * 2)
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
    on_batch_failure: str = "keep",
) -> dict[str, object]:
    """Use Qwen to prune graph noise without rewriting grounded content.

    ``on_batch_failure`` decides what happens when one batch cannot be reviewed
    after all retries: "keep" leaves that batch's items in the graph (the
    checker's rule is to keep when uncertain) and records the count in
    metadata; "raise" aborts with KnowledgeGraphCheckError.
    """
    if on_batch_failure not in ("keep", "raise"):
        raise ValueError('on_batch_failure must be "keep" or "raise"')
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    checked = deepcopy(dict(graph))
    metadata_value = checked.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}

    removed_positions: dict[str, set[int]] = {"facts": set(), "edges": set()}
    unreviewed = {"facts": 0, "edges": 0}
    for collection_name, item_kind in (("facts", "fact"), ("edges", "relationship")):
        values = checked.get(collection_name, [])
        if not isinstance(values, list):
            raise KnowledgeGraphCheckError(
                f"knowledge graph {collection_name} must be a list"
            )
        rows, positions = _candidate_rows(
            values, kind="fact" if collection_name == "facts" else "edge"
        )
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            if progress is not None:
                number = offset // batch_size + 1
                total = max(1, (len(rows) + batch_size - 1) // batch_size)
                progress(f"Knowledge graph {item_kind} review {number}/{total}...")
            try:
                removed = _review_batch(
                    backend,
                    batch,
                    item_kind=item_kind,
                    metadata=metadata,
                    max_retries=max_retries,
                    progress=progress,
                )
            except KnowledgeGraphCheckError as exc:
                if on_batch_failure == "raise":
                    raise
                unreviewed[collection_name] += len(batch)
                removed = set()
                if progress is not None:
                    progress(
                        f"WARNING: {exc}; keeping {len(batch)} {item_kind} "
                        "item(s) unreviewed."
                    )
            removed_positions[collection_name].update(
                positions[item] for item in removed
            )

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
    if (
        not original_facts
        and isinstance(original_edges, list)
        and original_edges
        and not checked["edges"]
    ):
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
                "unreviewed_facts": unreviewed["facts"],
                "unreviewed_edges": unreviewed["edges"],
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
