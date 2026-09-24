from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import Callable, Mapping, Sequence

from flashcard_types import ChatBackend, CompletionTruncatedError
from structured_module import is_unresolved_question_statement


class KnowledgeGraphCheckError(RuntimeError):
    """Raised when Qwen cannot return a safe graph-pruning decision."""


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

A question or unresolved alternative is not a factual assertion. Remove an
item that merely asks a question or presents an unanswered choice. If another
candidate explicitly answers it, keep the answer-bearing item as the usable
fact. A compound item may be kept when it includes an explicit answer after
the question. Never infer an answer, recommendation, or true/false value from
the wording of the question itself.

Remove any instance of mentioning the metadata of the module itself, such as the course code, module number, module title, or source file name. Do not remove any item that contains a substantive claim about the subject matter. Ensure that you do not rewrite or invent any facts.

Return JSON only. Classify every item in its given order. Never rewrite facts."""


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


# One boolean is returned for every row. The schema fixes the exact array
# length, preventing Qwen from looping over an unconstrained list until it hits
# max_tokens. The budget still leaves ample room for JSON punctuation and
# tokenization differences.
_BASE_MAX_TOKENS = 64
_TOKENS_PER_ROW = 8
_MAX_TOKENS_CAP = 512


def _initial_max_tokens(row_count: int) -> int:
    return min(_MAX_TOKENS_CAP, _BASE_MAX_TOKENS + _TOKENS_PER_ROW * row_count)


def _decision_schema(row_count: int) -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "remove": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": row_count,
                "maxItems": row_count,
            }
        },
        "required": ["remove"],
        "additionalProperties": False,
    }


def _review_batch(
    backend: ChatBackend,
    rows: Sequence[Mapping[str, object]],
    *,
    item_kind: str,
    metadata: Mapping[str, object],
    max_retries: int,
    progress: Callable[[str], None] | None = None,
) -> set[str]:
    context = {
        key: metadata.get(key)
        for key in ("course_code", "module_number", "module_title", "source_file")
        if metadata.get(key) not in (None, "")
    }
    user = (
        f"Module context:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"Candidate {item_kind} items:\n"
        f"{json.dumps(list(rows), ensure_ascii=False)}\n\n"
        f'Return {{"remove": [...]}} with exactly {len(rows)} booleans in '
        "the same order as the candidate items. Use true only when that item "
        "clearly meets the removal policy; otherwise use false. Output the "
        "JSON object only, with no explanation or reasoning."
    )
    schema = _decision_schema(len(rows))
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
                schema=schema,
            )
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {"remove"}:
                raise ValueError("response must contain only remove")
            decisions = value["remove"]
            if not isinstance(decisions, list) or any(
                not isinstance(item, bool) for item in decisions
            ):
                raise ValueError("remove must be a list of booleans")
            if len(decisions) != len(rows):
                raise ValueError(
                    f"remove must contain exactly {len(rows)} decisions"
                )
            return {
                str(row["id"])
                for row, remove in zip(rows, decisions)
                if remove
            }
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if progress is not None:
                progress(
                    f"  attempt {attempt} failed after "
                    f"{time.monotonic() - started:.0f}s: {last_error}"
                )
            if isinstance(exc, (CompletionTruncatedError, json.JSONDecodeError)):
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
        if collection_name == "facts":
            unresolved_ids = {
                str(row["id"])
                for row in rows
                if is_unresolved_question_statement(row.get("statement", ""))
            }
            removed_positions[collection_name].update(
                positions[item] for item in unresolved_ids
            )
            rows = [row for row in rows if str(row["id"]) not in unresolved_ids]
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
