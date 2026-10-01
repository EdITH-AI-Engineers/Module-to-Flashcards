"""Summaries for task-tagged local Qwen calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from statistics import mean


def _token_summary(calls: Sequence[Mapping[str, object]], field: str) -> dict[str, object]:
    values = sorted(
        value
        for call in calls
        if type(value := call.get(field)) is int
    )
    if not values:
        return {"mean": None, "p95": None}
    return {
        "mean": mean(values),
        "p95": values[ceil(0.95 * len(values)) - 1],
    }


def _task_summary(
    calls: Sequence[Mapping[str, object]], counts: Mapping[str, int]
) -> dict[str, object]:
    first_attempts = int(counts.get("first_attempts", 0))
    first_passes = int(counts.get("first_attempt_passes", 0))
    return {
        "calls": len(calls),
        "prompt_tokens": _token_summary(calls, "prompt_tokens"),
        "completion_tokens": _token_summary(calls, "completion_tokens"),
        "length_finishes": sum(call.get("finish_reason") == "length" for call in calls),
        "context_errors": sum(
            call.get("exception") == "ContextWindowExceededError" for call in calls
        ),
        "first_attempts": first_attempts,
        "first_attempt_passes": first_passes,
        "first_attempt_pass_rate": (
            first_passes / first_attempts if first_attempts else None
        ),
        "retry_count": int(counts.get("retries", 0)),
    }


def summarize_metrics(
    calls: Sequence[Mapping[str, object]],
    task_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    """Summarize recorded calls without including generated card content."""

    by_task: dict[str, list[Mapping[str, object]]] = {}
    for call in calls:
        by_task.setdefault(str(call.get("task") or "unspecified"), []).append(call)
    tasks = {
        task: _task_summary(by_task.get(task, ()), task_counts.get(task, {}))
        for task in sorted(set(by_task) | set(task_counts))
    }
    aggregate = {
        field: sum(int(counts.get(field, 0)) for counts in task_counts.values())
        for field in ("first_attempts", "first_attempt_passes", "retries")
    }
    return {
        "tasks": tasks,
        "overall": _task_summary(calls, aggregate),
    }
