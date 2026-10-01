"""Conservative concurrency budget for independent Qwen cluster contexts."""

from flashcard_contract import CLUSTERS_PER_MODULE


GIB = 1024**3


def parse_cluster_workers(value: str) -> int | str:
    """Parse an automatic policy or an explicit count of cluster jobs."""
    if value == "auto":
        return value
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cluster workers must be 'auto' or an integer from 1 to 20") from exc
    if not 1 <= count <= CLUSTERS_PER_MODULE:
        raise ValueError("cluster workers must be 'auto' or an integer from 1 to 20")
    return count


def choose_cluster_workers(
    setting: int | str,
    *,
    full_gpu: bool,
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
    cluster_count: int = CLUSTERS_PER_MODULE,
) -> tuple[int, str]:
    """Return the safe count and an explanation for progress output."""
    parsed = parse_cluster_workers(str(setting))
    if cluster_count < 1:
        raise ValueError("cluster count must be positive")
    if isinstance(parsed, int):
        count = min(parsed, cluster_count)
        return count, f"manual setting selected {count} cluster workers"
    if not full_gpu:
        return 1, "GPU full offload is unavailable; using one cluster worker"
    if before is None or after is None or after[1] <= 0:
        return 1, "GPU memory measurement is unavailable; using one cluster worker"
    footprint = before[0] - after[0]
    if footprint <= 0:
        return 1, "Qwen context GPU footprint is not measurable; using one cluster worker"
    reserve = max(GIB, (after[1] + 9) // 10)
    per_extra = (footprint * 5 + 3) // 4
    count = min(cluster_count, 1 + max(0, after[0] - reserve) // per_extra)
    return count, f"GPU VRAM budget selected {count} cluster workers"
