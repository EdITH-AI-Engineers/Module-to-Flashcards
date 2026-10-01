import pytest

from worker_budget import choose_cluster_workers, parse_cluster_workers


GIB = 1024**3


@pytest.mark.parametrize(
    ("full_gpu", "before", "after", "reason_fragment"),
    [
        (False, (20 * GIB, 24 * GIB), (18 * GIB, 24 * GIB), "GPU"),
        (True, None, (18 * GIB, 24 * GIB), "measurement"),
        (True, (20 * GIB, 24 * GIB), None, "measurement"),
        (True, (18 * GIB, 24 * GIB), (18 * GIB, 24 * GIB), "footprint"),
        (True, (17 * GIB, 24 * GIB), (18 * GIB, 24 * GIB), "footprint"),
    ],
)
def test_auto_workers_falls_back_when_gpu_budget_is_unreliable(
    full_gpu, before, after, reason_fragment
):
    count, reason = choose_cluster_workers(
        "auto", full_gpu=full_gpu, before=before, after=after
    )
    assert count == 1
    assert reason_fragment.casefold() in reason.casefold()


def test_auto_workers_can_select_more_than_five_from_measured_vram():
    count, reason = choose_cluster_workers(
        "auto",
        full_gpu=True,
        before=(20 * GIB, 24 * GIB),
        after=(18 * GIB, 24 * GIB),
    )
    assert count == 7
    assert "VRAM" in reason


def test_manual_twelve_workers_bypasses_gpu_measurement():
    assert choose_cluster_workers(
        12, full_gpu=False, before=None, after=None
    )[0] == 12
    assert choose_cluster_workers(
        12, full_gpu=False, before=None, after=None, cluster_count=8
    )[0] == 8


@pytest.mark.parametrize("raw", ["0", "21", "five", "", "-1"])
def test_worker_setting_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="cluster workers"):
        parse_cluster_workers(raw)


def test_worker_setting_accepts_auto_and_counts_up_to_twenty():
    assert parse_cluster_workers("auto") == "auto"
    assert parse_cluster_workers("12") == 12
    assert parse_cluster_workers("20") == 20
