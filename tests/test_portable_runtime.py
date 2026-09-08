import os
import sys
from types import SimpleNamespace

from portable_runtime import (
    configure_offline_environment,
    detect_acceleration,
)


def test_offline_environment_disables_hub_access_and_isolates_cache(
    tmp_path, monkeypatch
):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)

    configure_offline_environment(tmp_path / "model-cache")

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == str(tmp_path / "model-cache")


def test_acceleration_detection_reports_both_model_backends(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5070",
        )
    )
    fake_llama = SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)

    status = detect_acceleration()

    assert status.torch_cuda is True
    assert status.llama_gpu_offload is True
    assert status.torch_device_name == "NVIDIA GeForce RTX 5070"


def test_acceleration_detection_reports_cpu_only_without_querying_device_name(
    monkeypatch,
):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda index: (_ for _ in ()).throw(
                AssertionError("CPU mode must not query a CUDA device")
            ),
        )
    )
    fake_llama = SimpleNamespace(llama_supports_gpu_offload=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama)

    status = detect_acceleration()

    assert status.torch_cuda is False
    assert status.llama_gpu_offload is False
    assert status.torch_device_name is None
