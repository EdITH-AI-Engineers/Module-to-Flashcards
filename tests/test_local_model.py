from pathlib import Path
import inspect
import sys
from types import SimpleNamespace

import pytest

from local_model import (
    MODEL_FILENAME,
    MODEL_REPO,
    MODEL_REVISION,
    LocalModelBackend,
    ensure_model,
    thread_budget_for_workers,
)
from flashcard_types import CompletionTruncatedError, ContextWindowExceededError


def test_ensure_model_downloads_exact_ministral_3b_instruct_checkpoint(
    tmp_path, monkeypatch
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"gguf")
        return str(target)

    monkeypatch.setattr("local_model.hf_hub_download", fake_download)

    path = ensure_model(tmp_path)

    assert path.name == "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
    assert calls == [
        {
            "repo_id": "mistralai/Ministral-3-3B-Instruct-2512-GGUF",
            "revision": "eb599d408350ea2bb60452cb86be7c7b2fc28227",
            "filename": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
            "local_dir": str(tmp_path),
        }
    ]
    assert MODEL_REPO == "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
    assert MODEL_REVISION == "eb599d408350ea2bb60452cb86be7c7b2fc28227"


def test_existing_model_is_reused(tmp_path, monkeypatch):
    path = tmp_path / MODEL_FILENAME
    path.write_bytes(b"gguf")

    def fail_download(**kwargs):
        raise AssertionError("existing model should not be downloaded")

    monkeypatch.setattr("local_model.hf_hub_download", fail_download)

    assert ensure_model(tmp_path) == path


def test_missing_bundled_model_never_downloads_when_download_is_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "local_model.hf_hub_download",
        lambda **kwargs: pytest.fail("portable mode must never download a model"),
    )

    with pytest.raises(FileNotFoundError, match="bundled local model"):
        ensure_model(tmp_path, allow_download=False)


def test_backend_uses_unmodified_user_prompts_and_varies_deterministic_call_seeds():
    calls = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"cards": []}'}}]}

    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    schema = {
        "type": "object",
        "properties": {"cards": {"type": "array"}},
        "required": ["cards"],
        "additionalProperties": False,
    }
    first = backend.complete("SYSTEM", "USER", max_tokens=512, schema=schema)
    second = backend.complete("SYSTEM", "RETRY", max_tokens=256)

    assert first == '{"cards": []}'
    assert second == '{"cards": []}'
    assert calls == [
        {
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "USER"},
            ],
            "temperature": 0.2,
            "seed": 42,
            "max_tokens": 512,
            "response_format": {"type": "json_object", "schema": schema},
        },
        {
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "RETRY"},
            ],
            "temperature": 0.2,
            "seed": 43,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        },
    ]


def test_backend_rejects_empty_assistant_content():
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "  "}}]}

    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    try:
        backend.complete("SYSTEM", "USER", max_tokens=32)
    except RuntimeError as exc:
        assert "empty assistant content" in str(exc)
    else:
        raise AssertionError("empty model content should fail")


def test_backend_reports_length_limited_response_as_truncation():
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {"content": '{"cards": ['},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 6400, "completion_tokens": 1792},
            }

    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    with pytest.raises(CompletionTruncatedError) as captured:
        backend.complete("SYSTEM", "USER", max_tokens=2048)

    assert captured.value.partial_content == '{"cards": ['
    assert captured.value.prompt_tokens == 6400
    assert captured.value.completion_tokens == 1792


def test_backend_records_task_metrics_for_success_truncation_and_context_error():
    responses = iter(
        (
            {
                "choices": [{"message": {"content": '{"cards": []}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 15},
            },
            {
                "choices": [{"message": {"content": '{"cards": ['}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 130, "completion_tokens": 64},
            },
        )
    )

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            if kwargs["max_tokens"] == 3:
                raise ValueError("Requested tokens exceed context window")
            return next(responses)

    events = []
    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0
    backend._metric_sink = events.append

    assert backend.complete("S", "U", max_tokens=16, task="cluster") == '{"cards": []}'
    with pytest.raises(CompletionTruncatedError):
        backend.complete("S", "U", max_tokens=64, task="retry")
    with pytest.raises(ContextWindowExceededError):
        backend.complete("S", "U", max_tokens=3, task="concept_plan")

    assert [(item["task"], item["finish_reason"], item["max_tokens"]) for item in events] == [
        ("cluster", "stop", 16),
        ("retry", "length", 64),
        ("concept_plan", "context_error", 3),
    ]
    assert events[0]["prompt_tokens"] == 120
    assert events[0]["output_chars"] == len('{"cards": []}')
    assert events[0]["contains_think"] is False
    assert events[1]["completion_tokens"] == 64
    assert events[2]["exception"] == "ContextWindowExceededError"
    assert all(item["elapsed_seconds"] >= 0 for item in events)
    assert all("content" not in item for item in events)


def test_backend_has_no_qwen_only_hard_no_think_mode():
    assert "hard_no_think" not in inspect.signature(LocalModelBackend).parameters


def test_backend_rejects_runtime_before_integer_tokenizer_score_support(
    monkeypatch, tmp_path
):
    class FakeLlama:
        def __init__(self, **kwargs):
            raise AssertionError("an incompatible runtime must fail before model loading")

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(__version__="0.3.35", Llama=FakeLlama),
    )

    with pytest.raises(RuntimeError, match=r"llama-cpp-python>=0\.3\.36.*0\.3\.35"):
        LocalModelBackend(tmp_path / "model.gguf")


def test_optional_sampling_defaults_are_omitted_and_call_overrides_apply():
    calls = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "{}"}}]}

    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0
    backend._top_p = 0.8
    backend._top_k = 20
    backend._repeat_penalty = 1.05
    backend._presence_penalty = None

    backend.complete("S", "U", max_tokens=100, top_p=0.9, presence_penalty=0.2)

    assert calls[0]["temperature"] == 0.2
    assert calls[0]["top_p"] == 0.9
    assert calls[0]["top_k"] == 20
    assert calls[0]["repeat_penalty"] == 1.05
    assert calls[0]["presence_penalty"] == 0.2


def test_backend_translates_context_window_overflow():
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            raise ValueError("Requested tokens (8741) exceed context window of 8192")

    backend = LocalModelBackend.__new__(LocalModelBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    with pytest.raises(ContextWindowExceededError, match="increase --n-ctx"):
        backend.complete("SYSTEM", "USER", max_tokens=1536)


def test_backend_defaults_to_8k_context_and_low_temperature(monkeypatch, tmp_path):
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    backend = LocalModelBackend(tmp_path / "model.gguf")

    assert captured["n_ctx"] == 8192
    assert backend.context_window == 8192
    assert backend._temperature == 0.05
    backend.close()
    assert backend._llm.closed is True


def test_fork_creates_separate_model_context_with_worker_seed(monkeypatch, tmp_path):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    path = tmp_path / "model.gguf"
    metrics = []
    sink = metrics.append
    backend = LocalModelBackend(path, n_ctx=4096, n_gpu_layers=12, seed=42)
    backend.set_metric_sink(sink)

    forked = backend.fork(2)

    assert calls == [
        {"model_path": str(path), "n_ctx": 4096, "n_gpu_layers": 12,
         "seed": 42, "verbose": False},
        {"model_path": str(path), "n_ctx": 4096, "n_gpu_layers": 12,
         "seed": 200042, "verbose": False},
    ]
    assert forked is not backend
    assert forked._metric_sink is sink
    forked.close()
    assert forked._llm.closed is True
    assert backend._llm.closed is False


def test_parallel_model_contexts_share_the_available_cpu_threads(monkeypatch, tmp_path):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("local_model.os.cpu_count", lambda: 16)

    budget = thread_budget_for_workers(5)
    backend = LocalModelBackend(tmp_path / "model.gguf", n_threads=budget)
    backend.fork(1)

    assert budget == 3
    assert calls[0]["n_threads"] == 3
    assert calls[1]["n_threads"] == 3
    assert thread_budget_for_workers(1) is None


def test_full_gpu_load_measures_context_and_selects_more_than_five(monkeypatch, tmp_path):
    gib = 1024**3
    readings = iter(((20 * gib, 24 * gib), (18 * gib, 24 * gib)))
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda device: next(readings),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setitem(
        sys.modules, "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, llama_supports_gpu_offload=lambda: True),
    )

    backend = LocalModelBackend(tmp_path / "model.gguf")

    assert backend.auto_cluster_workers == 7
    assert "VRAM" in backend.auto_cluster_workers_reason
    assert "n_threads" not in calls[0]


def test_auto_workers_without_cuda_uses_one_and_does_not_probe_vram(
    monkeypatch, tmp_path
):
    class FakeLlama:
        def __init__(self, **kwargs):
            pass

    fake_cuda = SimpleNamespace(
        is_available=lambda: False,
        mem_get_info=lambda device: pytest.fail("CPU run must not probe VRAM"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setitem(
        sys.modules, "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, llama_supports_gpu_offload=lambda: True),
    )

    backend = LocalModelBackend(tmp_path / "model.gguf")

    assert backend.auto_cluster_workers == 1
    assert "GPU" in backend.auto_cluster_workers_reason
