from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from local_qwen import (
    MODEL_FILENAME,
    MODEL_REPO,
    MODEL_REVISION,
    LocalQwenBackend,
    ensure_model,
    thread_budget_for_workers,
)
from flashcard_types import CompletionTruncatedError, ContextWindowExceededError


def test_ensure_model_downloads_exact_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"gguf")
        return str(target)

    monkeypatch.setattr("local_qwen.hf_hub_download", fake_download)

    path = ensure_model(tmp_path)

    assert path.name == "Qwen3-8B-Q5_K_M.gguf"
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-8B-GGUF",
            "revision": "4f02e7c52b572082828edf5058a87e2e7dc3e4d5",
            "filename": "Qwen3-8B-Q5_K_M.gguf",
            "local_dir": str(tmp_path),
        }
    ]
    assert MODEL_REPO == "Qwen/Qwen3-8B-GGUF"
    assert MODEL_REVISION == "4f02e7c52b572082828edf5058a87e2e7dc3e4d5"


def test_existing_model_is_reused(tmp_path, monkeypatch):
    path = tmp_path / MODEL_FILENAME
    path.write_bytes(b"gguf")

    def fail_download(**kwargs):
        raise AssertionError("existing model should not be downloaded")

    monkeypatch.setattr("local_qwen.hf_hub_download", fail_download)

    assert ensure_model(tmp_path) == path


def test_missing_bundled_model_never_downloads_when_download_is_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "local_qwen.hf_hub_download",
        lambda **kwargs: pytest.fail("portable mode must never download a model"),
    )

    with pytest.raises(FileNotFoundError, match="bundled Qwen model"):
        ensure_model(tmp_path, allow_download=False)


def test_backend_disables_thinking_and_varies_deterministic_call_seeds():
    calls = []

    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": '{"cards": []}'}}]}

    backend = LocalQwenBackend.__new__(LocalQwenBackend)
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
                {"role": "user", "content": "USER\n\n/no_think"},
            ],
            "temperature": 0.2,
            "seed": 42,
            "max_tokens": 512,
            "response_format": {"type": "json_object", "schema": schema},
        },
        {
            "messages": [
                {"role": "system", "content": "SYSTEM"},
                {"role": "user", "content": "RETRY\n\n/no_think"},
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

    backend = LocalQwenBackend.__new__(LocalQwenBackend)
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

    backend = LocalQwenBackend.__new__(LocalQwenBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    with pytest.raises(CompletionTruncatedError) as captured:
        backend.complete("SYSTEM", "USER", max_tokens=2048)

    assert captured.value.partial_content == '{"cards": ['
    assert captured.value.prompt_tokens == 6400
    assert captured.value.completion_tokens == 1792


def test_backend_translates_context_window_overflow():
    class FakeLlama:
        def create_chat_completion(self, **kwargs):
            raise ValueError("Requested tokens (8741) exceed context window of 8192")

    backend = LocalQwenBackend.__new__(LocalQwenBackend)
    backend._llm = FakeLlama()
    backend._temperature = 0.2
    backend._seed = 42
    backend._completion_index = 0

    with pytest.raises(ContextWindowExceededError, match="increase --n-ctx"):
        backend.complete("SYSTEM", "USER", max_tokens=1536)


def test_backend_defaults_to_12k_context_and_can_close(monkeypatch, tmp_path):
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    backend = LocalQwenBackend(tmp_path / "model.gguf")

    assert captured["n_ctx"] == 12288
    assert backend.context_window == 12288
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
    backend = LocalQwenBackend(path, n_ctx=4096, n_gpu_layers=12, seed=42)

    forked = backend.fork(2)

    assert calls == [
        {"model_path": str(path), "n_ctx": 4096, "n_gpu_layers": 12,
         "seed": 42, "verbose": False},
        {"model_path": str(path), "n_ctx": 4096, "n_gpu_layers": 12,
         "seed": 200042, "verbose": False},
    ]
    assert forked is not backend
    forked.close()
    assert forked._llm.closed is True
    assert backend._llm.closed is False


def test_parallel_model_contexts_share_the_available_cpu_threads(monkeypatch, tmp_path):
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setattr("local_qwen.os.cpu_count", lambda: 16)

    budget = thread_budget_for_workers(5)
    backend = LocalQwenBackend(tmp_path / "model.gguf", n_threads=budget)
    backend.fork(1)

    assert budget == 3
    assert calls[0]["n_threads"] == 3
    assert calls[1]["n_threads"] == 3
    assert thread_budget_for_workers(1) is None
