from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from huggingface_hub import hf_hub_download

from flashcard_types import CompletionTruncatedError, ContextWindowExceededError
from worker_budget import choose_cluster_workers


MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "4f02e7c52b572082828edf5058a87e2e7dc3e4d5"
MODEL_FILENAME = "Qwen3-8B-Q5_K_M.gguf"
DEFAULT_N_CTX = 12288
QWEN_NON_THINKING_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.05,
}
_LOGGER = logging.getLogger(__name__)


def thread_budget_for_workers(workers: int) -> int | None:
    """Avoid multiplying CPU inference threads across model instances."""
    if workers <= 1:
        return None
    return max(1, (os.cpu_count() or 1) // workers)


def ensure_model(model_dir: Path, *, allow_download: bool = True) -> Path:
    """Return the exact local Q5 model path, downloading it when absent."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / MODEL_FILENAME
    if target.is_file():
        return target

    if not allow_download:
        raise FileNotFoundError(f"bundled Qwen model not found: {target}")

    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        filename=MODEL_FILENAME,
        local_dir=str(model_dir),
    )
    return Path(downloaded)


class LocalQwenBackend:
    supports_task_metrics = True

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        temperature: float = 0.2,
        seed: int = 42,
        hard_no_think: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed; run "
                "python -m pip install -r requirements.txt"
            ) from exc

        self._model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        memory_reader = None
        if n_gpu_layers == -1:
            try:
                from llama_cpp import llama_supports_gpu_offload

                if llama_supports_gpu_offload():
                    import torch

                    if torch.cuda.is_available():
                        memory_reader = lambda: torch.cuda.mem_get_info(0)
            except (ImportError, AttributeError, RuntimeError):
                pass

        def read_memory() -> tuple[int, int] | None:
            if memory_reader is None:
                return None
            try:
                free, total = memory_reader()
                return int(free), int(total)
            except (RuntimeError, OSError, ValueError, TypeError):
                return None

        before = read_memory()
        llama_kwargs = dict(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        if n_threads is not None:
            llama_kwargs["n_threads"] = n_threads
        self._llm = Llama(**llama_kwargs)
        after = read_memory()
        (
            self.auto_cluster_workers,
            self.auto_cluster_workers_reason,
        ) = choose_cluster_workers(
            "auto", full_gpu=memory_reader is not None, before=before, after=after
        )
        self._fork_n_threads = (
            n_threads
            if n_threads is not None
            else thread_budget_for_workers(self.auto_cluster_workers)
        )
        self._temperature = temperature
        self._hard_no_think = hard_no_think
        self._top_p = top_p
        self._top_k = top_k
        self._repeat_penalty = repeat_penalty
        self._presence_penalty = presence_penalty
        self._seed = seed
        self._completion_index = 0
        self._context_window = int(n_ctx)
        self._metric_sink: Callable[[dict[str, object]], None] | None = None

    def fork(self, worker_index: int) -> LocalQwenBackend:
        """Load an independent inference context for one cluster worker."""
        if worker_index < 1:
            raise ValueError("worker_index must be positive")
        forked = LocalQwenBackend(
            self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._fork_n_threads,
            temperature=self._temperature,
            seed=self._seed + 100_000 * worker_index,
            hard_no_think=self._hard_no_think,
            top_p=self._top_p,
            top_k=self._top_k,
            repeat_penalty=self._repeat_penalty,
            presence_penalty=self._presence_penalty,
        )
        forked._metric_sink = self._metric_sink
        return forked

    def set_metric_sink(
        self, sink: Callable[[dict[str, object]], None] | None
    ) -> None:
        """Register a metrics collector without changing generation settings."""

        self._metric_sink = sink

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_prompt_tokens(self, system: str, user: str) -> int:
        """Count the exact chat-template tokens used by the local backend."""
        if getattr(self, "_hard_no_think", False):
            return len(
                self._llm.tokenize(
                    self._hard_no_think_prompt(system, user).encode("utf-8"),
                    add_bos=False,
                    special=True,
                )
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{user}\n\n/no_think"},
        ]
        metadata = getattr(self._llm, "metadata", {})
        template = (
            metadata.get("tokenizer.chat_template")
            if isinstance(metadata, Mapping)
            else None
        )
        if isinstance(template, str) and template:
            from llama_cpp import llama_chat_format

            eos_token_id = self._llm.token_eos()
            bos_token_id = self._llm.token_bos()
            eos_token = (
                self._llm._model.token_get_text(eos_token_id)
                if eos_token_id != -1
                else ""
            )
            bos_token = (
                self._llm._model.token_get_text(bos_token_id)
                if bos_token_id != -1
                else ""
            )
            formatter = llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
                stop_token_ids=[eos_token_id],
            )
            rendered = formatter(messages=messages)
            return len(
                self._llm.tokenize(
                    rendered.prompt.encode("utf-8"),
                    add_bos=not rendered.added_special,
                    special=True,
                )
            )

        # The pinned Qwen GGUF contains a chat template. This fallback keeps
        # alternate/test models conservative when that metadata is absent.
        raw = f"System:\n{system}\nUser:\n{user}\n\n/no_think\nAssistant:\n"
        return len(
            self._llm.tokenize(raw.encode("utf-8"), add_bos=True, special=True)
        ) + 16

    @staticmethod
    def _hard_no_think_prompt(system: str, user: str) -> str:
        """Render exactly the ChatML bytes passed to create_completion."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}\n\n/no_think<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        schema: Mapping[str, object] | None = None,
        task: str | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> str:
        call_seed = self._seed + self._completion_index
        self._completion_index += 1
        started = perf_counter()
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        finish_reason: str | None = None
        exception_name: str | None = None
        contains_think: bool | None = None
        output_chars: int | None = None
        response_format: dict[str, object] = {"type": "json_object"}
        if schema is not None:
            response_format["schema"] = dict(schema)
        sampling = {
            "top_p": top_p if top_p is not None else getattr(self, "_top_p", None),
            "top_k": top_k if top_k is not None else getattr(self, "_top_k", None),
            "repeat_penalty": (
                repeat_penalty if repeat_penalty is not None
                else getattr(self, "_repeat_penalty", None)
            ),
            "presence_penalty": (
                presence_penalty if presence_penalty is not None
                else getattr(self, "_presence_penalty", None)
            ),
        }
        sampling = {key: value for key, value in sampling.items() if value is not None}
        try:
            if getattr(self, "_hard_no_think", False):
                from llama_cpp import LlamaGrammar

                grammar = LlamaGrammar.from_json_schema(
                    json.dumps(dict(schema) if schema is not None else {"type": "object"})
                )
                response: Any = self._llm.create_completion(
                    prompt=self._hard_no_think_prompt(system, user),
                    temperature=self._temperature,
                    seed=call_seed,
                    max_tokens=max_tokens,
                    grammar=grammar,
                    stop=["<|im_end|>"],
                    **sampling,
                )
            else:
                response = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"{user}\n\n/no_think"},
                    ],
                    temperature=self._temperature,
                    seed=call_seed,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **sampling,
                )
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            if isinstance(usage, dict):
                recorded_prompt = usage.get("prompt_tokens")
                recorded_completion = usage.get("completion_tokens")
                prompt_tokens = recorded_prompt if isinstance(recorded_prompt, int) else None
                completion_tokens = (
                    recorded_completion if isinstance(recorded_completion, int) else None
                )
            try:
                choice = response["choices"][0]
                content = (
                    choice["text"]
                    if getattr(self, "_hard_no_think", False)
                    else choice["message"]["content"]
                )
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("Qwen returned an unexpected response shape") from exc
            finish_reason = choice.get("finish_reason")
            contains_think = isinstance(content, str) and "<think>" in content
            output_chars = len(content) if isinstance(content, str) else None
            if finish_reason == "length":
                raise CompletionTruncatedError(
                    "Qwen response was truncated before it completed the requested JSON",
                    partial_content=content if isinstance(content, str) else "",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Qwen returned empty assistant content")
            return content.strip()
        except ValueError as exc:
            if "exceed context window" in str(exc):
                if prompt_tokens is None:
                    try:
                        prompt_tokens = self.count_prompt_tokens(system, user)
                    except Exception:
                        pass
                exception_name = "ContextWindowExceededError"
                finish_reason = "context_error"
                raise ContextWindowExceededError(
                    "Qwen prompt exceeded the configured context window; "
                    "reduce the prompt or increase --n-ctx"
                ) from exc
            exception_name = type(exc).__name__
            raise
        except Exception as exc:
            exception_name = type(exc).__name__
            raise
        finally:
            metric: dict[str, object] = {
                "task": task or "unspecified",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
                "elapsed_seconds": perf_counter() - started,
                "exception": exception_name,
                "contains_think": contains_think,
                "output_chars": output_chars,
            }
            _LOGGER.info("Qwen completion: %s", json.dumps(metric, sort_keys=True))
            sink = getattr(self, "_metric_sink", None)
            if sink is not None:
                try:
                    sink(metric)
                except Exception:
                    _LOGGER.exception("Could not record Qwen completion metric")
