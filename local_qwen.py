from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from huggingface_hub import hf_hub_download

from flashcard_types import CompletionTruncatedError, ContextWindowExceededError


MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "4f02e7c52b572082828edf5058a87e2e7dc3e4d5"
MODEL_FILENAME = "Qwen3-8B-Q5_K_M.gguf"
DEFAULT_N_CTX = 12288


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
    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        temperature: float = 0.2,
        seed: int = 42,
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
        self._temperature = temperature
        self._seed = seed
        self._completion_index = 0
        self._context_window = int(n_ctx)

    def fork(self, worker_index: int) -> LocalQwenBackend:
        """Load an independent inference context for one cluster worker."""
        if worker_index < 1:
            raise ValueError("worker_index must be positive")
        return LocalQwenBackend(
            self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            temperature=self._temperature,
            seed=self._seed + 100_000 * worker_index,
        )

    @property
    def context_window(self) -> int:
        return self._context_window

    def count_prompt_tokens(self, system: str, user: str) -> int:
        """Count the exact chat-template tokens used by the local backend."""
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
    ) -> str:
        call_seed = self._seed + self._completion_index
        self._completion_index += 1
        response_format: dict[str, object] = {"type": "json_object"}
        if schema is not None:
            response_format["schema"] = dict(schema)
        try:
            response: Any = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{user}\n\n/no_think"},
                ],
                temperature=self._temperature,
                seed=call_seed,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except ValueError as exc:
            if "exceed context window" in str(exc):
                raise ContextWindowExceededError(
                    "Qwen prompt exceeded the configured context window; "
                    "reduce the prompt or increase --n-ctx"
                ) from exc
            raise
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen returned an unexpected response shape") from exc
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        if choice.get("finish_reason") == "length":
            raise CompletionTruncatedError(
                "Qwen response was truncated before it completed the requested JSON",
                partial_content=content if isinstance(content, str) else "",
                prompt_tokens=(prompt_tokens if isinstance(prompt_tokens, int) else None),
                completion_tokens=(
                    completion_tokens if isinstance(completion_tokens, int) else None
                ),
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Qwen returned empty assistant content")
        return content.strip()
