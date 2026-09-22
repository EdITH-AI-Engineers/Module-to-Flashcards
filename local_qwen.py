from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "4f02e7c52b572082828edf5058a87e2e7dc3e4d5"
MODEL_FILENAME = "Qwen3-8B-Q5_K_M.gguf"
DEFAULT_N_CTX = 8192


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

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        self._temperature = temperature
        self._seed = seed
        self._completion_index = 0

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        call_seed = self._seed + self._completion_index
        self._completion_index += 1
        response: Any = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n\n/no_think"},
            ],
            temperature=self._temperature,
            seed=call_seed,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen returned an unexpected response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Qwen returned empty assistant content")
        return content.strip()
