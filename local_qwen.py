from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-3b-instruct-q8_0.gguf"


def ensure_model(model_dir: Path) -> Path:
    """Return the exact local Q8 model path, downloading it when absent."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / MODEL_FILENAME
    if target.is_file():
        return target

    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(model_dir),
    )
    return Path(downloaded)


class LocalQwenBackend:
    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 32768,
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

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        response: Any = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self._temperature,
            seed=self._seed,
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
