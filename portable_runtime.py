from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AccelerationStatus:
    torch_cuda: bool
    llama_gpu_offload: bool
    torch_device_name: str | None


def configure_offline_environment(cache_dir: Path | None = None) -> None:
    """Force model libraries to use bundled files without network access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if cache_dir is not None:
        os.environ["HF_HOME"] = str(Path(cache_dir))


def detect_acceleration() -> AccelerationStatus:
    """Report the CUDA capabilities used by both bundled model runtimes."""
    import llama_cpp
    import torch

    torch_cuda = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if torch_cuda else None
    return AccelerationStatus(
        torch_cuda=torch_cuda,
        llama_gpu_offload=bool(llama_cpp.llama_supports_gpu_offload()),
        torch_device_name=device_name,
    )
