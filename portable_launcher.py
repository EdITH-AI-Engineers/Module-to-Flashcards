from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import traceback
from typing import Any, Callable
from urllib.request import urlopen

from portable_manifest import verify_manifest
from portable_paths import PortablePaths, build_paths, prepare_data_directories
from portable_runtime import (
    AccelerationStatus,
    configure_offline_environment,
    detect_acceleration,
)
from version import __version__


FREE = "free"
SAME_APP = "same-application"
FOREIGN = "foreign-listener"


def valid_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline Module to Flashcards API server."
    )
    parser.add_argument("--port", type=valid_port, default=8000)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="rehash every bundled asset and exit",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser.parse_args(argv)


def probe_port(
    port: int,
    *,
    expected_version: str = __version__,
    timeout: float = 0.75,
) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return FREE

    response = None
    try:
        response = urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout)
        payload = json.loads(response.read())
    except Exception:
        return FOREIGN
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("version") == expected_version
    ):
        return SAME_APP
    return FOREIGN


def _default_load_server() -> Any:
    import api_server

    return api_server


def _default_run_server(app: object, **kwargs: object) -> None:
    import uvicorn

    uvicorn.run(app, **kwargs)


def _configure_ocr(paths: PortablePaths) -> None:
    import pytesseract

    from pdf_ingestion import _configure_tesseract

    _configure_tesseract(
        pytesseract,
        portable_executable=paths.tesseract_exe,
        tessdata_dir=paths.tessdata,
    )


def _device_settings(status: AccelerationStatus) -> tuple[int, str]:
    if status.torch_cuda and status.llama_gpu_offload:
        return -1, "cuda"
    return 0, "cpu"


def _device_message(status: AccelerationStatus) -> str:
    if status.torch_cuda and status.llama_gpu_offload:
        return f"GPU acceleration ready: {status.torch_device_name or 'CUDA device'}"
    missing = []
    if not status.torch_cuda:
        missing.append("PyTorch CUDA")
    if not status.llama_gpu_offload:
        missing.append("llama.cpp GPU offload")
    return "WARNING: CPU fallback active; unavailable: " + ", ".join(missing)


def _write_startup_log(paths: PortablePaths, exc: BaseException) -> Path:
    paths.logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = paths.logs / f"startup-{stamp}.log"
    destination.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    return destination


def run_launcher(
    args: argparse.Namespace,
    *,
    root: Path | None = None,
    prepare_data: Callable[[PortablePaths], None] = prepare_data_directories,
    configure_offline: Callable[[], None] | None = None,
    verify_bundle: Callable[[bool], object] | None = None,
    detect_devices: Callable[[], AccelerationStatus] = detect_acceleration,
    probe: Callable[[int], str] = probe_port,
    load_server: Callable[[], Any] = _default_load_server,
    configure_ocr: Callable[[PortablePaths], None] = _configure_ocr,
    run_server: Callable[..., None] = _default_run_server,
    output: Callable[[str], None] = print,
) -> int:
    if args.version:
        output(f"ModuleToFlashcards {__version__}")
        return 0

    paths = build_paths(root, portable=True)
    try:
        prepare_data(paths)
        if configure_offline is None:
            configure_offline_environment(paths.temporary / "huggingface")
        else:
            configure_offline()

        if verify_bundle is None:
            verify_manifest(
                paths.root,
                paths.models / "manifest.json",
                cache_path=paths.data / ".verification-cache.json",
                force_hash=args.verify,
            )
        else:
            verify_bundle(args.verify)

        status = detect_devices()
        output(_device_message(status))
        if args.verify:
            output("Bundle verification passed.")
            return 0

        port_status = probe(args.port)
        if port_status == SAME_APP:
            output(f"ModuleToFlashcards is already running on http://127.0.0.1:{args.port}")
            return 0
        if port_status != FREE:
            raise RuntimeError(
                f"port {args.port} is already in use by another application"
            )

        server = load_server()
        n_gpu_layers, kg_device = _device_settings(status)
        server.configure_api_storage(paths)
        server.configure_api_runtime(
            n_gpu_layers=n_gpu_layers,
            kg_device=kg_device,
        )
        configure_ocr(paths)
        output(f"Ready at http://127.0.0.1:{args.port} (Ctrl+C to stop)")
        run_server(
            server.app,
            host="127.0.0.1",
            port=args.port,
            reload=False,
        )
        return 0
    except Exception as exc:
        try:
            log_path = _write_startup_log(paths, exc)
            output(f"Startup failed: {exc}. Details: {log_path}")
        except OSError:
            output(f"Startup failed: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_launcher(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
