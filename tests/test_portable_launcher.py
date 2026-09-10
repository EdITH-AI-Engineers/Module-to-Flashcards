from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from portable_launcher import FREE, FOREIGN, SAME_APP, parse_args, probe_port, run_launcher
from portable_runtime import AccelerationStatus


def _gpu_status() -> AccelerationStatus:
    return AccelerationStatus(True, True, "RTX Test")


def _cpu_status() -> AccelerationStatus:
    return AccelerationStatus(False, False, None)


class FakeServer:
    app = object()

    def __init__(self, events):
        self.events = events

    def configure_api_storage(self, paths):
        self.events.append(("storage", paths.root))

    def configure_api_runtime(self, *, n_gpu_layers, kg_device):
        self.events.append(("devices", n_gpu_layers, kg_device))


def _dependencies(events, *, status=None, port_status=FREE):
    server = FakeServer(events)
    return {
        "prepare_data": lambda _paths: events.append("prepare"),
        "configure_offline": lambda: events.append("offline"),
        "verify_bundle": lambda force: events.append(("verify", force)),
        "detect_devices": lambda: events.append("detect") or (status or _gpu_status()),
        "probe": lambda _port: events.append("probe") or port_status,
        "load_server": lambda: events.append("server") or server,
        "configure_ocr": lambda _paths: events.append("ocr"),
        "run_server": lambda *_args, **kwargs: events.append(("run", kwargs)),
        "output": lambda message: events.append(("output", message)),
    }


def test_parse_args_supports_port_verify_and_version():
    assert parse_args([]).port == 8000
    args = parse_args(["--port", "8010", "--verify", "--version"])
    assert (args.port, args.verify, args.version) == (8010, True, True)


@pytest.mark.parametrize("value", ("0", "65536", "invalid"))
def test_parse_args_rejects_invalid_ports(value):
    with pytest.raises(SystemExit):
        parse_args(["--port", value])


def test_launcher_configures_offline_mode_before_loading_server(tmp_path):
    events = []

    result = run_launcher(parse_args([]), root=tmp_path, **_dependencies(events))

    assert result == 0
    assert events.index("offline") < events.index("server")
    assert ("devices", -1, "cuda") in events
    assert any(event[0] == "run" for event in events if isinstance(event, tuple))


def test_verify_mode_forces_hashes_and_does_not_start_server(tmp_path):
    events = []

    result = run_launcher(
        parse_args(["--verify"]), root=tmp_path, **_dependencies(events)
    )

    assert result == 0
    assert ("verify", True) in events
    assert "detect" in events
    assert "server" not in events
    assert not any(event[0] == "run" for event in events if isinstance(event, tuple))


def test_cpu_fallback_configures_both_models_for_cpu(tmp_path):
    events = []

    result = run_launcher(
        parse_args([]),
        root=tmp_path,
        **_dependencies(events, status=_cpu_status()),
    )

    assert result == 0
    assert ("devices", 0, "cpu") in events
    messages = [event[1] for event in events if isinstance(event, tuple) and event[0] == "output"]
    assert any("CPU fallback" in message for message in messages)


def test_healthy_existing_instance_exits_without_loading_server(tmp_path):
    events = []

    result = run_launcher(
        parse_args([]),
        root=tmp_path,
        **_dependencies(events, port_status=SAME_APP),
    )

    assert result == 0
    assert "server" not in events


def test_foreign_port_owner_is_logged_and_returns_failure(tmp_path):
    events = []

    result = run_launcher(
        parse_args([]),
        root=tmp_path,
        **_dependencies(events, port_status=FOREIGN),
    )

    assert result == 1
    logs = list((tmp_path / "data" / "logs").glob("startup-*.log"))
    assert len(logs) == 1
    assert "already in use" in logs[0].read_text(encoding="utf-8")


def test_probe_port_recognizes_matching_health_response(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def settimeout(self, _timeout):
            return None

        def connect_ex(self, _address):
            return 0

    response = SimpleNamespace(
        __enter__=lambda self: self,
        __exit__=lambda self, *_args: None,
        read=lambda: json.dumps({"status": "ok", "version": "1.2.3"}).encode(),
    )
    monkeypatch.setattr("portable_launcher.socket.socket", lambda *_args: FakeSocket())
    monkeypatch.setattr("portable_launcher.urlopen", lambda *_args, **_kwargs: response)

    assert probe_port(8000, expected_version="1.2.3") == SAME_APP


def test_version_mode_does_not_touch_bundle(tmp_path):
    events = []

    result = run_launcher(
        parse_args(["--version"]), root=tmp_path, **_dependencies(events)
    )

    assert result == 0
    assert events and events[0][0] == "output"
    assert "prepare" not in events
