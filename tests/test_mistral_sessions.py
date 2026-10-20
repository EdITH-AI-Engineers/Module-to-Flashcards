import asyncio
import json
import threading

import httpx
import pytest

import api_server as server
from batch_pipeline import BatchResult
from tests.batch_helpers import structured_content


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch, tmp_path):
    server._SESSIONS.clear()
    server._PIPELINE_STATUS.reset()
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path / "pipeline_output")
    yield
    server._SESSIONS.clear()
    server._PIPELINE_STATUS.reset()


def test_configuration_never_returns_secrets_even_on_validation_failure():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            for payload in [{"apiKeys": ["secret-value"]}, {"apiKeys": [123, "secret-value"]},
                            {"apiKeys": ["secret-value"], "workers": 99}]:
                result = await client.post("/api/sessions", json=payload)
                assert "secret-value" not in result.text
            result = await client.post("/api/sessions", json={"apiKeys": ["secret-value"]}, headers={"Origin": "https://untrusted.example"})
            assert result.status_code == 403
    asyncio.run(scenario())


def test_browser_and_new_sessions_default_to_accessible_hosted_model():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            page = await client.get("/")
            assert page.status_code == 200
            assert 'id="model" value="ministral-14b-2512"' in page.text
            assert "__DEFAULT_MODEL__" not in page.text
            result = await client.post("/api/sessions", json={"apiKeys": ["private-key"]})
            session_id = result.json()["sessionId"]
            assert server._get_session(session_id).model == "ministral-14b-2512"
            status = await client.get("/api/sessions/" + session_id)
            assert status.json()["model"] == "ministral-14b-2512"
            assert "private-key" not in status.text
    asyncio.run(scenario())


@pytest.mark.parametrize("outcome,status", [("ok", 200), ("access_denied", 403), ("model_unavailable", 429), ("bad_json", 502), ("incomplete", 502)])
def test_model_connection_check_before_uploads(monkeypatch, tmp_path, outcome, status):
    from mistral_backend import MistralRequestError
    calls = []
    class Probe:
        def __init__(self, pool, **options):
            calls.append(options)
            assert options["max_retries"] == 1 and options["timeout"] == 30
            assert options["model"] == "ministral-14b-2512"
        def complete(self, system, user, **options):
            assert options["max_tokens"] == 32 and options["schema"]["required"] == ["ok"]
            if outcome in {"access_denied", "model_unavailable"}:
                raise MistralRequestError("Model unavailable for private-key", category=outcome, status=status)
            if outcome == "incomplete":
                from flashcard_types import IncompleteCompletionError
                raise IncompleteCompletionError("Unfinished response (finish_reason: error).", finish_reason="error")
            return '{"ok":true}' if outcome == "ok" else 'not json'
    monkeypatch.setattr(server, "MistralBackend", Probe)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            created = await client.post("/api/sessions", json={"apiKeys": ["private-key"]})
            session_id = created.json()["sessionId"]
            result = await client.post(f"/api/sessions/{session_id}/check")
            assert result.status_code == status
            assert "private-key" not in result.text
            if outcome == "ok":
                assert result.json()["model"] == "ministral-14b-2512"
            session = server._get_session(session_id)
            assert not session.lock.locked()
            assert (await client.get(f"/api/sessions/{session_id}")).json()["modules"] == []
            assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 200
    asyncio.run(scenario())
    assert len(calls) == 1
    assert not (tmp_path / "runs").exists() and not (tmp_path / "uploads").exists()


def test_two_overlapping_uploads_are_isolated(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    calls = []
    def process(items, **kwargs):
        item = items[0]
        content = item.args.input.read_text(encoding="utf-8")
        backend = server._get_session(next(key for key, value in server._SESSIONS.items()
            if value.pool is item.args._mistral_pool))
        calls.append((item, content, backend))
        barrier.wait(timeout=3)  # Fails if a global processing lock serializes runs.
        for path in item.paths.flashcard_parts:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return BatchResult(item.paths.flashcard_parts, ())
    monkeypatch.setattr(server, "run_batch", process)
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            ids = []
            for key, model in [("key-A", "mistral-small-latest"), ("key-B", "mistral-large-latest")]:
                result = await client.post("/api/sessions", json={"apiKeys": [key], "model": model})
                ids.append(result.json()["sessionId"])
            replies = await asyncio.gather(*[
                client.post("/process?session_id=" + session_id,
                    data={"courseCode": "CPE0021"}, files={"files": ("CPE0021-M1.txt",
                        structured_content().replace("processor", label), "text/plain")})
                for session_id, label in zip(ids, ["Alpha", "Beta"])])
            assert all(reply.status_code == 200 and not reply.json()["errors"] for reply in replies)
            assert set(replies[0].json()["outputs"]).isdisjoint(replies[1].json()["outputs"])
            for session_id, label, other in zip(ids, ["Alpha", "Beta"], ["Beta", "Alpha"]):
                status = (await client.get("/api/sessions/" + session_id)).json()
                assert len(status["modules"]) == 1 and not status["running"]
                output = await client.get(f"/api/sessions/{session_id}/files/0/1")
                assert output.status_code == 200 and label in output.text and other not in output.text
                assert (await client.delete("/api/sessions/" + session_id)).status_code == 200
                assert (await client.get("/api/sessions/" + session_id)).status_code == 404
    asyncio.run(scenario())
    assert len(calls) == 2
    assert calls[0][0].args._mistral_pool is not calls[1][0].args._mistral_pool
    assert {item.args.mistral_model for item, _, _ in calls} == {"mistral-small-latest", "mistral-large-latest"}
    assert not (tmp_path / "uploads").exists()


def test_busy_run_rejects_duplicate_upload_and_key_deletion(monkeypatch):
    session_id = server.create_session(server.SessionSettings(apiKeys=["key"]), server.Response())["sessionId"]
    session = server._get_session(session_id)
    session.lock.acquire()
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            result = await client.post("/process?session_id=" + session_id,
                data={"courseCode": "CPE0021"}, files={"files": ("CPE0021-M1.txt", structured_content(), "text/plain")})
            assert result.status_code == 409
            assert (await client.post("/api/sessions/" + session_id + "/check")).status_code == 409
            assert (await client.delete("/api/sessions/" + session_id)).status_code == 409
    try:
        asyncio.run(scenario())
    finally:
        session.lock.release()


def test_model_change_invalidates_cached_batch_outputs(tmp_path, monkeypatch):
    import batch_pipeline
    from tests.batch_helpers import make_items
    item = make_items(tmp_path, "one.txt")[0]
    first = batch_pipeline._manifest_settings(item.args)
    item.args.mistral_model = "mistral-large-latest"
    assert first != batch_pipeline._manifest_settings(item.args)
    assert "key" not in json.dumps(first).lower()


def test_status_explains_rate_wait_without_marking_backups_exhausted():
    import time
    from mistral_backend import MistralRequestError
    session_id = server.create_session(server.SessionSettings(apiKeys=["private-key-one", "private-key-two"]), server.Response())["sessionId"]
    session = server._get_session(session_id)
    index, _ = session.pool.acquire(time.monotonic() + 1)
    session.pool.release(index, cooldown=60, error=MistralRequestError(
        "Token rate limit exceeded for private-key-one", category="rate_limited", status=429, request_id="req-test"))
    data = server.session_status(session_id, server.Response())
    assert [key["state"] for key in data["keys"]] == ["waiting", "standby"]
    assert data["apiDiagnostics"][0]["status"] == 429
    assert data["apiDiagnostics"][0]["requestId"] == "req-test"
    assert "private-key-one" not in json.dumps(data) and "private-key-two" not in json.dumps(data)
    assert server.SessionSettings(apiKeys=["key"]).workers == 1


def test_refresh_rediscovers_running_and_finished_runs_without_processing_again(monkeypatch):
    calls = []

    def process(items, **_kwargs):
        calls.append(items)
        for item in items:
            for path in item.paths.flashcard_parts:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("saved result", encoding="utf-8")
        return BatchResult(tuple(path for item in items for path in item.paths.flashcard_parts), ())

    monkeypatch.setattr(server, "run_batch", process)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            first = (await client.post("/api/sessions", json={"apiKeys": ["private-first"]})).json()
            second = (await client.post("/api/sessions", json={"apiKeys": ["private-second"]})).json()
            assert first["runNumber"] == 1 and second["runNumber"] == 2
            # An unused connection check is not a submitted module run.
            assert (await client.get("/api/sessions")).json() == {"sessions": []}
            completed = await client.post("/process?session_id=" + first["sessionId"],
                data={"courseCode": "CPE0021"},
                files={"files": ("CPE0021-M1.txt", structured_content(), "text/plain")})
            assert completed.status_code == 200 and not completed.json()["errors"]
            active = server._get_session(second["sessionId"])
            active.request_id = server._PIPELINE_STATUS.enqueue("OTHER", ["OTHER-M2.pdf"])
            active.lock.acquire()
            try:
                for _refresh in range(2):
                    listed = await client.get("/api/sessions")
                    assert listed.status_code == 200
                    assert listed.headers["Cache-Control"] == "no-store"
                    sessions = listed.json()["sessions"]
                    assert sessions == [
                        {"sessionId": first["sessionId"], "runNumber": 1,
                         "model": "ministral-14b-2512", "running": False},
                        {"sessionId": second["sessionId"], "runNumber": 2,
                         "model": "ministral-14b-2512", "running": True}]
                    assert "private-first" not in listed.text and "private-second" not in listed.text
                    for metadata in sessions:
                        status = (await client.get("/api/sessions/" + metadata["sessionId"])).json()
                        assert status["runNumber"] == metadata["runNumber"]
                        assert len(status["modules"]) == 1
                    download = await client.get(f'/api/sessions/{first["sessionId"]}/files/0/1')
                    assert download.status_code == 200 and download.text == "saved result"
                assert len(calls) == 1
            finally:
                active.lock.release()
            assert (await client.delete("/api/sessions/" + first["sessionId"])).status_code == 200
            assert [s["sessionId"] for s in (await client.get("/api/sessions")).json()["sessions"]] == [second["sessionId"]]

    asyncio.run(scenario())


def test_session_discovery_rejects_other_origins():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            result = await client.get("/api/sessions", headers={"Origin": "https://untrusted.example"})
            assert result.status_code == 403

    asyncio.run(scenario())
