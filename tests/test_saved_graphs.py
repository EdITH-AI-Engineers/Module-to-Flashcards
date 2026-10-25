import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json

import httpx
import pytest

import api_server as server
import batch_pipeline
from batch_pipeline import run_batch
from flashcard_csv import valid_written_receipt
from flashcard_types import ModuleIdentity
from saved_graphs import (
    SavedGraphModule, SavedStructuredModule, copy_checkpoint, copy_graph_checkpoint,
    discover_checkpoints, discover_graphs, saved_run_root, saved_run_summary,
)
from tests.batch_helpers import fake_dependencies, make_items, materialize, structured_content
from tests.factories import valid_clusters


SOURCE_ID = "a" * 32


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    server._SESSIONS.clear()
    server._PIPELINE_STATUS.reset()
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path / "pipeline_output")
    root = tmp_path / "runs" / SOURCE_ID
    root.mkdir(parents=True)
    originals = make_items(root, "module.one.txt", "module.two.txt")
    items = []
    for item in originals:
        paths = server.pipeline_paths(item.args.input, root / "pipeline_output",
                                      item.args.course_code, item.args.module_number)
        item = replace(item, paths=paths)
        materialize(item, "graph")
        graph = json.loads(item.paths.unchecked_graph_json.read_text(encoding="utf-8"))
        graph["metadata"]["course_code"] = item.args.course_code
        item.paths.unchecked_graph_json.write_text(json.dumps(graph), encoding="utf-8")
        items.append(item)
    materialize(items[0], "flashcards")
    # Recovery must work without any original inputs or reuse manifests.
    for item in items:
        item.args.input.unlink()
    items[0].paths.unchecked_graph_json.unlink()
    corpus = items[0].paths.flashcards.parent / "course_corpus.json"
    corpus.write_text(json.dumps({"concept_names": ["Processor"], "questions": ["What is a processor?"]}), encoding="utf-8")
    yield root, tuple(items)
    server._SESSIONS.clear()
    server._PIPELINE_STATUS.reset()


def test_discovery_and_copy_preserve_checked_graphs_completed_pairs_and_corpus(checkpoint):
    root, items = checkpoint
    modules, errors = discover_graphs(root)
    assert not errors
    assert [(m.module_number, m.checked, m.completed) for m in modules] == [
        ("1", True, True), ("2", False, False)]
    summary = saved_run_summary(root)
    assert summary["moduleCount"] == 2 and summary["completedCount"] == 1 and summary["pendingCount"] == 1
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    destination = root.parent / ("b" * 32)
    for module in modules:
        copied = copy_graph_checkpoint(module, destination)
        assert copied.workspace.name == module.paths.workspace.name
        copied_graph = copied.graph_json if module.checked else copied.unchecked_graph_json
        assert copied_graph.read_bytes() == module.graph.read_bytes()
        if module.completed:
            assert valid_written_receipt(copied.flashcards, ModuleIdentity(module.course_code, module.module_number))
            assert (copied.flashcards.parent / "course_corpus.json").read_bytes() == (
                module.paths.flashcards.parent / "course_corpus.json").read_bytes()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # A tampered pair must be regenerated, never offered as a completed result.
    items[0].paths.flashcard_parts[0].write_text("truncated", encoding="utf-8")
    assert saved_run_summary(root)["completedCount"] == 0


def test_discovery_falls_back_from_broken_checked_graph_and_reports_bad_modules(checkpoint):
    root, items = checkpoint
    items[1].paths.graph_dir.mkdir(parents=True)
    items[1].paths.graph_json.write_text("broken", encoding="utf-8")
    modules, errors = discover_graphs(root)
    assert len(modules) == 2 and not errors and not modules[1].checked
    items[1].paths.unchecked_graph_json.write_text('{"nodes":[],"edges":[],"facts":[]}', encoding="utf-8")
    modules, errors = discover_graphs(root)
    assert len(modules) == 1 and len(errors) == 1
    assert errors[0]["file"] == items[1].paths.workspace.name


def test_graph_only_batch_never_ingests_or_extracts_and_skips_completed_modules(checkpoint, monkeypatch):
    _root, items = checkpoint
    for item in items:
        item.args.start_from_graph = True
        item.args.mistral_model = "changed-model"
    events = []

    @contextmanager
    def model_loader(_args):
        events.append(("model",))
        yield object()

    @contextmanager
    def forbidden_loader(_args):
        raise AssertionError("Graph extraction must not load")
        yield

    def forbidden_manifest(_item):
        raise AssertionError("Missing PDFs must not be hashed")

    monkeypatch.setattr(batch_pipeline, "_write_manifest", forbidden_manifest)
    result = run_batch(items, dependencies=fake_dependencies(events, model_loader, forbidden_loader))
    assert not result.errors
    assert result.outputs == tuple(path for item in items for path in item.paths.flashcard_parts)
    assert [event[0] for event in events] == ["model", "flashcards"]
    assert events[1][1] == items[1].filename


def test_missing_graph_fails_without_falling_back_or_deleting_existing_outputs(checkpoint):
    _root, items = checkpoint
    item = items[0]
    item.args.start_from_graph = True
    item.paths.graph_json.unlink()
    before = item.paths.flashcard_parts[0].read_bytes()
    progress = []
    result = run_batch((item,), dependencies=fake_dependencies([], None, None), progress=progress.append)
    assert not result.outputs and "No usable saved knowledge graph" in result.errors[0]["error"]
    assert progress[-1].state == "failed"
    assert item.paths.flashcard_parts[0].read_bytes() == before


def test_production_graph_recovery_reviews_pending_graph_and_publishes_csvs(checkpoint, monkeypatch):
    root, _items = checkpoint
    reviewed, generated = [], []

    def review(graph, backend, **_kwargs):
        reviewed.append(graph["metadata"]["module_number"])
        return {
            **graph,
            "edges": [{**edge, "evidence": []} for edge in graph["edges"]],
            "metadata": {**graph["metadata"], "node_count": len(graph["nodes"]),
                         "edge_count": len(graph["edges"]), "graph_checker": {"status": "checked"}},
        }

    class Generator:
        def __init__(self, backend, config, **_kwargs):
            assert config.final_review

        def run(self, identity, facts, **kwargs):
            assert facts and kwargs["prior_concept_names"] == ["Processor"]
            generated.append(identity.module_number)
            return valid_clusters()

    @contextmanager
    def loader(_args):
        yield object()

    monkeypatch.setattr(batch_pipeline.flashcard_generator, "check_knowledge_graph", review)
    monkeypatch.setattr(batch_pipeline.flashcard_generator, "FlashcardPipeline", Generator)
    modules, _errors = discover_graphs(root)
    items = []
    for index, module in enumerate(modules):
        paths = copy_graph_checkpoint(module, root.parent / ("b" * 32))
        args = server.pipeline_args(paths.graph_json if module.checked else paths.unchecked_graph_json,
                                    module.course_code, module.module_number)
        args.start_from_graph = True
        args.skip_final_review = False
        items.append(batch_pipeline.BatchItem(module.graph.name, args, paths, request_index=index))
    dependencies = replace(batch_pipeline.PRODUCTION_DEPENDENCIES, model_loader=loader)
    result = run_batch(items, dependencies=dependencies)
    assert not result.errors and len(result.outputs) == 4
    assert reviewed == generated == ["2"]
    assert not items[1].paths.unchecked_graph_json.exists()
    assert modules[1].graph.is_file()
    assert saved_run_summary(root.parent / ("b" * 32))["completedCount"] == 2


def test_resume_api_after_restart_with_downloads_progress_and_no_source_changes(checkpoint, monkeypatch):
    root, _items = checkpoint
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    events, progress = [], []

    @contextmanager
    def loader(_args):
        yield object()

    def process(items, **kwargs):
        assert all(item.args.start_from_graph and item.args.skip_final_review is False for item in items)
        assert all(item.args.mistral_model == "recovery-model" for item in items)
        assert all(item.args.cluster_workers == 2 for item in items)
        def record(event):
            progress.append(event)
            kwargs["progress"](event)
        return run_batch(items, dependencies=fake_dependencies(events, loader, None), progress=record)

    monkeypatch.setattr(server, "run_batch", process)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            listed = await client.get("/api/saved-runs")
            assert listed.status_code == 200 and listed.headers["Cache-Control"] == "no-store"
            assert listed.json()["runs"][0]["pendingCount"] == 1
            created = (await client.post("/api/sessions", json={
                "apiKeys": ["private-key"], "model": "recovery-model", "workers": 2})).json()
            session_id = created["sessionId"]
            resumed = await client.post(f"/api/sessions/{session_id}/resume", json={"sourceRunId": SOURCE_ID})
            assert resumed.status_code == 200 and resumed.json()["errors"] == []
            assert len(resumed.json()["outputs"]) == 4
            assert all(session_id in path and SOURCE_ID not in path for path in resumed.json()["outputs"])
            status = await client.get(f"/api/sessions/{session_id}")
            assert "private-key" not in status.text and not status.json()["running"]
            assert status.json()["summary"]["completed"] == 2
            for index in (0, 1):
                for part in (1, 2):
                    download = await client.get(f"/api/sessions/{session_id}/files/{index}/{part}")
                    assert download.status_code == 200 and "question" in download.text.lower()
            assert (await client.get("/api/sessions")).json()["sessions"][0]["sessionId"] == session_id
            # A second recovery keeps both completed pairs even after session memory is lost.
            server._SESSIONS.clear()
            saved = (await client.get("/api/saved-runs")).json()["runs"]
            assert next(run for run in saved if run["runId"] == session_id)["completedCount"] == 2
            next_id = (await client.post("/api/sessions", json={
                "apiKeys": ["new-key"], "model": "recovery-model", "workers": 2})).json()["sessionId"]
            again = await client.post(f"/api/sessions/{next_id}/resume", json={"sourceRunId": session_id})
            assert again.status_code == 200 and len(again.json()["outputs"]) == 4

    asyncio.run(scenario())
    assert [event[0] for event in events] == ["flashcards"]
    assert all(event.stage in {"flashcards", "complete"} for event in progress)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_resume_rejects_busy_sources_invalid_paths_and_reused_sessions(checkpoint):
    root, _items = checkpoint
    source_session = server._MistralSession(server.ApiKeyPool(["source-key"]), "model", 1)
    server._SESSIONS[SOURCE_ID] = source_session
    target = server.create_session(server.SessionSettings(apiKeys=["target-key"]), server.Response())["sessionId"]

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            for source, status in [("../outside", 422), ("f" * 32, 404), (target, 409)]:
                response = await client.post(f"/api/sessions/{target}/resume", json={"sourceRunId": source})
                assert response.status_code == status
                assert not server._get_session(target).lock.locked()
            source_session.lock.acquire()
            try:
                assert (await client.get("/api/saved-runs")).json()["runs"][0]["running"]
                response = await client.post(f"/api/sessions/{target}/resume", json={"sourceRunId": SOURCE_ID})
                assert response.status_code == 409 and not server._get_session(target).lock.locked()
            finally:
                source_session.lock.release()
            session = server._get_session(target)
            session.request_id = 123
            assert (await client.post(f"/api/sessions/{target}/resume", json={"sourceRunId": SOURCE_ID})).status_code == 409
            response = await client.get("/api/saved-runs", headers={"Origin": "https://untrusted.example"})
            assert response.status_code == 403

    asyncio.run(scenario())
    with pytest.raises(ValueError):
        saved_run_root(root.parent, "../outside")


def test_browser_exposes_graph_recovery_without_file_upload_requirements(checkpoint):
    from web_ui import HTML
    assert '<option value="graphs">Saved graphs or structured modules</option>' in HTML
    assert 'id="saved-run"' in HTML and "'/api/saved-runs'" in HTML
    assert "/resume" in HTML and "sourceRunId" in HTML
    assert "input.disabled=graphs" in HTML


@pytest.fixture
def structured_checkpoint(checkpoint):
    root, items = checkpoint
    for item in items:
        item.paths.structured_text.write_text(structured_content(item.args.module_number), encoding="utf-8")
    source = root / "CPE0021_M3.txt"
    source.write_text(structured_content("3"), encoding="utf-8-sig")
    args = server.pipeline_args(source, "CPE0021", "3")
    paths = server.pipeline_paths(source, root / "pipeline_output", "CPE0021", "3")
    item = batch_pipeline.BatchItem(source.name, args, paths)
    paths.structured_text.parent.mkdir(parents=True)
    paths.structured_text.write_bytes(source.read_bytes())
    source.unlink()
    return root, (*items, item)


def recovery_dependencies(events):
    @contextmanager
    def loader(_args):
        yield object()

    dependencies = fake_dependencies(events, loader, loader)

    def graph(item, runtime):
        assert item.paths.structured_text.is_file()
        dependencies.graph_stage(item, runtime)
        graph = json.loads(item.paths.unchecked_graph_json.read_text(encoding="utf-8"))
        graph["metadata"]["course_code"] = item.args.course_code
        item.paths.unchecked_graph_json.write_text(json.dumps(graph), encoding="utf-8")

    def forbidden_ingest(*_args):
        pytest.fail("Recovery must never repeat ingestion")

    return replace(dependencies, graph_stage=graph, ingest_stage=forbidden_ingest)


def test_structured_discovery_prefers_graphs_and_copies_text_unchanged(structured_checkpoint):
    root, items = structured_checkpoint
    # Broken text cannot prevent a usable graph from being recovered.
    items[1].paths.structured_text.write_text("broken text", encoding="utf-8")
    modules, errors = discover_checkpoints(root)
    assert not errors
    assert [type(module) for module in modules] == [SavedGraphModule, SavedGraphModule, SavedStructuredModule]
    summary = saved_run_summary(root)
    assert (summary["moduleCount"], summary["graphCount"], summary["structuredCount"],
            summary["completedCount"], summary["pendingCount"]) == (3, 2, 1, 1, 2)
    before = modules[2].source.read_bytes()
    paths = copy_checkpoint(modules[2], root.parent / ("b" * 32))
    assert paths.structured_text.read_bytes() == before
    assert paths.workspace != modules[2].paths.workspace
    assert modules[2].source.read_bytes() == before


@pytest.mark.parametrize("broken_graph", [False, True])
def test_structured_fallback_accepts_missing_or_broken_graphs(structured_checkpoint, broken_graph):
    root, items = structured_checkpoint
    if broken_graph:
        items[2].paths.graph_dir.mkdir()
        items[2].paths.graph_json.write_text("broken graph", encoding="utf-8")
    modules, errors = discover_checkpoints(root)
    assert not errors and isinstance(modules[2], SavedStructuredModule)


@pytest.mark.parametrize("content", [b"", b"not a module", b"\xff\xfe", b"[MODULE]\nformat_version: 2\n"])
def test_invalid_structured_modules_are_reported_without_blocking_graphs(structured_checkpoint, content):
    root, items = structured_checkpoint
    items[2].paths.structured_text.write_bytes(content)
    modules, errors = discover_checkpoints(root)
    assert len(modules) == 2 and len(errors) == 1
    assert errors[0]["file"] == items[2].paths.workspace.name


@pytest.mark.parametrize("report", [False, True])
def test_structured_recovery_preserves_filename_identity_over_conflicting_metadata(structured_checkpoint, report):
    root, items = structured_checkpoint
    if report:
        from tests.test_slide_report_input import REPORT
        content = REPORT
    else:
        content = structured_content("99").replace("CPE0021", "OTHER")
    items[2].paths.structured_text.write_text(content, encoding="utf-8")
    modules, errors = discover_checkpoints(root)
    assert not errors
    module = modules[2]
    assert (module.course_code, module.module_number, module.identity_from_filename) == ("CPE0021", "3", True)
    paths = copy_checkpoint(module, root.parent / ("b" * 32))
    assert paths.structured_text.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("force", [False, True])
def test_structured_batch_skips_ingestion_with_missing_pdf_and_changed_settings(structured_checkpoint, force):
    root, _items = structured_checkpoint
    modules, _errors = discover_checkpoints(root)
    module = modules[2]
    paths = copy_checkpoint(module, root.parent / ("b" * 32))
    args = server.pipeline_args(root / "missing.pdf", module.course_code, module.module_number)
    args.start_from_structured = True
    args.mistral_model = "changed-model"
    args.force = force
    item = batch_pipeline.BatchItem("CPE0021_M3.txt", args, paths)
    events, progress = [], []
    result = run_batch((item,), dependencies=recovery_dependencies(events), progress=progress.append)
    assert not result.errors and len(result.outputs) == 2
    assert [event[0] for event in events] == ["graph", "flashcards"]
    assert progress[0].stage == "graph" and all(event.stage != "ingest" for event in progress)


def test_missing_structured_checkpoint_fails_before_mutating_outputs(structured_checkpoint):
    root, items = structured_checkpoint
    item = items[0]
    item.args.start_from_structured = True
    before = item.paths.flashcard_parts[0].read_bytes()
    item.paths.structured_text.unlink()
    events = []
    result = run_batch((item,), dependencies=recovery_dependencies(events))
    assert not result.outputs and "No usable saved structured module" in result.errors[0]["error"]
    assert not events and item.paths.flashcard_parts[0].read_bytes() == before


def test_duplicate_structured_identity_is_reported_without_replacing_graph(structured_checkpoint):
    root, items = structured_checkpoint
    source = root / "pipeline_output" / "CPE0021" / "duplicate" / "structured_module.txt"
    source.parent.mkdir()
    source.write_text(structured_content("1"), encoding="utf-8")
    modules, errors = discover_checkpoints(root)
    assert len(modules) == 3 and len(errors) == 1
    assert modules[0].source == items[0].paths.graph_json
    assert "More than one" in errors[0]["error"]


def test_same_workspace_name_in_other_course_keeps_its_graph_error(structured_checkpoint):
    root, _items = structured_checkpoint
    graph = root / "pipeline_output" / "OTHER" / "CPE0021_M3" / "knowledge_graph" / "knowledge_graph.json"
    graph.parent.mkdir(parents=True)
    graph.write_text("broken", encoding="utf-8")
    modules, errors = discover_checkpoints(root)
    assert len(modules) == 3 and len(errors) == 1
    assert "valid JSON" in errors[0]["error"]


def test_production_graph_adapter_uses_saved_structured_text(structured_checkpoint):
    root, _items = structured_checkpoint
    modules, _errors = discover_checkpoints(root)
    module = modules[2]
    paths = copy_checkpoint(module, root.parent / ("b" * 32))
    args = server.pipeline_args(root / "missing.pdf", module.course_code, module.module_number)
    args.start_from_structured = True
    item = batch_pipeline.BatchItem("CPE0021_M3.txt", args, paths)

    class Backend:
        model_name = "test-recovery"

        def complete(self, system, user, **kwargs):
            assert "processor" in user
            return '{"triples":[]}'

    @contextmanager
    def loader(_args):
        yield Backend()

    def forbidden_ingest(*_args):
        pytest.fail("Saved text must skip PDF ingestion")

    dependencies = replace(batch_pipeline.PRODUCTION_DEPENDENCIES,
        rebel_loader=loader, model_loader=loader, ingest_stage=forbidden_ingest,
        flashcard_stage=lambda current, _backend: materialize(current, "flashcards"))
    result = run_batch((item,), dependencies=dependencies)
    assert not result.errors and len(result.outputs) == 2
    graph = json.loads(paths.graph_json.read_text(encoding="utf-8"))
    assert graph["metadata"]["course_code"] == "CPE0021"
    assert graph["metadata"]["module_number"] == "3" and graph["facts"]


@pytest.mark.parametrize("structured_only", [False, True])
def test_resume_api_recovers_structured_and_mixed_runs_after_restart(structured_checkpoint, monkeypatch, structured_only):
    root, items = structured_checkpoint
    if structured_only:
        for item in items:
            item.paths.graph_json.unlink(missing_ok=True)
            item.paths.unchecked_graph_json.unlink(missing_ok=True)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    events, progress = [], []

    def process(batch, **kwargs):
        assert all(item.args.skip_final_review is False for item in batch)
        assert all(item.args.mistral_model == "recovery-model" for item in batch)
        def record(event):
            progress.append(event)
            kwargs["progress"](event)
        return run_batch(batch, dependencies=recovery_dependencies(events), progress=record)

    monkeypatch.setattr(server, "run_batch", process)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
            summary = (await client.get("/api/saved-runs")).json()["runs"][0]
            assert summary["structuredCount"] == (3 if structured_only else 1)
            assert summary["graphCount"] == (0 if structured_only else 2)
            created = (await client.post("/api/sessions", json={
                "apiKeys": ["private-key"], "model": "recovery-model"})).json()
            session_id = created["sessionId"]
            response = await client.post(f"/api/sessions/{session_id}/resume", json={"sourceRunId": SOURCE_ID})
            assert response.status_code == 200 and response.json()["errors"] == []
            assert len(response.json()["outputs"]) == 6
            assert all(session_id in path for path in response.json()["outputs"])
            status = (await client.get(f"/api/sessions/{session_id}")).json()
            assert status["summary"]["completed"] == 3 and not status["running"]
            for index in range(3):
                for part in (1, 2):
                    assert (await client.get(f"/api/sessions/{session_id}/files/{index}/{part}")).status_code == 200
            server._SESSIONS.clear()
            summaries = (await client.get("/api/saved-runs")).json()["runs"]
            recovered = next(run for run in summaries if run["runId"] == session_id)
            assert recovered["completedCount"] == recovered["graphCount"] == 3
            assert recovered["structuredCount"] == 0

    asyncio.run(scenario())
    assert sum(event[0] == "graph" for event in events) == (3 if structured_only else 1)
    assert sum(event[0] == "flashcards" for event in events) == (3 if structured_only else 2)
    assert all(event.stage != "ingest" for event in progress)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
