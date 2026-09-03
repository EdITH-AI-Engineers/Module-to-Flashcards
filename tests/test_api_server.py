import asyncio
from pathlib import Path
import threading

import api_server
from batch_pipeline import BatchResult


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename
        self.closed = False

    async def close(self):
        self.closed = True


def test_process_files_saves_all_uploads_then_runs_one_batch(monkeypatch, tmp_path):
    uploads = [FakeUpload("CPE-M1.pdf"), FakeUpload("CPE-M2.pdf")]
    events = []

    async def fake_save(upload, course_code):
        events.append(("save", upload.filename))
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        events.append(("batch", tuple(item.filename for item in items)))
        return BatchResult(
            outputs=tuple(item.paths.flashcards for item in items),
            errors=(),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert events == [
        ("save", "CPE-M1.pdf"),
        ("save", "CPE-M2.pdf"),
        ("batch", ("CPE-M1.pdf", "CPE-M2.pdf")),
    ]
    assert len(response["outputs"]) == 2
    assert all(upload.closed for upload in uploads)


def test_save_failure_does_not_prevent_other_files_from_batching(monkeypatch, tmp_path):
    uploads = [FakeUpload("bad.pdf"), FakeUpload("good.pdf")]
    captured = []

    async def fake_save(upload, course_code):
        if upload.filename == "bad.pdf":
            raise OSError("disk write failed")
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        captured.extend(item.filename for item in items)
        return BatchResult(outputs=(items[0].paths.flashcards,), errors=())

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert captured == ["good.pdf"]
    assert response["errors"] == [{"pdf": "bad.pdf", "error": "disk write failed"}]
    assert all(upload.closed for upload in uploads)


def test_batch_runtime_failure_becomes_ordered_per_file_errors(monkeypatch, tmp_path):
    uploads = [FakeUpload("CPE-M1.pdf"), FakeUpload("CPE-M2.pdf")]

    async def fake_save(upload, course_code):
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert response == {
        "outputs": [],
        "errors": [
            {"pdf": "CPE-M1.pdf", "error": "model load failed"},
            {"pdf": "CPE-M2.pdf", "error": "model load failed"},
        ],
        "courseCode": "CPE",
        "processUrl": "/process/CPE",
    }
    assert all(upload.closed for upload in uploads)


def test_process_files_runs_batch_off_the_event_loop_thread(monkeypatch, tmp_path):
    upload = FakeUpload("CPE-M1.pdf")
    event_loop_thread = []
    batch_thread = []

    async def fake_save(upload, course_code):
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        batch_thread.append(threading.get_ident())
        return BatchResult(outputs=(), errors=())

    async def process():
        event_loop_thread.append(threading.get_ident())
        return await api_server.process_files("CPE", [upload])

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    asyncio.run(process())

    assert batch_thread[0] != event_loop_thread[0]


def test_pipeline_args_use_low_resource_api_defaults():
    args = api_server.pipeline_args(Path("module.pdf"), "CPE", "1")

    assert args.n_ctx == api_server.DEFAULT_N_CTX
    assert args.kg_batch_size == 1
    assert args.kg_num_beams == 1
    assert args.skip_final_review is True
    assert args.timeout == 0


def test_all_save_failures_still_invoke_empty_batch(monkeypatch):
    uploads = [FakeUpload("bad-1.pdf"), FakeUpload("bad-2.pdf")]
    batches = []

    async def fake_save(upload, course_code):
        raise OSError("disk write failed")

    def fake_batch(items, **kwargs):
        batches.append(tuple(items))
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert batches == [()]
    assert response["outputs"] == []
    assert response["errors"] == [
        {"pdf": "bad-1.pdf", "error": "disk write failed"},
        {"pdf": "bad-2.pdf", "error": "disk write failed"},
    ]
    assert all(upload.closed for upload in uploads)
