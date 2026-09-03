import asyncio

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
