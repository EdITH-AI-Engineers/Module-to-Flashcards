import asyncio
from pathlib import Path
import threading

import pytest

import api_server
from batch_pipeline import BatchResult


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename
        self.closed = False

    async def close(self):
        self.closed = True


class ClosingUpload(FakeUpload):
    def __init__(self, filename, *, close_error=None):
        super().__init__(filename)
        self.close_error = close_error

    async def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


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


@pytest.mark.parametrize("course_code", ("", " ", ".", ".."))
def test_course_upload_directory_rejects_blank_and_dot_components(course_code):
    with pytest.raises(ValueError):
        api_server.course_upload_directory(course_code)


def test_course_upload_directory_sanitizes_separators_and_stays_contained(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")

    destination = api_server.course_upload_directory("CPE/../2026\\A")

    assert destination.parent == (tmp_path / "uploads").resolve()
    assert destination.name == "CPE_.._2026_A"


def test_later_sanitized_filename_collision_is_an_ordered_file_error(
    monkeypatch, tmp_path
):
    uploads = [FakeUpload("lecture .pdf"), FakeUpload("lecture_.pdf"), FakeUpload("ok.pdf")]
    saved = []
    batches = []

    async def fake_save(upload, course_code):
        saved.append(upload.filename)
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        batches.append(tuple(item.filename for item in items))
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert saved == ["lecture .pdf", "ok.pdf"]
    assert batches == [("lecture .pdf", "ok.pdf")]
    assert response["errors"] == [
        {
            "pdf": "lecture_.pdf",
            "error": "sanitized filename collides with an earlier upload: lecture_.pdf",
        }
    ]


def test_later_workspace_collision_is_an_ordered_file_error(monkeypatch, tmp_path):
    uploads = [FakeUpload("lecture-.pdf"), FakeUpload("lecture.pdf"), FakeUpload("ok.pdf")]
    saved = []

    async def fake_save(upload, course_code):
        saved.append(upload.filename)
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(
        api_server,
        "run_batch",
        lambda items, **kwargs: BatchResult(outputs=(), errors=()),
    )

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert saved == ["lecture-.pdf", "ok.pdf"]
    assert response["errors"] == [
        {
            "pdf": "lecture.pdf",
            "error": "workspace collides with an earlier upload: lecture",
        }
    ]


def test_api_errors_follow_the_original_upload_order_across_phases(
    monkeypatch, tmp_path
):
    uploads = [FakeUpload("batch.pdf"), FakeUpload("save.pdf")]

    async def fake_save(upload, course_code):
        if upload.filename == "save.pdf":
            raise OSError("save failed")
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        return BatchResult(
            outputs=(),
            errors=({"pdf": "batch.pdf", "error": "batch failed"},),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert response["errors"] == [
        {"pdf": "batch.pdf", "error": "batch failed"},
        {"pdf": "save.pdf", "error": "save failed"},
    ]


def test_all_uploads_are_closed_after_a_close_error(monkeypatch, tmp_path):
    uploads = [
        ClosingUpload("first.pdf", close_error=RuntimeError("close failed")),
        ClosingUpload("second.pdf"),
    ]

    async def fake_save(upload, course_code):
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(
        api_server,
        "run_batch",
        lambda items, **kwargs: BatchResult(
            outputs=(), errors=({"pdf": "first.pdf", "error": "batch failed"},)
        ),
    )

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert response["errors"] == [
        {"pdf": "first.pdf", "error": "batch failed"},
        {"pdf": "first.pdf", "error": "close failed"},
    ]
    assert all(upload.closed for upload in uploads)


def test_complete_api_requests_are_serialized_without_blocking_batch_work(
    monkeypatch, tmp_path
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    events = []

    async def fake_save(upload, course_code):
        events.append(("save", upload.filename))
        if upload.filename == "first.pdf":
            first_started.set()
            await release_first.wait()
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        events.append(("batch", items[0].filename))
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    async def exercise():
        first = asyncio.create_task(
            api_server.process_files("CPE", [FakeUpload("first.pdf")])
        )
        await first_started.wait()
        second = asyncio.create_task(
            api_server.process_files("CPE", [FakeUpload("second.pdf")])
        )
        await asyncio.sleep(0)
        assert events == [("save", "first.pdf")]
        release_first.set()
        await asyncio.gather(first, second)

    asyncio.run(exercise())

    assert events == [
        ("save", "first.pdf"),
        ("batch", "first.pdf"),
        ("save", "second.pdf"),
        ("batch", "second.pdf"),
    ]


def test_same_stem_uploads_from_different_courses_get_distinct_workspaces(
    monkeypatch, tmp_path
):
    captured = []

    async def fake_save(upload, course_code):
        path = tmp_path / "uploads" / course_code / upload.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(course_code.encode())
        return path

    def fake_batch(items, **kwargs):
        captured.append(items[0].paths.workspace)
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    asyncio.run(api_server.process_files("CPE-A", [FakeUpload("Module 1.pdf")]))
    asyncio.run(api_server.process_files("CPE-B", [FakeUpload("Module 1.pdf")]))

    assert captured[0] != captured[1]
    assert captured == [
        tmp_path / "output" / "CPE-A" / "Module_1",
        tmp_path / "output" / "CPE-B" / "Module_1",
    ]
