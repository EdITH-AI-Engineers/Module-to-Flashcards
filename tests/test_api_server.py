import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import threading

import pytest
from fastapi import Response

import api_server
from batch_pipeline import BatchResult
from portable_paths import build_paths
from tests.batch_helpers import structured_content


_REAL_STRUCTURED_MODULE_NUMBER = api_server.structured_module_number


@pytest.fixture(autouse=True)
def stub_structured_identity_for_process_tests(monkeypatch):
    api_server._PIPELINE_STATUS.reset()

    def module_number(source, _course_code):
        stem = Path(source).stem
        match = re.search(r"(?:^|[-_ ])M(?:odule)?[-_ ]?(\d+)(?:\D|$)", stem, re.I)
        return match.group(1) if match else stem

    monkeypatch.setattr(
        api_server,
        "structured_module_number",
        module_number,
    )
    yield
    api_server._PIPELINE_STATUS.reset()


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename
        self.closed = False

    async def close(self):
        self.closed = True


class ContentUpload(FakeUpload):
    def __init__(self, filename, content: str):
        super().__init__(filename)
        self._content = content.encode("utf-8")
        self._offset = 0

    async def read(self, size=-1):
        if self._offset >= len(self._content):
            return b""
        end = len(self._content) if size < 0 else self._offset + size
        chunk = self._content[self._offset : end]
        self._offset += len(chunk)
        return chunk


class ClosingUpload(FakeUpload):
    def __init__(self, filename, *, close_error=None):
        super().__init__(filename)
        self.close_error = close_error

    async def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def test_safe_filename_accepts_txt_and_rejects_pdf():
    assert api_server.safe_filename("Module 1.txt") == "Module_1.txt"
    with pytest.raises(ValueError, match="TXT module reports"):
        api_server.safe_filename("Module 1.pdf")


def test_structured_module_number_uses_embedded_metadata(tmp_path):
    source = tmp_path / "misleading-M9.txt"
    source.write_text(structured_content("02", source.name), encoding="utf-8")

    assert _REAL_STRUCTURED_MODULE_NUMBER(source, "CPE0021") == "02"


def test_structured_module_number_rejects_course_mismatch(tmp_path):
    source = tmp_path / "module.txt"
    source.write_text(structured_content("1", source.name), encoding="utf-8")

    with pytest.raises(ValueError, match="course_code does not match"):
        _REAL_STRUCTURED_MODULE_NUMBER(source, "OTHER")


def test_find_module_number_searches_metadata_and_labeled_fallback():
    assert api_server.find_module_number(
        "[MODULE]\nformat_version: 1\nmodule_number: 07\n[/MODULE]\n"
    ) == "07"
    assert api_server.find_module_number("Lesson title\nModule Number: 12\n") == "12"

    with pytest.raises(ValueError, match="declare module_number"):
        api_server.find_module_number("Lesson without a module label")


def test_canonical_module_filename_uses_payload_course_and_discovered_number():
    assert api_server.canonical_module_filename("CPE 0021", "07") == "CPE_0021_M7.txt"


def test_publish_module_upload_renames_the_staged_file(tmp_path):
    staged = tmp_path / ".incoming.tmp"
    staged.write_text("module", encoding="utf-8")

    published = api_server.publish_module_upload(staged, "CPE0021", "07")

    assert published == tmp_path / "CPE0021_M7.txt"
    assert published.read_text(encoding="utf-8") == "module"
    assert not staged.exists()


def test_process_endpoint_requires_course_code_in_multipart_payload():
    schema = api_server.app.openapi()
    request_schema = schema["paths"]["/process"]["post"]["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    schema_name = request_schema["$ref"].rsplit("/", 1)[-1]
    body = schema["components"]["schemas"][schema_name]

    assert body["required"] == ["courseCode", "files"]
    assert set(body["properties"]) == {"courseCode", "files"}
    assert "/process/{course_code}" not in schema["paths"]


@pytest.mark.parametrize("filename", (
    "arbitrary-original-name.txt",
    "CS0003_M2.txt",
    "CS0003-M2-notes.txt",
    "CS0003-Mtwo.txt",
    "CS0003 -M2.txt",
))
def test_process_discovers_module_number_and_renames_upload(
    monkeypatch, tmp_path, filename
):
    upload = ContentUpload(
        filename,
        structured_content("02", filename),
    )
    captured = []

    def fake_batch(items, **kwargs):
        captured.extend(items)
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(
        api_server,
        "structured_module_number",
        _REAL_STRUCTURED_MODULE_NUMBER,
    )
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE0021", [upload]))

    renamed = tmp_path / "uploads" / "CPE0021" / "CPE0021_M2.txt"
    assert response["errors"] == []
    assert renamed.is_file()
    assert captured[0].args.input == renamed
    assert captured[0].args.module_number == "02"


@pytest.mark.parametrize("payload_course", ("", "WRONG999"))
@pytest.mark.parametrize("declared_module", ("9", "Not Specified"))
def test_process_uses_course_and_module_from_exact_filename(
    monkeypatch, tmp_path, payload_course, declared_module
):
    upload = ContentUpload(
        "CS0003-M2.txt",
        """Module #: 9
Module Title: Algorithms

Slide 1:
{
Title:
Search algorithms
Content:
Binary search repeatedly halves a sorted search interval.
Image/Diagram Description:
Not Specified
}
Brief Explanation:
This slide explains binary search.
""".replace("Module #: 9", f"Module #: {declared_module}"),
    )
    captured = []

    def fake_batch(items, **kwargs):
        captured.extend(items)
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files(payload_course, [upload]))

    renamed = tmp_path / "uploads" / "CS0003" / "CS0003_M2.txt"
    assert response["errors"] == []
    assert renamed.is_file()
    assert captured[0].args.input == renamed
    assert captured[0].args.course_code == "CS0003"
    assert captured[0].args.module_number == "2"
    status = api_server._PIPELINE_STATUS.snapshot()
    assert status["modules"][0]["courseCode"] == "CS0003"


@pytest.mark.parametrize("payload_course", ("COE0041", "ECE0099"))
def test_process_accepts_nine_slide_reports_with_course_abbreviation(
    monkeypatch, tmp_path, payload_course
):
    def report(number):
        return f"""Module #: Not Specified
Module Title: BASIC ELECTRICAL ENGINEERING

Slide 1:
{{
Title:
BASIC ELECTRICAL ENGINEERING
Content:
Course: BASICEE
Image/Diagram Description:
Not Specified
}}
Brief Explanation:
This slide introduces the course.

Slide 2:
{{
Title:
MODULE {number}
Content:
Not Specified
Image/Diagram Description:
Not Specified
}}
Brief Explanation:
This slide introduces the module.

Slide 3:
{{
Title:
Frequency
Content:
Frequency is the number of cycles per second.
Image/Diagram Description:
Not Specified
}}
Brief Explanation:
This slide defines frequency.
"""

    uploads = [
        ContentUpload(f"EDITH-{number}.txt", report(number))
        for number in range(1, 10)
    ]
    captured = []

    def fake_batch(items, **kwargs):
        captured.extend(items)
        return BatchResult(outputs=(), errors=())

    monkeypatch.setattr(
        api_server, "structured_module_number", _REAL_STRUCTURED_MODULE_NUMBER
    )
    monkeypatch.setattr(api_server, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api_server, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files(payload_course, uploads))

    assert response["errors"] == []
    assert response["courseCode"] == payload_course
    assert [item.args.module_number for item in captured] == [
        str(number) for number in range(1, 10)
    ]
    assert all(item.args.course_code == payload_course for item in captured)
    for number in range(1, 10):
        saved = (
            tmp_path
            / "uploads"
            / payload_course
            / f"{payload_course}_M{number}.txt"
        )
        assert saved.read_text(encoding="utf-8") == report(number)


def test_configure_api_storage_routes_portable_work_into_adjacent_data(tmp_path):
    original = build_paths(api_server.PROJECT_DIR, portable=False)
    portable = build_paths(tmp_path, portable=True)
    try:
        api_server.configure_api_storage(portable)

        args = api_server.pipeline_args(Path("module.txt"), "CPE", "1")
        assert api_server.UPLOAD_DIR == tmp_path / "data" / "uploads"
        assert args.output_root == tmp_path / "data" / "pipeline_output"
        assert args.model_dir == tmp_path / "models"
        assert args.rebel_model == tmp_path / "models" / "rebel-large"
        assert args.portable is True
    finally:
        api_server.configure_api_storage(original)


def test_configure_api_runtime_sets_pipeline_device_options():
    try:
        api_server.configure_api_runtime(n_gpu_layers=0, kg_device="cpu")
        args = api_server.pipeline_args(Path("module.txt"), "CPE", "1")
        assert args.n_gpu_layers == 0
        assert args.kg_device == "cpu"
    finally:
        api_server.configure_api_runtime(n_gpu_layers=-1, kg_device="auto")


def test_process_files_saves_all_uploads_then_runs_one_batch(monkeypatch, tmp_path):
    uploads = [FakeUpload("CPE-M1.txt"), FakeUpload("CPE-M2.txt")]
    events = []

    async def fake_save(upload, course_code):
        events.append(("save", upload.filename))
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        events.append(("batch", tuple(item.filename for item in items)))
        return BatchResult(
            outputs=tuple(path for item in items for path in item.paths.flashcard_parts),
            errors=(),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert events == [
        ("save", "CPE-M1.txt"),
        ("save", "CPE-M2.txt"),
        ("batch", ("CPE-M1.txt", "CPE-M2.txt")),
    ]
    assert len(response["outputs"]) == 4
    modules = api_server.pipeline_status(Response())["modules"]
    assert all(len(module["outputs"]) == 2 for module in modules)
    assert all(module["output"] == module["outputs"][0] for module in modules)
    assert all(upload.closed for upload in uploads)


def test_duplicate_upload_names_get_distinct_completed_status(monkeypatch, tmp_path):
    uploads = [FakeUpload("same.txt"), FakeUpload("same.txt")]
    saved = iter((1, 2))

    async def fake_save(upload, course_code):
        number = next(saved)
        path = tmp_path / f"staged-{number}.txt"
        path.write_text(f"module {number}", encoding="utf-8")
        return path

    def module_number(source, course_code):
        return Path(source).stem.rsplit("-", 1)[-1]

    def fake_batch(items, **kwargs):
        return BatchResult(
            outputs=tuple(path for item in items for path in item.paths.flashcard_parts),
            errors=(),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "structured_module_number", module_number)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert len(response["outputs"]) == 4
    modules = api_server.pipeline_status(Response())["modules"]
    assert [module["moduleNumber"] for module in modules] == ["1", "2"]
    assert all(len(module["outputs"]) == 2 for module in modules)
    assert all(module["output"] == module["outputs"][0] for module in modules)


def test_duplicate_upload_name_failure_targets_its_own_status(monkeypatch, tmp_path):
    uploads = [FakeUpload("same.txt"), FakeUpload("same.txt")]
    saved = iter((1, 2))

    async def fake_save(upload, course_code):
        number = next(saved)
        path = tmp_path / f"staged-{number}.txt"
        path.write_text(f"module {number}", encoding="utf-8")
        return path

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(
        api_server,
        "structured_module_number",
        lambda source, course_code: Path(source).stem.rsplit("-", 1)[-1],
    )

    def fake_batch(items, **kwargs):
        return BatchResult(
            outputs=items[1].paths.flashcard_parts,
            errors=(
                {
                    "file": items[0].filename,
                    "error": "first upload failed",
                    "uploadIndex": items[0].request_index,
                },
            ),
        )

    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))
    modules = api_server.pipeline_status(Response())["modules"]

    assert response["errors"] == [{"file": "same.txt", "error": "first upload failed"}]
    assert [module["state"] for module in modules] == ["failed", "completed"]
    assert modules[0]["outputs"] == []
    assert len(modules[1]["outputs"]) == 2


def test_save_failure_does_not_prevent_other_files_from_batching(monkeypatch, tmp_path):
    uploads = [FakeUpload("bad.txt"), FakeUpload("good.txt")]
    captured = []

    async def fake_save(upload, course_code):
        if upload.filename == "bad.txt":
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

    assert captured == ["good.txt"]
    assert response["errors"] == [{"file": "bad.txt", "error": "disk write failed"}]
    assert all(upload.closed for upload in uploads)


def test_batch_runtime_failure_becomes_ordered_per_file_errors(monkeypatch, tmp_path):
    uploads = [FakeUpload("CPE-M1.txt"), FakeUpload("CPE-M2.txt")]

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
            {"file": "CPE-M1.txt", "error": "model load failed"},
            {"file": "CPE-M2.txt", "error": "model load failed"},
        ],
        "courseCode": "CPE",
        "processUrl": "/process",
    }
    assert all(upload.closed for upload in uploads)


def test_process_files_runs_batch_off_the_event_loop_thread(monkeypatch, tmp_path):
    upload = FakeUpload("CPE-M1.txt")
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
    args = api_server.pipeline_args(Path("module.txt"), "CPE", "1")

    assert args.n_ctx == api_server.DEFAULT_N_CTX
    assert args.kg_batch_size == 1
    assert args.kg_num_beams == 1
    assert args.skip_final_review is True
    assert args.timeout == 0


def test_status_endpoint_is_idle_before_any_pipeline_work():
    response = Response()

    assert api_server.pipeline_status(response) == {
        "status": "idle",
        "active": None,
        "queue": [],
        "modules": [],
        "summary": {
            "total": 0,
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        },
    }
    assert response.headers["cache-control"] == "no-store"


def test_status_endpoint_shows_active_progress_and_module_queue(monkeypatch, tmp_path):
    uploads = [FakeUpload("CPE-M1.txt"), FakeUpload("CPE-M2.txt")]
    batch_started = threading.Event()
    release_batch = threading.Event()

    async def fake_save(upload, course_code):
        path = tmp_path / upload.filename
        path.write_bytes(b"module")
        return path

    def fake_batch(items, **kwargs):
        report = kwargs["progress"]
        report(
            api_server.BatchProgressEvent(
                filename=items[0].filename,
                module_number="1",
                state="processing",
                stage="flashcards",
                progress_percent=72,
                message="Generating cluster 10/20: Memory",
            )
        )
        batch_started.set()
        assert release_batch.wait(timeout=2)
        return BatchResult(
            outputs=tuple(path for item in items for path in item.paths.flashcard_parts),
            errors=(),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    async def exercise():
        task = asyncio.create_task(api_server.process_files("CPE", uploads))
        while not batch_started.is_set():
            await asyncio.sleep(0)
        status = api_server.pipeline_status(Response())
        release_batch.set()
        await asyncio.wait_for(task, timeout=2)
        return status

    status = asyncio.run(exercise())

    assert status["status"] == "processing"
    assert status["active"]["filename"] == "CPE_M1.txt"
    assert status["active"]["stage"] == "flashcards"
    assert status["active"]["progressPercent"] == 72
    assert [module["filename"] for module in status["queue"]] == ["CPE_M2.txt"]
    assert status["queue"][0]["position"] == 1
    assert status["summary"] == {
        "total": 2,
        "queued": 1,
        "processing": 1,
        "completed": 0,
        "failed": 0,
    }

    completed = api_server.pipeline_status(Response())
    assert completed["status"] == "idle"
    assert completed["summary"]["completed"] == 2
    assert all(module["output"] for module in completed["modules"])
    assert all(len(module["outputs"]) == 2 for module in completed["modules"])


def test_all_save_failures_still_invoke_empty_batch(monkeypatch):
    uploads = [FakeUpload("bad-1.txt"), FakeUpload("bad-2.txt")]
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
        {"file": "bad-1.txt", "error": "disk write failed"},
        {"file": "bad-2.txt", "error": "disk write failed"},
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


def test_later_canonical_module_filename_collision_is_an_ordered_file_error(
    monkeypatch, tmp_path
):
    uploads = [FakeUpload("lecture .txt"), FakeUpload("lecture_.txt"), FakeUpload("ok.txt")]
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

    assert saved == ["lecture .txt", "lecture_.txt", "ok.txt"]
    assert batches == [("lecture .txt", "ok.txt")]
    assert response["errors"] == [
        {
            "file": "lecture_.txt",
            "error": "module_number collides with an earlier upload: lecture_",
        }
    ]


def test_later_discovered_module_collision_is_an_ordered_file_error(
    monkeypatch, tmp_path
):
    uploads = [FakeUpload("lecture-.txt"), FakeUpload("lecture.txt"), FakeUpload("ok.txt")]
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

    assert saved == ["lecture-.txt", "lecture.txt", "ok.txt"]
    assert response["errors"] == [
        {
            "file": "lecture.txt",
            "error": "module_number collides with an earlier upload: lecture",
        }
    ]


def test_api_errors_follow_the_original_upload_order_across_phases(
    monkeypatch, tmp_path
):
    uploads = [FakeUpload("batch.txt"), FakeUpload("save.txt")]

    async def fake_save(upload, course_code):
        if upload.filename == "save.txt":
            raise OSError("save failed")
        path = tmp_path / upload.filename
        path.write_bytes(b"pdf")
        return path

    def fake_batch(items, **kwargs):
        return BatchResult(
            outputs=(),
            errors=({"file": "batch.txt", "error": "batch failed"},),
        )

    monkeypatch.setattr(api_server, "save_upload", fake_save)
    monkeypatch.setattr(api_server, "run_batch", fake_batch)

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert response["errors"] == [
        {"file": "batch.txt", "error": "batch failed"},
        {"file": "save.txt", "error": "save failed"},
    ]


def test_all_uploads_are_closed_after_a_close_error(monkeypatch, tmp_path):
    uploads = [
        ClosingUpload("first.txt", close_error=RuntimeError("close failed")),
        ClosingUpload("second.txt"),
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
            outputs=(), errors=({"file": "first.txt", "error": "batch failed"},)
        ),
    )

    response = asyncio.run(api_server.process_files("CPE", uploads))

    assert response["errors"] == [
        {"file": "first.txt", "error": "batch failed"},
        {"file": "first.txt", "error": "close failed"},
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
        if upload.filename == "first.txt":
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
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=1)
        )
        first = asyncio.create_task(
            api_server.process_files("CPE", [FakeUpload("first.txt")])
        )
        await first_started.wait()
        second = asyncio.create_task(
            api_server.process_files("CPE", [FakeUpload("second.txt")])
        )
        await asyncio.sleep(0)
        assert events == [("save", "first.txt")]
        release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    asyncio.run(exercise())

    assert events == [
        ("save", "first.txt"),
        ("batch", "first.txt"),
        ("save", "second.txt"),
        ("batch", "second.txt"),
    ]


def test_cancelled_request_lock_waiter_cannot_leak_the_lock():
    async def exercise():
        assert api_server._REQUEST_LOCK.acquire(blocking=False)
        try:
            waiter = asyncio.create_task(api_server.acquire_request_lock())
            await asyncio.sleep(0)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            api_server._REQUEST_LOCK.release()

        barrier = api_server._REQUEST_LOCK_EXECUTOR.submit(lambda: None)
        await asyncio.wait_for(asyncio.wrap_future(barrier), timeout=2)

        assert api_server._REQUEST_LOCK.acquire(blocking=False)
        api_server._REQUEST_LOCK.release()

    asyncio.run(exercise())


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

    asyncio.run(api_server.process_files("CPE-A", [FakeUpload("Module 1.txt")]))
    asyncio.run(api_server.process_files("CPE-B", [FakeUpload("Module 1.txt")]))

    assert captured[0] != captured[1]
    assert captured == [
        tmp_path / "output" / "CPE-A" / "CPE-A_M1",
        tmp_path / "output" / "CPE-B" / "CPE-B_M1",
    ]
