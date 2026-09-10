# Low-Resource Batch Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process multi-PDF requests as a resource-aware staged batch using Qwen Q5_K_M and low-memory defaults while preserving resumability, per-file error isolation, and the exact 100-card output contract.

**Architecture:** Refactor the three existing stages so callers may inject already-loaded Qwen or REBEL runtimes, then add an in-process batch orchestrator that groups work by stage and owns those runtimes. The API will save all uploads first and invoke that orchestrator off the async event loop; standalone commands retain their current interfaces and load dependencies when none are injected.

**Tech Stack:** Python 3.12, FastAPI, llama-cpp-python, Hugging Face Transformers, PyTorch, pytest

**Spec:** `docs/superpowers/specs/2026-09-03-low-resource-batch-performance-design.md`

## Global Constraints

- Use `Qwen/Qwen2.5-3B-Instruct-GGUF` file `qwen2.5-3b-instruct-q5_k_m.gguf`.
- Use an 8,192-token Qwen context by default.
- Keep heavyweight inference sequential; do not process PDFs concurrently.
- API fast mode uses REBEL batch size 1, one beam, and no optional model-assisted final review.
- Deterministic schema, grounding, duplicate, UUID, cluster-count, and 100-question validation always remains enabled.
- Preserve endpoint fields, response keys, output locations, module-number parsing, standalone commands, and resumable artifact behavior.
- A failed module must not stop other modules in the same request.
- Existing Q8 files must not be deleted.

---

### Task 1: Q5 model and low-memory Qwen defaults

**Files:**
- Modify: `local_qwen.py`
- Modify: `main.py`
- Modify: `slides_pdf_to_txt.py`
- Modify: `pipeline.py`
- Test: `tests/test_local_qwen.py`
- Test: `tests/test_main.py`
- Test: `tests/test_slides_pdf_to_txt.py`
- Test: `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `MODEL_FILENAME = "qwen2.5-3b-instruct-q5_k_m.gguf"`.
- Produces: `DEFAULT_N_CTX = 8192` used by every Qwen CLI and pipeline default.
- Produces: `LocalQwenBackend.close() -> None`, which releases the underlying llama runtime when supported.

- [ ] **Step 1: Write failing tests for Q5, 8K defaults, and cleanup**

Update the local model assertion and add:

```python
def test_backend_defaults_to_low_memory_context_and_can_close(monkeypatch, tmp_path):
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    backend = LocalQwenBackend(tmp_path / "model.gguf")

    assert captured["n_ctx"] == 8192
    backend.close()
    assert backend._llm.closed is True
```

Assert `main.parse_args(["graph.json"]).n_ctx == 8192`,
`slides_pdf_to_txt.parse_args(["module.pdf"]).n_ctx == 8192`, and
`pipeline.parse_args(["module.pdf", "--course-code", "CPE0021", "--module-number", "1"]).n_ctx == 8192`. Update download expectations to
`qwen2.5-3b-instruct-q5_k_m.gguf`.

- [ ] **Step 2: Run the focused tests and verify the expected failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_local_qwen.py tests/test_main.py tests/test_slides_pdf_to_txt.py tests/test_pipeline_runner.py -q
```

Expected: failures report the old Q8 filename, 32,768-token defaults, and missing
`close()` method.

- [ ] **Step 3: Implement the shared Q5 and context defaults**

Define in `local_qwen.py`:

```python
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILENAME = "qwen2.5-3b-instruct-q5_k_m.gguf"
DEFAULT_N_CTX = 8192
```

Use `DEFAULT_N_CTX` in `LocalQwenBackend.__init__` and import it into both Qwen
CLIs and `pipeline.py`. Implement cleanup defensively:

```python
def close(self) -> None:
    close = getattr(self._llm, "close", None)
    if callable(close):
        close()
```

- [ ] **Step 4: Re-run the focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add local_qwen.py main.py slides_pdf_to_txt.py pipeline.py tests/test_local_qwen.py tests/test_main.py tests/test_slides_pdf_to_txt.py tests/test_pipeline_runner.py
git commit -m "perf: use Q5 low-memory defaults"
```

---

### Task 2: Injectable Qwen-backed stages

**Files:**
- Modify: `slides_pdf_to_txt.py`
- Modify: `main.py`
- Test: `tests/test_slides_pdf_to_txt.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Produces: `slides_pdf_to_txt.run(args, *, backend: CompletionBackend | None = None) -> Path`.
- Produces: `main.run(args, *, backend: ChatBackend | None = None) -> Path | None`.
- Behavior: injected backends bypass `ensure_model` and `LocalQwenBackend`; standalone callers still load their own backend.

- [ ] **Step 1: Write failing tests proving injected backends are reused**

Add to `tests/test_slides_pdf_to_txt.py`:

```python
def test_run_reuses_injected_backend(tmp_path, monkeypatch):
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    output = tmp_path / "structured.txt"
    shared_backend = object()
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "extract_pdf_pages",
        lambda *a, **k: (ExtractedPage(1, "Page text", "text"),),
    )
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "ensure_model",
        lambda path: pytest.fail("an injected backend must bypass model loading"),
    )
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "normalize_document",
        lambda backend, *a, **k: backend,
    )
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "render_structured_module",
        lambda value: "shared\n" if value is shared_backend else "wrong\n",
    )
    args = slides_pdf_to_txt.parse_args([str(source), "--output", str(output)])

    assert slides_pdf_to_txt.run(args, backend=shared_backend) == output
    assert output.read_text(encoding="utf-8") == "shared\n"
```

Add to `tests/test_main.py`:

```python
def test_run_reuses_injected_backend(tmp_path, monkeypatch, valid_clusters):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        '{"metadata": {}, "nodes": [], "edges": '
        '[{"id":"e1","subject":"a","relation":"is","object":"b"}]}',
        encoding="utf-8",
    )
    output_path = tmp_path / "cards.txt"
    shared_backend = object()
    monkeypatch.setattr(
        main,
        "ensure_model",
        lambda path: pytest.fail("an injected backend must bypass model loading"),
    )

    class FakePipeline:
        def __init__(self, backend, config, **kwargs):
            assert backend is shared_backend

        def run(self, identity, facts):
            return valid_clusters

    monkeypatch.setattr(main, "FlashcardPipeline", FakePipeline)
    args = main.parse_args(
        [
            str(graph_path),
            "--course-code", "CPE0021",
            "--module-number", "1",
            "--output", str(output_path),
        ]
    )

    assert main.run(args, backend=shared_backend) == output_path
```

- [ ] **Step 2: Run tests and verify they fail with an unexpected keyword argument**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_slides_pdf_to_txt.py tests/test_main.py -q
```

Expected: FAIL because neither `run` function accepts `backend`.

- [ ] **Step 3: Add optional keyword-only backend injection**

Change each run signature to accept `backend=None`. Only call `ensure_model`
and construct `LocalQwenBackend` when `backend is None`. Do not close an injected
backend because its lifetime belongs to the caller. Preserve existing exception
messages for standalone setup failures.

- [ ] **Step 4: Re-run focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add slides_pdf_to_txt.py main.py tests/test_slides_pdf_to_txt.py tests/test_main.py
git commit -m "refactor: allow shared Qwen stage runtime"
```

---

### Task 3: Importable and reusable REBEL stage

**Files:**
- Create: `text_extractor.py`
- Modify: `text-extractor.py`
- Modify: `pipeline.py`
- Test: `tests/test_text_extractor_structured.py`
- Test: `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `RebelRuntime(tokenizer, model, device)` dataclass.
- Produces: `load_runtime(model_name: str, device: str) -> RebelRuntime`.
- Produces: `run(args: argparse.Namespace, *, runtime: RebelRuntime | None = None) -> tuple[Path, Path]`.
- Produces: `release_runtime(runtime: RebelRuntime) -> None`.
- Compatibility: `python text-extractor.py module.txt --output-dir output` remains valid through a thin wrapper.

- [ ] **Step 1: Write failing tests for runtime injection and fast parameters**

Import `text_extractor` normally and add:

```python
def test_run_reuses_injected_runtime(tmp_path, monkeypatch):
    source = tmp_path / "module.txt"
    source.write_text(structured_text(), encoding="utf-8")
    output_dir = tmp_path / "graph"
    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            return [1]

        def decode(self, ids, *, skip_special_tokens):
            return "processor contains ALU"

    tokenizer = FakeTokenizer()
    runtime = text_extractor.RebelRuntime(tokenizer, object(), "cpu")
    monkeypatch.setattr(
        text_extractor,
        "load_runtime",
        lambda *a, **k: pytest.fail("injected runtime must be reused"),
    )
    triples = [{
        "subject": "processor",
        "relation": "contains",
        "object": "ALU",
        "evidence": [],
    }]
    monkeypatch.setattr(text_extractor, "extract_relations", lambda *a, **k: triples)
    args = text_extractor.parse_args(
        [str(source), "--output-dir", str(output_dir), "--batch-size", "1", "--num-beams", "1"]
    )

    json_path, csv_path = text_extractor.run(args, runtime=runtime)

    assert json_path.is_file()
    assert csv_path.is_file()
    assert args.batch_size == 1
    assert args.num_beams == 1
```

Add a pipeline command assertion that `--batch-size 1` and `--num-beams 1` are
forwarded when those parsed values are supplied.

- [ ] **Step 2: Run focused tests and verify import/interface failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_text_extractor_structured.py tests/test_pipeline_runner.py -q
```

Expected: FAIL because `text_extractor.py`, `RebelRuntime`, and reusable `run`
do not exist.

- [ ] **Step 3: Move implementation behind an importable module**

Move the implementation from `text-extractor.py` to `text_extractor.py`. Change
`parse_args` to accept `argv: Sequence[str] | None = None`. Extract dependency
loading into `load_runtime`, stage work into `run`, and cleanup into:

```python
def release_runtime(runtime: RebelRuntime) -> None:
    del runtime.model
    gc.collect()
    if runtime.device == "cuda":
        import torch
        torch.cuda.empty_cache()
```

The thin compatibility wrapper is:

```python
from text_extractor import *

if __name__ == "__main__":
    main()
```

Add `--kg-batch-size` and `--kg-num-beams` to `pipeline.parse_args` with current
standalone defaults `4` and `3`, then forward them to the existing stage command.

- [ ] **Step 4: Re-run focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add text_extractor.py text-extractor.py pipeline.py tests/test_text_extractor_structured.py tests/test_pipeline_runner.py
git commit -m "refactor: expose reusable graph extraction stage"
```

---

### Task 4: Resource-aware staged batch orchestrator

**Files:**
- Create: `batch_pipeline.py`
- Create: `tests/batch_helpers.py`
- Create: `tests/test_batch_pipeline.py`

**Interfaces:**
- Produces: `BatchItem(filename: str, args: argparse.Namespace, paths: PipelinePaths)`.
- Produces: `BatchResult(outputs: tuple[Path, ...], errors: tuple[dict[str, str], ...])`.
- Produces: `BatchDependencies` containing `qwen_loader(args)`, `rebel_loader(args)`, `normalize_stage(item, backend)`, `graph_stage(item, runtime)`, `flashcard_stage(item, backend)`, and `monotonic()` callables.
- Produces: `run_batch(items: Sequence[BatchItem], *, dependencies: BatchDependencies = PRODUCTION_DEPENDENCIES, timeout_seconds: float | None = None) -> BatchResult`.
- Produces for later tests: `tests.batch_helpers.make_items`, `materialize`, `fake_dependencies`, and `counting_dependencies`.

- [ ] **Step 1: Write a failing test for stage grouping and runtime reuse**

In `tests/batch_helpers.py`, import `contextmanager`, `Counter`, `json`, `time`,
`pipeline`, the public batch types, and structured-module factories. Define
these helpers so every test uses real artifact validation:

```python
def make_items(tmp_path, *names):
    items = []
    for number, name in enumerate(names, start=1):
        pdf = tmp_path / name
        pdf.write_bytes(b"pdf")
        args = pipeline.parse_args([
            str(pdf),
            "--course-code", "CPE0021",
            "--module-number", str(number),
            "--output-root", str(tmp_path / "output"),
            "--kg-batch-size", "1",
            "--kg-num-beams", "1",
            "--skip-final-review",
            "--timeout", "0",
        ])
        items.append(BatchItem(name, args, pipeline.pipeline_paths(pdf, args.output_root)))
    return tuple(items)


def materialize(item, stage):
    if stage == "normalize":
        item.paths.structured_text.parent.mkdir(parents=True, exist_ok=True)
        item.paths.structured_text.write_text(structured_content(), encoding="utf-8")
    elif stage == "graph":
        item.paths.graph_dir.mkdir(parents=True, exist_ok=True)
        item.paths.graph_json.write_text(
            json.dumps({
                "metadata": {"module_number": item.args.module_number},
                "nodes": [],
                "edges": [{
                    "id": "e1",
                    "subject": "processor",
                    "relation": "contains",
                    "object": "ALU",
                }],
            }),
            encoding="utf-8",
        )
    else:
        item.paths.flashcards.write_text(
            f"Module {item.args.module_number}.1\n"
            f"Module {item.args.module_number}.2\n",
            encoding="utf-8",
        )


def fake_dependencies(events, qwen_loader, rebel_loader, *, fail_normalize=None, monotonic=None):
    def normalize(item, backend):
        events.append(("normalize", item.filename, backend))
        if item.filename == fail_normalize:
            raise RuntimeError("normalization failed")
        materialize(item, "normalize")

    def graph(item, runtime):
        events.append(("graph", item.filename, runtime))
        materialize(item, "graph")

    def flashcards(item, backend):
        events.append(("flashcards", item.filename, backend))
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        normalize_stage=normalize,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=monotonic or time.monotonic,
    )


def counting_dependencies(counters: Counter):
    @contextmanager
    def qwen_loader(args):
        counters["qwen_load"] += 1
        yield object()

    @contextmanager
    def rebel_loader(args):
        counters["rebel_load"] += 1
        yield object()

    def normalize(item, backend):
        counters["normalize"] += 1
        materialize(item, "normalize")

    def graph(item, runtime):
        counters["graph"] += 1
        materialize(item, "graph")

    def flashcards(item, backend):
        counters["flashcards"] += 1
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        normalize_stage=normalize,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=time.monotonic,
    )
```

Put `structured_content()` in `tests/batch_helpers.py`, copying its complete
`StructuredModule` and `StructuredSlide` construction from
`tests/test_pipeline_runner.py`. In `tests/test_batch_pipeline.py`, import the
four helpers plus `BatchResult`, `run_batch`, `contextmanager`, and `time`.
Then record stage order and runtime identity:

```python
def test_batch_groups_stages_and_reuses_each_runtime(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []

    @contextmanager
    def qwen_loader(args):
        phase_backend = object()
        events.append(("qwen-load", phase_backend))
        yield phase_backend
        events.append(("qwen-release", phase_backend))

    @contextmanager
    def rebel_loader(args):
        runtime = object()
        events.append(("rebel-load", runtime))
        yield runtime
        events.append(("rebel-release", runtime))

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            qwen_loader=qwen_loader,
            rebel_loader=rebel_loader,
        ),
    )

    assert len(result.outputs) == 2
    assert not result.errors
    assert [event[0] for event in events] == [
        "qwen-load", "normalize", "normalize", "qwen-release",
        "rebel-load", "graph", "graph", "rebel-release",
        "qwen-load", "flashcards", "flashcards", "qwen-release",
    ]
    assert events[1][2] is events[2][2]
    assert events[5][2] is events[6][2]
    assert events[9][2] is events[10][2]
```

- [ ] **Step 2: Add failing tests for resume, failure isolation, and timeout accounting**

Add these concrete tests below the shared-runtime test:

```python
def test_batch_reuses_valid_artifacts_without_loading_models(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    for item in items:
        for stage in ("normalize", "graph", "flashcards"):
            materialize(item, stage)

    @contextmanager
    def forbidden_loader(args):
        raise AssertionError("valid artifacts must not load a model")
        yield

    dependencies = fake_dependencies([], forbidden_loader, forbidden_loader)

    result = run_batch(items, dependencies=dependencies)

    assert result.outputs == tuple(item.paths.flashcards for item in items)
    assert not result.errors


def test_failed_module_does_not_stop_other_module(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []

    @contextmanager
    def loader(args):
        yield object()

    result = run_batch(
        items,
        dependencies=fake_dependencies(
            events,
            loader,
            loader,
            fail_normalize="one.pdf",
        ),
    )

    assert result.outputs == (items[1].paths.flashcards,)
    assert result.errors == ({"pdf": "one.pdf", "error": "normalization failed"},)
    assert ("flashcards", "two.pdf") in [event[:2] for event in events]
    assert ("graph", "one.pdf") not in [event[:2] for event in events]


def test_timeout_counts_only_each_modules_active_work(tmp_path):
    items = make_items(tmp_path, "one.pdf", "two.pdf")
    events = []
    clock_values = iter([0, 4, 100, 101, 200, 202, 300, 301, 400, 401])

    @contextmanager
    def loader(args):
        yield object()

    dependencies = fake_dependencies(
        events,
        loader,
        loader,
        monotonic=lambda: next(clock_values),
    )

    result = run_batch(items, dependencies=dependencies, timeout_seconds=5)

    assert result.outputs == (items[1].paths.flashcards,)
    assert result.errors == ({
        "pdf": "one.pdf",
        "error": "module exceeded 5-second active-processing timeout",
    },)
    assert ("flashcards", "one.pdf") not in [event[:2] for event in events]
```

- [ ] **Step 3: Run the new test module and verify missing-module failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_batch_pipeline.py -q
```

Expected: FAIL because `batch_pipeline` does not exist.

- [ ] **Step 4: Implement the minimal staged state machine**

For every item, compute these flags once:

```python
needs_normalize = args.force or not _valid_structured_text(paths.structured_text)
needs_graph = needs_normalize or args.force or not _valid_graph(paths.graph_json)
needs_flashcards = (
    needs_graph
    or args.force
    or not _valid_flashcards(paths.flashcards, args.module_number)
)
```

Run three sequential loops over active items. Enter a loader context only when
at least one active item needs that stage. Record elapsed time around the active
item's own stage call, subtract it from that item's optional budget, and fail it
before the next call when the remaining budget is non-positive. Catch
`OSError`, `PdfExtractionError`, `SlideNormalizationError`, `GraphInputError`,
`GenerationError`, `RuntimeError`, and `ValueError` per item. Do not catch
`KeyboardInterrupt` or `SystemExit`.

Production dependency loaders must build Qwen with the first pending item's
common model configuration, close it in `finally`, and call
`release_runtime` for REBEL in `finally`. Validate that batch items share model
directory, Qwen context, GPU-layer, seed, REBEL device, batch-size, and beam
settings before loading shared dependencies.

- [ ] **Step 5: Re-run batch tests**

Run the command from Step 3. Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add batch_pipeline.py tests/batch_helpers.py tests/test_batch_pipeline.py
git commit -m "feat: add staged low-resource batch runner"
```

---

### Task 5: Integrate the staged batch with FastAPI

**Files:**
- Modify: `api_server.py`
- Create: `tests/test_api_server.py`

**Interfaces:**
- Consumes: `batch_pipeline.BatchItem` and `batch_pipeline.run_batch`.
- Behavior: the endpoint saves all uploads first, calls `run_batch` once through `asyncio.to_thread`, and preserves the existing JSON response shape.
- Fast settings: `n_ctx=8192`, `kg_batch_size=1`, `kg_num_beams=1`, `skip_final_review=True`, `timeout=0`.

- [ ] **Step 1: Write a failing API orchestration test**

Import `asyncio`, `pytest`, `api_server`, and `BatchResult`. Define this upload
double, then patch `save_upload` and `run_batch`:

```python
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
```

Add a second test proving a save failure becomes a per-file error while other
saved files still enter the batch:

```python
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
```

- [ ] **Step 2: Run API tests and verify the old per-file runner behavior fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api_server.py -q
```

Expected: FAIL because the endpoint does not import or call `run_batch`.

- [ ] **Step 3: Replace the per-file pipeline loop**

Build all `BatchItem` instances after saving. Set the API Namespace defaults to:

```python
n_ctx=DEFAULT_N_CTX
kg_batch_size=1
kg_num_beams=1
skip_final_review=True
timeout=0
```

Call the synchronous batch off the server event loop:

```python
batch_result = await asyncio.to_thread(run_batch, tuple(items))
```

Merge save errors with `batch_result.errors`, resolve output paths as strings,
and close every upload in one outer `finally` block.

- [ ] **Step 4: Re-run API tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add api_server.py tests/test_api_server.py
git commit -m "perf: process uploads as a staged batch"
```

---

### Task 6: Make timeouts low-end safe

**Files:**
- Modify: `pipeline.py`
- Modify: `tests/test_pipeline_runner.py`

**Interfaces:**
- Behavior: `--timeout 0` disables the deadline; any positive value retains the existing subprocess timeout behavior.
- Default: `--timeout 0`.

- [ ] **Step 1: Write failing timeout tests**

Add:

```python
def test_timeout_defaults_to_disabled(tmp_path):
    assert make_args(tmp_path).timeout == 0


def test_zero_timeout_does_not_pass_subprocess_timeout(tmp_path):
    args = make_args(tmp_path)
    paths = pipeline.pipeline_paths(args.pdf, args.output_root)
    calls = []

    def runner(command, **kwargs):
        calls.append(kwargs)
        stage = {
            "slides_pdf_to_txt.py": "pdf-to-text",
            "text-extractor.py": "knowledge-graph",
            "main.py": "flashcards",
        }[Path(command[1]).name]
        materialize(stage, paths)
        return subprocess.CompletedProcess(command, 0)

    pipeline.run(args, command_runner=runner)

    assert all("timeout" not in kwargs for kwargs in calls)
```

Retain the positive-timeout tests and assertions added by commit `e654573`.

- [ ] **Step 2: Run focused tests and verify zero is currently rejected**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_runner.py -q
```

Expected: FAIL because the default is 300 and zero raises `PipelineRunError`.

- [ ] **Step 3: Normalize zero to no deadline**

Set the parser default to `0`. Reject negative values, and convert zero to
`None` before computing a deadline:

```python
if effective_timeout is not None and effective_timeout < 0:
    raise PipelineRunError("module timeout must not be negative")
if effective_timeout == 0:
    effective_timeout = None
```

- [ ] **Step 4: Re-run focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add pipeline.py tests/test_pipeline_runner.py
git commit -m "fix: disable unsafe default module timeout"
```

---

### Task 7: Documentation, regression suite, and orchestration benchmark

**Files:**
- Modify: `README.md`
- Create: `tests/test_batch_benchmark.py`

**Interfaces:**
- Documents: staged sequential batches, Q5 download/reuse, 8K context, API fast mode, full-quality CLI switches, and disabled default timeout.
- Produces: deterministic orchestration benchmark asserting dependency-load reduction for three files.

- [ ] **Step 1: Write the benchmark regression test**

Use fake stage work and counters rather than real models:

```python
from collections import Counter

from batch_pipeline import run_batch
from tests.batch_helpers import counting_dependencies, make_items


def test_three_file_batch_loads_qwen_twice_and_rebel_once(tmp_path):
    counters = Counter()
    items = make_items(tmp_path, "one.pdf", "two.pdf", "three.pdf")

    result = run_batch(items, dependencies=counting_dependencies(counters))

    assert len(result.outputs) == 3
    assert counters["qwen_load"] == 2
    assert counters["rebel_load"] == 1
    assert counters["normalize"] == 3
    assert counters["graph"] == 3
    assert counters["flashcards"] == 3
```

This encodes the improvement over the old three-file behavior of six Qwen
loads and three REBEL loads.

- [ ] **Step 2: Run the benchmark regression test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_batch_benchmark.py -q
```

Expected: PASS using the completed orchestrator.

- [ ] **Step 3: Update user documentation**

Replace Q8/32K/five-minute statements with Q5_K_M, 8K, and no default timeout.
Explain that the endpoint accepts multiple files, processes heavy inference
sequentially, and reuses each model across a staged request. Document the API
fast defaults and show the full-quality CLI override:

```powershell
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\module.pdf" `
  --course-code CPE0021 --module-number 1 `
  --kg-batch-size 4 --kg-num-beams 3
```

State that final model review remains enabled by default for the CLI and can be
disabled with `--skip-final-review`.

- [ ] **Step 4: Run formatting and complete verification**

Run:

```powershell
git diff --check
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: no whitespace errors and all tests pass.

- [ ] **Step 5: Verify command help without loading models**

Run:

```powershell
.\.venv\Scripts\python.exe pipeline.py --help
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe slides_pdf_to_txt.py --help
.\.venv\Scripts\python.exe text-extractor.py --help
```

Expected: every command exits zero; help shows Q5/8K or the relevant low-resource
and quality controls.

- [ ] **Step 6: Commit**

```powershell
git add README.md tests/test_batch_benchmark.py
git commit -m "docs: explain low-resource staged batches"
```
