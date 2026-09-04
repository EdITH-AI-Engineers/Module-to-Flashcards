# Quality-Preserving Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 20-600-slide single and batch runs faster and resumable on a Ryzen 5/12 GB machine while automatically scaling to stronger computers without changing generation quality.

**Architecture:** Keep the existing Q5 → REBEL → Q5 stage barriers, add portable resource profiles at process boundaries, and wrap expensive extraction/normalization work with validated content-addressed caches. OCR concurrency and REBEL batching remain bounded by the resolved profile; all generated or cached content must pass the existing validators.

**Tech Stack:** Python 3.11+, pytest, llama-cpp-python 0.3.35-compatible cache API, PyMuPDF, Pillow, pytesseract, PyTorch/Transformers, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-04-quality-preserving-performance-design.md`

## Global Constraints

- Preserve Qwen2.5 3B Instruct `qwen2.5-3b-instruct-q5_k_m.gguf`.
- Preserve caller-selected temperature, seed, context, token limits, REBEL beams, retries, reviews, and validators.
- Never keep Q5 and REBEL resident concurrently.
- Default profile is `auto`; supported names are exactly `auto`, `low-memory`, and `performance`.
- `auto` uses `<4 GiB`, `4-<12 GiB`, and `>=12 GiB` available-memory tiers.
- `low-memory` uses 256 MiB Q5 cache, two OCR workers, and REBEL batch size one.
- `performance` uses 512 MiB Q5 cache, at most four OCR workers, and REBEL batch size four.
- Explicit resource overrides win field-by-field over profile defaults.
- Unknown hardware information falls back to `low-memory` behavior.
- Cache identity uses content and model identity, never an absolute project/model directory.
- Cache corruption or write failure cannot bypass validation or prevent an otherwise valid run.
- Tests must not download or load real Q5/REBEL model weights.
- Synthetic timing is not evidence of real hardware speed.

---

## File Structure

- Create `runtime_profile.py`: hardware detection, profile resolution, explicit overrides, and memory-pressure batch limiting.
- Create `normalization_cache.py`: PDF hashing, portable model identity, extraction cache, slide checkpoints, and atomic JSON writes.
- Modify `local_qwen.py`: optional bounded `LlamaRAMCache`, model identity exposure, and cache cleanup.
- Modify `pdf_ingestion.py`: caller-thread rendering plus bounded OCR worker scheduling.
- Modify `slide_normalizer.py`: validated slide checkpoint reads/writes around existing parsing.
- Modify `slides_pdf_to_txt.py`: profile resolution, extraction-cache orchestration, checkpoint-store construction, and cache statistics.
- Modify `pipeline.py`: profile/override CLI flags, portable subprocess propagation, and portable manifest settings.
- Modify `batch_pipeline.py`: resolved profile use, live REBEL batch limiting, portable manifest identity, and timing metrics.
- Modify `main.py`: profile/Q5-cache CLI support without changing flashcard quality configuration.
- Modify `api_server.py`: environment-driven profile selection and server-side timing logging.
- Modify `README.md`: profile selection, cache behavior, portability, and benchmark guidance.
- Create `tests/test_runtime_profile.py`: deterministic profile and hardware boundary coverage.
- Create `tests/test_normalization_cache.py`: cache key, portability, corruption, and atomicity coverage.
- Modify `tests/test_local_qwen.py`: prompt-cache setup/fallback/release coverage.
- Modify `tests/test_pdf_ingestion.py`: concurrency, order, memory-bound, and error coverage.
- Modify `tests/test_slide_normalizer.py`: checkpoint hit/miss/validation coverage.
- Modify `tests/test_slides_pdf_to_txt.py`: interrupted-run resume and extraction-cache integration coverage.
- Modify `tests/test_pipeline_runner.py`: flag propagation and portable manifest coverage.
- Modify `tests/test_batch_pipeline.py`: profile integration, loader barriers, and metrics coverage.
- Modify `tests/test_api_server.py`: environment profile and unchanged response-contract coverage.
- Modify `tests/test_batch_benchmark.py`: deterministic cold/resume work-count comparison.

---

### Task 1: Portable Runtime Profiles

**Files:**
- Create: `runtime_profile.py`
- Create: `tests/test_runtime_profile.py`

**Interfaces:**
- Produces: `PROFILE_NAMES: tuple[str, ...]`.
- Produces: `HardwareSnapshot(logical_cpus: int, available_memory_bytes: int | None)`.
- Produces: `RuntimeOverrides(ocr_workers: int | None = None, qwen_cache_mb: int | None = None, kg_batch_size: int | None = None, n_gpu_layers: int | None = None, kg_device: str | None = None)`.
- Produces: `RuntimeProfile(name: str, ocr_workers: int, qwen_cache_mb: int, kg_batch_size: int, n_gpu_layers: int, kg_device: str, kg_batch_size_explicit: bool)`.
- Produces: `detect_hardware() -> HardwareSnapshot`.
- Produces: `resolve_runtime_profile(name: str, *, hardware: HardwareSnapshot | None = None, overrides: RuntimeOverrides = RuntimeOverrides()) -> RuntimeProfile`.
- Produces: `limit_rebel_batch_size(requested: int, *, available_memory_bytes: int | None) -> int`.

- [ ] **Step 1: Write failing profile-resolution tests**

Create `tests/test_runtime_profile.py` with literal boundary expectations:

```python
import pytest

from runtime_profile import (
    GIB,
    HardwareSnapshot,
    RuntimeOverrides,
    limit_rebel_batch_size,
    resolve_runtime_profile,
)


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        (3 * GIB, (1, 0, 1)),
        (4 * GIB, (2, 256, 2)),
        (12 * GIB, (4, 512, 4)),
    ),
)
def test_auto_profile_uses_documented_memory_tiers(available, expected):
    profile = resolve_runtime_profile(
        "auto", hardware=HardwareSnapshot(logical_cpus=8, available_memory_bytes=available)
    )
    assert (profile.ocr_workers, profile.qwen_cache_mb, profile.kg_batch_size) == expected


def test_detection_failure_falls_back_to_low_memory():
    profile = resolve_runtime_profile(
        "auto", hardware=HardwareSnapshot(logical_cpus=6, available_memory_bytes=None)
    )
    assert (profile.ocr_workers, profile.qwen_cache_mb, profile.kg_batch_size) == (2, 256, 1)


def test_explicit_overrides_win_field_by_field_and_workers_are_cpu_capped():
    profile = resolve_runtime_profile(
        "performance",
        hardware=HardwareSnapshot(logical_cpus=2, available_memory_bytes=24 * GIB),
        overrides=RuntimeOverrides(ocr_workers=7, qwen_cache_mb=128, kg_batch_size=3),
    )
    assert profile.ocr_workers == 2
    assert profile.qwen_cache_mb == 128
    assert profile.kg_batch_size == 3
    assert profile.kg_batch_size_explicit is True


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown runtime profile"):
        resolve_runtime_profile("turbo")


@pytest.mark.parametrize(
    ("available", "expected"),
    ((None, 1), (3 * GIB, 1), (6 * GIB, 2), (16 * GIB, 4)),
)
def test_live_memory_pressure_only_lowers_rebel_batch(available, expected):
    assert limit_rebel_batch_size(4, available_memory_bytes=available) == expected
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_profile.py -q
```

Expected: collection fails because `runtime_profile` does not exist.

- [ ] **Step 3: Implement hardware detection and profile resolution**

Create `runtime_profile.py` with frozen dataclasses and this resolution logic:

```python
GIB = 1024 ** 3
PROFILE_NAMES = ("auto", "low-memory", "performance")


def _auto_controls(available: int | None) -> tuple[int, int, int]:
    if available is None:
        return 2, 256, 1
    if available < 4 * GIB:
        return 1, 0, 1
    if available < 12 * GIB:
        return 2, 256, 2
    return 4, 512, 4


def limit_rebel_batch_size(requested: int, *, available_memory_bytes: int | None) -> int:
    if requested < 1:
        raise ValueError("REBEL batch size must be at least 1")
    if available_memory_bytes is None or available_memory_bytes < 4 * GIB:
        return min(requested, 1)
    if available_memory_bytes < 12 * GIB:
        return min(requested, 2)
    return requested
```

Implement `detect_hardware` without a new dependency: use
`GlobalMemoryStatusEx` through `ctypes` on Windows and `os.sysconf` on POSIX.
Return `None` for available memory on any detection failure. Normalize
`os.cpu_count()` to at least one. Validate override values (`ocr_workers >= 1`,
`qwen_cache_mb >= 0`, `kg_batch_size >= 1`, `kg_device in {auto,cpu,cuda}`),
then cap workers to logical CPU count.

- [ ] **Step 4: Run profile tests and the existing argument tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_profile.py tests\test_pipeline_runner.py tests\test_main.py tests\test_slides_pdf_to_txt.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the profile component**

```powershell
git add runtime_profile.py tests/test_runtime_profile.py
git commit -m "feat: add portable runtime profiles"
```

---

### Task 2: Bounded Q5 Prompt Cache

**Files:**
- Modify: `local_qwen.py`
- Modify: `tests/test_local_qwen.py`

**Interfaces:**
- Consumes: `qwen_cache_mb` from the resolved runtime profile.
- Produces: `LocalQwenBackend(..., cache_mb: int = 0)`.
- Produces: `LocalQwenBackend.model_identity -> dict[str, str | int]` containing repository, filename, and byte size.
- Preserves: `complete(system, user, *, max_tokens) -> str`.

- [ ] **Step 1: Write failing prompt-cache tests**

Add tests that install a complete fake `llama_cpp` module:

```python
def test_backend_attaches_bounded_ram_cache(monkeypatch, tmp_path):
    events = []

    class FakeCache:
        def __init__(self, capacity_bytes):
            self.capacity_bytes = capacity_bytes

    class FakeLlama:
        def __init__(self, **kwargs):
            self.cache = None
        def set_cache(self, cache):
            self.cache = cache
            events.append(cache)
        def close(self):
            events.append("closed")

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, LlamaRAMCache=FakeCache),
    )
    model = tmp_path / MODEL_FILENAME
    model.write_bytes(b"gguf")

    backend = LocalQwenBackend(model, cache_mb=256)

    assert backend._llm.cache.capacity_bytes == 256 * 1024 * 1024
    assert backend.model_identity == {
        "repository": MODEL_REPO,
        "filename": MODEL_FILENAME,
        "byte_size": 4,
    }
    backend.close()
    assert events[-2:] == [None, "closed"]
```

Add a second test where `FakeCache.__init__` raises `RuntimeError`; assert a
`RuntimeWarning` is emitted, backend construction succeeds, and completion can
still run. Add a zero-capacity test asserting no cache object is constructed.
Add a fake cache whose `cache_size` remains zero after the first completion;
assert the backend detaches it, emits one warning that the state exceeded the
usable cap, and still returns the assistant content.

- [ ] **Step 2: Run cache tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_qwen.py -q
```

Expected: new tests fail because `cache_mb` and `model_identity` do not exist.

- [ ] **Step 3: Implement optional cache setup and cleanup**

Update construction after `Llama(...)` succeeds:

```python
self._cache = None
if cache_mb < 0:
    raise ValueError("cache_mb must not be negative")
if cache_mb:
    try:
        from llama_cpp import LlamaRAMCache
        self._cache = LlamaRAMCache(cache_mb * 1024 * 1024)
        self._llm.set_cache(self._cache)
    except Exception as exc:
        self._cache = None
        warnings.warn(f"Q5 prompt cache disabled: {exc}", RuntimeWarning, stacklevel=2)
```

Store the resolved `Path(model_path)` and expose repository, exact filename,
and `stat().st_size`. In `close`, call `set_cache(None)` when available, clear
`self._cache`, then close the Llama object. Make cleanup idempotent. Do not
change chat messages, response format, temperature, seed, or output parsing.
After a successful first completion, if a configured cache still reports
`cache_size == 0`, detach it and warn once so later calls do not repeatedly pay
cache bookkeeping that cannot retain a state. Use `getattr` so fake/older cache
implementations without `cache_size` keep working.

- [ ] **Step 4: Run Q5 and flashcard pipeline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_local_qwen.py tests\test_flashcard_pipeline.py tests\test_main.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Q5 cache support**

```powershell
git add local_qwen.py tests/test_local_qwen.py
git commit -m "perf: add bounded Q5 prompt caching"
```

---

### Task 3: Bounded OCR Concurrency

**Files:**
- Modify: `pdf_ingestion.py`
- Modify: `tests/test_pdf_ingestion.py`

**Interfaces:**
- Produces: `_render_page_png(page: object, dpi: int) -> bytes`.
- Produces: `_ocr_png(png: bytes) -> str`.
- Changes: `extract_pdf_pages(..., ocr_workers: int = 1, render: Callable[[object, int], bytes] | None = None, ocr_image: Callable[[bytes], str] | None = None)`.
- Preserves compatibility: existing `ocr(page, dpi)` injection remains supported for current tests/callers and is scheduled through the same bounded pool.

- [ ] **Step 1: Write failing ordering and concurrency tests**

Extend `FakePage` with a `get_pixmap`-compatible injected renderer only in new
tests. Use a locked counter inside the injected OCR callable:

```python
def test_ocr_workers_are_bounded_and_results_keep_page_order(tmp_path):
    source = pdf_file(tmp_path)
    document = FakeDocument(["", "", ""])
    active = 0
    peak = 0
    gate = threading.Barrier(2)
    lock = threading.Lock()

    def ocr(page, dpi):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if page in document.pages[:2]:
            gate.wait(timeout=2)
        result = f"Recovered page {document.pages.index(page) + 1}"
        with lock:
            active -= 1
        return result

    pages = extract_pdf_pages(
        source,
        document_factory=lambda path: document,
        ocr=ocr,
        ocr_workers=2,
    )

    assert peak == 2
    assert [page.text for page in pages] == [
        "Recovered page 1", "Recovered page 2", "Recovered page 3"
    ]
```

Add tests asserting `ocr_workers=0` raises before opening the document, direct
text pages never enter the pool, and an injected OCR failure mentions its
one-based page number while the document is still closed. Add a blocked-worker
test proving page three is not rendered until one of the first two futures has
completed; this locks the in-flight rendered-image bound, not only thread count.

- [ ] **Step 2: Run PDF ingestion tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pdf_ingestion.py -q
```

Expected: new tests fail because `ocr_workers` is unsupported and OCR is
sequential.

- [ ] **Step 3: Split rendering from OCR and add a bounded scheduler**

Keep PyMuPDF calls on the caller thread. Default OCR should render PNG bytes
before submitting work:

```python
def _render_page_png(page: object, dpi: int) -> bytes:
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    return bytes(pixmap.tobytes("png"))


def _ocr_png(png: bytes) -> str:
    image = Image.open(BytesIO(png))
    return str(pytesseract.image_to_string(image))
```

Use `ThreadPoolExecutor(max_workers=ocr_workers)` plus a FIFO deque containing
at most `ocr_workers` futures. Resolve the oldest future before rendering and
submitting another OCR page. Store results by page index and construct the
final tuple in source order. On a future exception, cancel queued futures and
raise `PdfExtractionError(f"local OCR failed on page {number}: {exc}")`.

Keep `ocr(page, dpi)` as an injected compatibility branch. Configure Tesseract
once before default workers start, not inside every worker.

- [ ] **Step 4: Run ingestion and slides tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pdf_ingestion.py tests\test_slides_pdf_to_txt.py -q
```

Expected: all selected tests pass and the concurrency test reports peak two.

- [ ] **Step 5: Commit bounded OCR**

```powershell
git add pdf_ingestion.py tests/test_pdf_ingestion.py
git commit -m "perf: bound concurrent slide OCR"
```

---

### Task 4: Portable Normalization Cache Primitives

**Files:**
- Create: `normalization_cache.py`
- Create: `tests/test_normalization_cache.py`

**Interfaces:**
- Consumes: `ExtractedPage` from `pdf_ingestion.py`.
- Produces: `CACHE_FORMAT_VERSION = 1` and `NORMALIZATION_PROMPT_VERSION = 1`.
- Produces: `cache_directory(output: Path) -> Path`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `ModelIdentity(repository: str, filename: str, byte_size: int | None)`; `None` is transient and cannot produce a reusable slide checkpoint.
- Produces: `ExtractionIdentity(source_sha256: str, min_chars: int, dpi: int)`.
- Produces: `SlideIdentity(source_sha256: str, model: ModelIdentity, n_ctx: int, max_tokens: int, seed: int, temperature: float)`.
- Produces: `slide_cache_key(identity: SlideIdentity, number: int, source_text: str, extraction_method: str) -> str`.
- Produces: `NormalizationCache(root: Path, extraction: ExtractionIdentity, slide: SlideIdentity)` with `load_pages`, `save_pages`, `load_slide`, `save_slide`, and `stats`.

- [ ] **Step 1: Write failing cache primitive tests**

Create literal tests for portability and corruption:

```python
def test_cache_directory_is_relative_to_output_location(tmp_path):
    output = tmp_path / "workspace" / "structured_module.txt"
    assert cache_directory(output) == output.parent / ".structured_module.normalization-cache"


def test_slide_key_does_not_include_absolute_model_directory(tmp_path):
    left = SlideIdentity(
        source_sha256="source",
        model=ModelIdentity(MODEL_REPO, MODEL_FILENAME, 1234),
        n_ctx=8192,
        max_tokens=2048,
        seed=42,
        temperature=0.2,
    )
    right = replace(left, model=ModelIdentity(MODEL_REPO, MODEL_FILENAME, 1234))
    assert slide_cache_key(left, 17, "page text", "ocr") == slide_cache_key(
        right, 17, "page text", "ocr"
    )


def test_corrupt_extraction_cache_is_a_miss(tmp_path):
    cache = make_cache(tmp_path)
    cache.extraction_path.parent.mkdir(parents=True)
    cache.extraction_path.write_text("not json", encoding="utf-8")
    assert cache.load_pages() is None
    assert cache.stats.extraction_hits == 0
```

Also test: matching extraction round trip; changed PDF digest/min-chars/DPI is a
miss; slide round trip; page text/method/number/model filename/model byte
size/n_ctx/max_tokens/seed/temperature change is a miss; a partial temporary
file is ignored; and `OSError` during `os.replace` emits a warning without
leaving a target file.

- [ ] **Step 2: Run cache tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_normalization_cache.py -q
```

Expected: collection fails because `normalization_cache` does not exist.

- [ ] **Step 3: Implement content-addressed cache storage**

Serialize JSON with `sort_keys=True`, UTF-8, and a trailing newline. Use
`NamedTemporaryFile(delete=False, dir=target.parent)` followed by `os.replace`.
Always remove an abandoned temporary file in `finally`.

Name extraction storage `extracted_pages.json`. Name slide files
`slides/{number:06d}-{sha256(canonical_identity_json)}.json`. Store the complete
identity beside each payload and require exact equality on read. Page payloads
must be a non-empty ordered list of dictionaries containing integer `number`,
string `text`, and method `text` or `ocr`; otherwise return a miss. Slide
payloads must be JSON objects; semantic validation remains in
`slide_normalizer._parse_page`.

Implement a frozen `CacheStats(extraction_hits=0, slide_hits=0,
extraction_misses=0, slide_misses=0)` replacement on each read so callers can
report counts without mutating files.

- [ ] **Step 4: Run cache tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_normalization_cache.py -q
```

Expected: all cache tests pass.

- [ ] **Step 5: Commit cache primitives**

```powershell
git add normalization_cache.py tests/test_normalization_cache.py
git commit -m "feat: add portable normalization cache"
```

---

### Task 5: Resumable Extraction and Slide Normalization

**Files:**
- Modify: `slide_normalizer.py`
- Modify: `slides_pdf_to_txt.py`
- Modify: `tests/test_slide_normalizer.py`
- Modify: `tests/test_slides_pdf_to_txt.py`

**Interfaces:**
- Consumes: `NormalizationCache` from Task 4.
- Changes: `normalize_document(..., checkpoint_store: NormalizationCache | None = None)`.
- Changes: `slides_pdf_to_txt.run(args, *, backend=None, metrics_sink: Callable[[CacheStats], None] | None = None) -> Path` resolves output/cache before extraction.
- Produces CLI flags: `--force`, `--profile`, `--ocr-workers`, and `--qwen-cache-mb`.

- [ ] **Step 1: Write a failing checkpoint-validation test**

Add a small in-memory store double whose `load_slide` returns a JSON object.
Assert a valid hit avoids the backend, while a heading-only/invalid hit is
ignored and replaced:

```python
def test_valid_slide_checkpoint_avoids_model_call():
    store = FakeCheckpointStore({1: json.loads(response())})
    backend = FakeBackend([])

    module = normalize_document(
        backend,
        (SimpleNamespace(number=1, text="Page text", method="text"),),
        source_file="module.pdf",
        course_code="CPE0021",
        module_number="1",
        module_title=None,
        checkpoint_store=store,
    )

    assert module.slides[0].title == "Instruction Cycle"
    assert backend.calls == []
```

Add a test where `load_slide` returns `{"title": 7}` followed by one backend
response; assert the backend is called once and `save_slide` receives the new
validated JSON object.

- [ ] **Step 2: Run the slide tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_slide_normalizer.py -q
```

Expected: `checkpoint_store` is an unexpected argument.

- [ ] **Step 3: Add checkpoint hooks around the existing parser**

In `_normalize_page`, ask the store for the page value before calling Q5. Pass
every hit through `_parse_page`; catch only validation/cache read errors and
continue to the model path. After `_extract_json_object` and `_parse_page` both
succeed, call `save_slide` before returning. Do not cache failed attempts. Keep
the existing focused retry feedback and scalar `visual_text` normalization.

- [ ] **Step 4: Write a failing interrupted-run integration test**

In `tests/test_slides_pdf_to_txt.py`, use three extracted pages. The first run's
backend returns valid slide 1 and fails slide 2 for all attempts. The second run
uses the same cache directory and a backend with only valid slide 2 and slide 3
responses. Assert extraction runs once total, slide 1 is not regenerated, and
the second run writes a valid three-slide module.

Also assert changed PDF bytes or `--force` calls extraction and every slide
again. Use a tiny fake GGUF file so the portable model identity is available;
never instantiate llama-cpp.

- [ ] **Step 5: Run the integration test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_slides_pdf_to_txt.py -q
```

Expected: repeated extraction and earlier slide generation prove caches are not
yet wired into `run`.

- [ ] **Step 6: Wire extraction and slide caches into the command**

Compute the output path before extraction. Resolve the profile and explicit
overrides. Unless forced, attempt `cache.load_pages()`; on a miss call
`extract_pdf_pages(..., ocr_workers=resolved.ocr_workers)` and immediately
`save_pages`. Build `SlideIdentity` from the PDF digest, exact model identity,
and normalization settings. Pass the store to `normalize_document`.

After rendering the complete module, call `metrics_sink(cache.stats)` when a
sink was supplied. Cache statistics are observational; failure or absence of a
sink cannot change output or error behavior.

For an injected backend without a readable model identity, allow tests/callers
to supply `args.model_identity`; otherwise disable slide checkpoint reuse with
a `RuntimeWarning` rather than inventing an identity. Production loaders expose
the identity through `LocalQwenBackend.model_identity`.

- [ ] **Step 7: Run normalization integration and structured-output tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_slide_normalizer.py tests\test_slides_pdf_to_txt.py tests\test_structured_module.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit resumable normalization**

```powershell
git add slide_normalizer.py slides_pdf_to_txt.py tests/test_slide_normalizer.py tests/test_slides_pdf_to_txt.py
git commit -m "perf: resume validated slide normalization"
```

---

### Task 6: Propagate Profiles Through CLI, API, and Batch Stages

**Files:**
- Modify: `pipeline.py`
- Modify: `batch_pipeline.py`
- Modify: `main.py`
- Modify: `api_server.py`
- Modify: `text_extractor.py`
- Modify: `tests/test_pipeline_runner.py`
- Modify: `tests/test_batch_pipeline.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_api_server.py`
- Modify: `tests/test_text_extractor_structured.py`

**Interfaces:**
- Consumes: `resolve_runtime_profile`, `RuntimeOverrides`, `detect_hardware`, and `limit_rebel_batch_size`.
- Produces common CLI option: `--profile {auto,low-memory,performance}`.
- Produces Q5 options: `--qwen-cache-mb` and the existing GPU/context options.
- Produces OCR option: `--ocr-workers`.
- Preserves explicit REBEL option: `--kg-batch-size` / `--batch-size`.
- Produces API environment configuration documented in the spec.

- [ ] **Step 1: Write failing CLI propagation tests**

Update parser tests to assert the default profile is `auto` and resource
overrides default to `None` before resolution. Add a command-building test:

```python
def test_profile_and_resource_overrides_reach_stage_commands(tmp_path):
    args = make_args(
        tmp_path,
        "--profile", "low-memory",
        "--ocr-workers", "1",
        "--qwen-cache-mb", "128",
        "--kg-batch-size", "3",
    )
    commands = pipeline.build_stage_commands(args, pipeline.pipeline_paths(args.pdf, args.output_root))
    assert command_value(commands[0].command, "--profile") == "low-memory"
    assert command_value(commands[0].command, "--ocr-workers") == "1"
    assert command_value(commands[0].command, "--qwen-cache-mb") == "128"
    assert command_value(commands[1].command, "--batch-size") == "3"
    assert command_value(commands[2].command, "--qwen-cache-mb") == "128"
```

Add `main.py`, `slides_pdf_to_txt.py`, and `text_extractor.py` help/default
tests for the same profile names.

- [ ] **Step 2: Run parser/command tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline_runner.py tests\test_main.py tests\test_slides_pdf_to_txt.py tests\test_text_extractor_structured.py -q
```

Expected: new profile and cache flags are absent.

- [ ] **Step 3: Add flags and resolve profiles at entry boundaries**

Add `--profile` everywhere a model stage can run. Use `None` parser defaults
for resource fields that profiles own, preserving whether a caller explicitly
set them. Resolve once before constructing a runtime. Pass
`cache_mb=profile.qwen_cache_mb` to each `LocalQwenBackend` construction.

For pipeline subprocesses, propagate the profile name and only emit explicit
override options when the original value was not `None`. For in-process batch
items, store resolved numeric fields plus `kg_batch_size_explicit` on the
namespace.

- [ ] **Step 4: Write failing API environment tests**

Use `monkeypatch.setenv` for `MODULE_FLASHCARDS_PROFILE=low-memory` and each
resource override. Assert `pipeline_args` contains the documented resolved
values. Add invalid integer/profile cases and assert `/process` returns a
batch-level error before `save_upload` or `run_batch` is called. Retain the
existing response keys assertion.

- [ ] **Step 5: Implement API environment resolution**

Parse only these names:

```text
MODULE_FLASHCARDS_PROFILE
MODULE_FLASHCARDS_OCR_WORKERS
MODULE_FLASHCARDS_QWEN_CACHE_MB
MODULE_FLASHCARDS_KG_BATCH_SIZE
MODULE_FLASHCARDS_N_GPU_LAYERS
MODULE_FLASHCARDS_KG_DEVICE
```

Convert integer fields with one shared function that raises
`ValueError("<NAME> must be an integer")`. Resolve before acquiring the request
lock or saving uploads. Do not include profile/timing information in the public
response.

- [ ] **Step 6: Write failing live-memory REBEL tests**

Add a batch adapter test with initial resolved batch four and
`available_memory_bytes=3 * GIB`; assert `text_extractor.run` receives batch one.
Add another with `kg_batch_size_explicit=True`; assert batch four remains. In
`text_extractor` standalone tests, assert auto resolution uses the same limiter
immediately before `load_runtime`.

- [ ] **Step 7: Implement live-memory limiting without changing beams**

Immediately before REBEL loads, call `detect_hardware` and lower automatic
batch sizes through `limit_rebel_batch_size`. Never alter explicit values and
never alter `num_beams`, chunks, overlap, or generation length. Record the
actual numeric value on the run namespace for manifest creation.

- [ ] **Step 8: Replace absolute model-directory manifest identity**

Change `_manifest_settings` to remove `str(Path(args.model_dir).resolve())` and
store this literal structure instead:

```python
"qwen_model": {
    "repository": MODEL_REPO,
    "filename": MODEL_FILENAME,
    "byte_size": model_path.stat().st_size if model_path.is_file() else None,
}
```

Write a regression that creates equal-size fake models under two different
absolute directories and asserts `_manifest_contents` is equal. Change the
filename or byte size and assert inequality. Missing model files make the
manifest non-reusable rather than raising out of the API result path.

- [ ] **Step 9: Run all profile integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_profile.py tests\test_pipeline_runner.py tests\test_batch_pipeline.py tests\test_main.py tests\test_api_server.py tests\test_text_extractor_structured.py -q
```

Expected: all selected tests pass, API response keys remain unchanged, and
loader-order assertions still show Q5 → REBEL → Q5.

- [ ] **Step 10: Commit profile propagation**

```powershell
git add pipeline.py batch_pipeline.py main.py api_server.py text_extractor.py tests/test_pipeline_runner.py tests/test_batch_pipeline.py tests/test_main.py tests/test_api_server.py tests/test_text_extractor_structured.py
git commit -m "perf: adapt pipeline resources to local hardware"
```

---

### Task 7: Stage Metrics and User Documentation

**Files:**
- Modify: `batch_pipeline.py`
- Modify: `pipeline.py`
- Modify: `api_server.py`
- Modify: `slides_pdf_to_txt.py`
- Modify: `README.md`
- Modify: `tests/test_batch_pipeline.py`
- Modify: `tests/test_api_server.py`
- Modify: `tests/test_batch_benchmark.py`

**Interfaces:**
- Produces: `StageMetric(name: str, elapsed_seconds: float)`.
- Produces: `NormalizationStageOutcome(path: Path, cache_stats: CacheStats)` from the in-process normalization adapter.
- Changes: `BatchResult(..., metrics: tuple[StageMetric, ...] = (), extraction_cache_hits: int = 0, slide_cache_hits: int = 0)`.
- Preserves: API JSON keys `outputs`, `errors`, `courseCode`, and `processUrl`.

- [ ] **Step 1: Write failing deterministic metrics tests**

Inject a `perf_counter` iterator into `BatchDependencies` and assert literal
stage totals without sleeping:

```python
def test_batch_result_reports_stage_metrics_without_changing_outputs(tmp_path):
    clock = iter((0.0, 2.0, 3.0, 8.0, 10.0, 14.0))
    dependencies = replace(
        fake_dependencies(events=[]),
        perf_counter=lambda: next(clock),
    )
    result = run_batch((make_item(tmp_path, "one.pdf"),), dependencies=dependencies)
    assert [(m.name, m.elapsed_seconds) for m in result.metrics] == [
        ("normalize", 2.0), ("graph", 5.0), ("flashcards", 4.0)
    ]
```

Add an API test asserting `set(response) == {"outputs", "errors",
"courseCode", "processUrl"}` even when the internal result carries metrics.

- [ ] **Step 2: Run metrics tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_batch_pipeline.py tests\test_api_server.py -q
```

Expected: `BatchDependencies` lacks `perf_counter` and `BatchResult` lacks
metrics.

- [ ] **Step 3: Add observational metrics**

Wrap each whole `_run_stage` call with `dependencies.perf_counter`; append a
metric even when no item needs that stage, using `0.0`. Do not use the metrics
clock for timeout accounting. `_normalize_stage` passes a `metrics_sink` to
`slides_pdf_to_txt.run`, then returns `NormalizationStageOutcome`; `_run_stage`
sums those returned hit counts without treating metrics as validation output.
Print one concise CLI line and one server log line:

```text
Performance: normalize=12.34s graph=8.20s flashcards=44.10s extraction_hits=1 slide_hits=37
```

Use the standard `logging` module in the API. Do not add public response fields.

- [ ] **Step 4: Replace benchmark assertions with work-count evidence**

Extend `tests/test_batch_benchmark.py` with a cold run that records extraction,
normalization, graph, and flashcard call counts, followed by an interrupted and
resumed normalization run. Assert completed slide checkpoints are not called
again. Do not assert elapsed time or speedup ratios.

- [ ] **Step 5: Document profiles, portability, and honest benchmarking**

Update `README.md` with:

```text
--profile auto          Detect resources on each PC (default)
--profile low-memory    Ryzen 5 / 12 GB-safe limits
--profile performance   24 GB+ or supported-GPU limits
```

Document API environment equivalents, private cache directory names, `--force`
behavior, copying exact GGUF/generated outputs to another PC, and the need to
install machine-specific GPU builds/drivers. State that warm/resumed runs gain
the most and that first-run time still depends on slide/OCR complexity.

- [ ] **Step 6: Run metrics, benchmark, API, and help tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_batch_pipeline.py tests\test_batch_benchmark.py tests\test_api_server.py tests\test_pipeline_runner.py -q
.\.venv\Scripts\python.exe pipeline.py --help
.\.venv\Scripts\python.exe slides_pdf_to_txt.py --help
.\.venv\Scripts\python.exe text-extractor.py --help
.\.venv\Scripts\python.exe main.py --help
```

Expected: all tests and all four help commands exit zero; help lists the three
profiles where applicable.

- [ ] **Step 7: Commit metrics and documentation**

```powershell
git add batch_pipeline.py pipeline.py api_server.py slides_pdf_to_txt.py README.md tests/test_batch_pipeline.py tests/test_api_server.py tests/test_batch_benchmark.py tests/test_pipeline_runner.py
git commit -m "docs: expose portable performance profiles"
```

---

### Task 8: Full Regression and Release Readiness

**Files:**
- Modify only if verification finds a concrete regression; add its failing test before any production correction.

**Interfaces:**
- Verifies every interface and invariant produced by Tasks 1-7.

- [ ] **Step 1: Run focused quality invariants**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_slide_normalizer.py tests\test_structured_module.py tests\test_graph_input.py tests\test_flashcard_pipeline.py tests\test_flashcard_validator.py tests\test_flashcard_csv.py -q
```

Expected: every quality/structure test passes without changed fixtures that
weaken validation.

- [ ] **Step 2: Run the complete model-free suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: zero failures. Record the exact pass count and duration.

- [ ] **Step 3: Compile every changed Python module**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile runtime_profile.py normalization_cache.py local_qwen.py pdf_ingestion.py slide_normalizer.py slides_pdf_to_txt.py pipeline.py batch_pipeline.py main.py api_server.py text_extractor.py
```

Expected: exit zero with no output.

- [ ] **Step 4: Check repository integrity**

Run:

```powershell
git diff --check
git status --short
git log --oneline --decorate -10
```

Expected: no whitespace errors and no uncommitted production/test changes.

- [ ] **Step 5: Perform an opt-in real-model benchmark only when weights already exist**

If the exact Q5 file and REBEL cache already exist locally, run one representative
20-50-slide PDF twice under `--profile low-memory`, recording the emitted cold
and warm stage metrics and validating the final 100-card output. Do not download
multi-gigabyte weights or claim a 600-slide duration without explicit user
approval and a real 600-slide fixture. If weights are absent, record this step
as skipped and rely on deterministic work-count evidence.

- [ ] **Step 6: Commit only concrete verification fixes**

If Step 1-5 exposed a regression, first add a focused failing test, implement
the minimal fix, rerun the affected command and the full suite, then commit:

```powershell
git status --short
git add -A
git commit -m "fix: preserve quality under performance profiles"
```

Execution occurs in a clean isolated worktree, so review the status output before
`git add -A`. If verification is clean, create no empty commit.
