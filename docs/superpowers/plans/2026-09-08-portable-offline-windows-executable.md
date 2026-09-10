# Portable Offline Windows Executable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and document a console-based, one-directory Windows distribution that runs the existing local API with bundled Qwen, REBEL, Tesseract, and CUDA dependencies while keeping generated data beside the executable.

**Architecture:** A small path/configuration layer distinguishes source and frozen execution. The launcher establishes offline-only environment variables, validates the adjacent bundle, reports GPU capability, and starts the existing FastAPI app. Model loaders receive explicit local paths in portable mode, while source mode retains its current developer download behavior. PyInstaller freezes Python code into a one-directory runtime; a tested assembly utility copies large external assets and writes their integrity manifest after freezing.

**Tech Stack:** Python 3.11+, FastAPI/Uvicorn, PyInstaller, llama-cpp-python, PyTorch, Transformers, Hugging Face Hub, Tesseract, PowerShell, pytest

**Spec:** `docs/superpowers/specs/2026-09-08-portable-offline-windows-executable-design.md`

## Global Constraints

- Target Windows 11 x64, AMD Ryzen 7 5800X, 16 GB RAM, NVIDIA GeForce RTX 5070 12 GB.
- Produce a console-enabled PyInstaller one-directory distribution named `ModuleToFlashcards`.
- Keep `models`, `tesseract`, `licenses`, and `data` beside `ModuleToFlashcards.exe`; place frozen Python dependencies under `runtime`.
- Frozen execution must not contact Hugging Face or depend on a user model cache.
- Qwen and REBEL remain stage-sequential and are never held in GPU memory together.
- CUDA failures warn and select CPU; missing/corrupt bundled assets fail without downloading.
- Source-mode Python commands and current test injection seams remain supported.
- Large models, Tesseract binaries, CUDA binaries, generated bundle contents, uploads, outputs, temporary data, and logs remain outside Git.
- Final CUDA build and GPU acceptance happen on the RTX 5070 target; this CPU-only worktree may verify source behavior and packaging configuration only.

---

### Task 1: Repair the inherited flashcard-generation regression

**Files:**
- Modify: `flashcard_prompt.py`
- Modify: `flashcard_pipeline.py`
- Modify: `flashcard_validator.py`
- Test: `tests/test_flashcard_prompt.py`
- Test: `tests/test_flashcard_validator.py`
- Test: `tests/test_flashcard_pipeline.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `build_concept_plan_prompt(identity, facts, prior_concept_names=())`, `parse_concept_plan(raw, known_facts)`, and `FlashcardPipeline.run(identity, facts, *, prior_concept_names=(), prior_questions=())`.
- Produces: a prompt that serializes previous concepts, a parser that rejects fewer than 20 concepts and deterministically keeps the first 20 from an overshoot, and a pipeline that forwards previous concepts.

- [ ] **Step 1: Reproduce the merge regression with the existing tests**

Run:

```powershell
python -m pytest tests/test_flashcard_prompt.py tests/test_flashcard_validator.py tests/test_flashcard_pipeline.py tests/test_main.py -q
```

Expected: 15 failures, led by `NameError: overlap_guidance is not defined` and concept-count expectation mismatches.

- [ ] **Step 2: Restore the cross-module prompt guidance and count contract**

In `build_concept_plan_prompt`, define the conditional payload and guidance before constructing the prompt:

```python
overlap_guidance = ""
if prior_concept_names:
    payload["previously_covered_concepts"] = list(prior_concept_names)
    overlap_guidance = (
        "\nAvoid selecting a concept that assesses the same underlying learning "
        "point as any entry in previously_covered_concepts, even if phrased "
        "differently. Prefer concepts distinctive to this module's own facts.\n"
    )
```

Retain the twenty-position checklist and literal `fact_ids` instruction already asserted by `tests/test_flashcard_prompt.py`. Remove the duplicated insufficient-content paragraph introduced by the merge.

Pass `prior_concept_names` from `FlashcardPipeline.run`:

```python
plan_prompt = build_concept_plan_prompt(identity, facts, prior_concept_names)
```

In `parse_concept_plan`, reject only underfilled arrays and validate exactly the selected first twenty:

```python
if len(concepts_value) < CONCEPTS_PER_MODULE:
    raise ValidationError(
        f"expected at least {CONCEPTS_PER_MODULE} concepts, "
        f"received {len(concepts_value)}"
    )
concepts_value = concepts_value[:CONCEPTS_PER_MODULE]
```

- [ ] **Step 3: Verify the focused regression suite passes**

Run the Step 1 command again.

Expected: all selected tests pass.

- [ ] **Step 4: Verify the complete pre-packaging suite passes**

Run:

```powershell
python -m pytest -q
```

Expected: 200 tests pass.

- [ ] **Step 5: Commit the isolated regression repair**

```powershell
git add flashcard_prompt.py flashcard_pipeline.py flashcard_validator.py
git commit -m "fix: repair merged flashcard planning contract"
```

---

### Task 2: Add source/frozen runtime paths and adjacent data storage

**Files:**
- Create: `portable_paths.py`
- Create: `tests/test_portable_paths.py`
- Modify: `api_server.py`
- Modify: `tests/test_api_server.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `PortablePaths`, `application_root()`, `build_paths()`, `prepare_data_directories()`, and `configure_api_storage()`.
- Consumes later: `portable_launcher.py`, model loaders, Tesseract configuration, and bundle verification.

- [ ] **Step 1: Write failing tests for source roots, frozen roots, and data layout**

Create tests that express this API:

```python
def test_build_paths_uses_adjacent_data_in_portable_mode(tmp_path):
    paths = build_paths(tmp_path, portable=True)
    assert paths.qwen_model == tmp_path / "models" / MODEL_FILENAME
    assert paths.rebel_model == tmp_path / "models" / "rebel-large"
    assert paths.uploads == tmp_path / "data" / "uploads"
    assert paths.outputs == tmp_path / "data" / "pipeline_output"
    assert paths.temporary == tmp_path / "data" / "temporary"
    assert paths.logs == tmp_path / "data" / "logs"


def test_application_root_uses_executable_parent_when_frozen(tmp_path, monkeypatch):
    executable = tmp_path / "ModuleToFlashcards.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    assert application_root() == tmp_path.resolve()
```

Add an API test that calls `configure_api_storage(paths)` and proves upload/output helpers use the injected adjacent data directories.

- [ ] **Step 2: Run the tests and verify they fail because the API is absent**

```powershell
python -m pytest tests/test_portable_paths.py tests/test_api_server.py -q
```

Expected: collection/import failure for `portable_paths` or missing functions.

- [ ] **Step 3: Implement the focused path abstraction**

Create:

```python
@dataclass(frozen=True)
class PortablePaths:
    root: Path
    models: Path
    qwen_model: Path
    rebel_model: Path
    tesseract_exe: Path
    tessdata: Path
    data: Path
    uploads: Path
    outputs: Path
    temporary: Path
    logs: Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def application_root() -> Path:
    base = Path(sys.executable) if is_frozen() else Path(__file__)
    return base.resolve().parent


def build_paths(root: Path | None = None, *, portable: bool | None = None) -> PortablePaths:
    root = (root or application_root()).resolve()
    portable = is_frozen() if portable is None else portable
    data = root / "data" if portable else root
    return PortablePaths(
        root=root,
        models=root / "models",
        qwen_model=root / "models" / MODEL_FILENAME,
        rebel_model=root / "models" / "rebel-large",
        tesseract_exe=root / "tesseract" / "tesseract.exe",
        tessdata=root / "tesseract" / "tessdata",
        data=data,
        uploads=data / "uploads" if portable else root / "pipeline_uploads",
        outputs=data / "pipeline_output" if portable else root / "pipeline_output",
        temporary=data / "temporary" if portable else root / "pipeline_temporary",
        logs=data / "logs" if portable else root / "pipeline_logs",
    )


def prepare_data_directories(paths: PortablePaths) -> None:
    for directory in (
        paths.data, paths.uploads, paths.outputs, paths.temporary, paths.logs
    ):
        directory.mkdir(parents=True, exist_ok=True)
    marker: Path | None = None
    try:
        with NamedTemporaryFile(dir=paths.data, prefix=".write-test-", delete=False) as handle:
            marker = Path(handle.name)
    except OSError as exc:
        raise RuntimeError(f"portable data directory is not writable: {paths.data}") from exc
    finally:
        if marker is not None:
            marker.unlink(missing_ok=True)
```

`prepare_data_directories` creates only `data`, `uploads`, `outputs`, `temporary`, and `logs`, then performs a create/delete probe inside `data` and raises a path-specific `RuntimeError` on failure.

Add `configure_api_storage(paths)` to update the existing `UPLOAD_DIR` and `OUTPUT_ROOT` injection-friendly globals. Source mode retains `pipeline_uploads/` and `pipeline_output/`; portable mode uses the adjacent `data` tree.

Ignore `/data/`, `/dist/`, `/build/`, and `/packaging/assets/` in `.gitignore`.

- [ ] **Step 4: Run focused and full tests**

Run the Step 2 command, then `python -m pytest -q`.

Expected: all tests pass.

- [ ] **Step 5: Commit runtime path support**

```powershell
git add portable_paths.py tests/test_portable_paths.py api_server.py tests/test_api_server.py .gitignore
git commit -m "feat: add portable runtime paths"
```

---

### Task 3: Make Qwen, REBEL, and Tesseract strictly local in portable mode

**Files:**
- Create: `portable_runtime.py`
- Create: `tests/test_portable_runtime.py`
- Modify: `local_qwen.py`
- Modify: `text_extractor.py`
- Modify: `pdf_ingestion.py`
- Modify: `batch_pipeline.py`
- Modify: `api_server.py`
- Modify: `tests/test_local_qwen.py`
- Modify: `tests/test_text_extractor_structured.py`
- Modify: `tests/test_pdf_ingestion.py`
- Modify: `tests/test_batch_pipeline.py`

**Interfaces:**
- Produces: `configure_offline_environment()`, `AccelerationStatus`, `detect_acceleration()`, `ensure_model(model_dir, *, allow_download=True)`, `load_runtime(model_name, device, *, local_files_only=False)`, and bundled-Tesseract preference.
- Consumes: `PortablePaths` from Task 2.
- Used later by: the launcher and startup report.

- [ ] **Step 1: Write failing local-only behavior tests**

Cover these public behaviors:

```python
def test_missing_qwen_model_never_downloads_when_download_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(local_qwen, "hf_hub_download", fail_if_called)
    with pytest.raises(FileNotFoundError, match="bundled Qwen model"):
        ensure_model(tmp_path, allow_download=False)


def test_rebel_local_mode_forwards_local_files_only(tmp_path, fake_transformers):
    load_runtime(tmp_path / "rebel-large", "cpu", local_files_only=True)
    assert fake_transformers.tokenizer_kwargs == {"local_files_only": True}
    assert fake_transformers.model_kwargs == {"local_files_only": True}


def test_offline_environment_is_configured_before_model_imports(monkeypatch):
    configure_offline_environment()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
```

Add an OCR test proving an explicit bundled executable and `tessdata` directory override PATH and standard Windows locations.

- [ ] **Step 2: Run the focused tests and verify the expected signature failures**

```powershell
python -m pytest tests/test_portable_runtime.py tests/test_local_qwen.py tests/test_text_extractor_structured.py tests/test_pdf_ingestion.py tests/test_batch_pipeline.py -q
```

Expected: failures for missing portable runtime APIs and unsupported keyword arguments.

- [ ] **Step 3: Implement offline and device configuration**

Use:

```python
@dataclass(frozen=True)
class AccelerationStatus:
    torch_cuda: bool
    llama_gpu_offload: bool
    torch_device_name: str | None


def configure_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def detect_acceleration() -> AccelerationStatus:
    import llama_cpp
    import torch
    available = bool(torch.cuda.is_available())
    return AccelerationStatus(
        torch_cuda=available,
        llama_gpu_offload=bool(llama_cpp.llama_supports_gpu_offload()),
        torch_device_name=torch.cuda.get_device_name(0) if available else None,
    )
```

`ensure_model(model_dir, allow_download=True)` raises `FileNotFoundError` rather than calling the hub when downloads are disabled. `load_runtime` accepts `str | Path` and passes `local_files_only=True` to both Transformers loaders in portable mode.

Add optional `tesseract_executable` and `tessdata_dir` inputs at the OCR boundary. In portable mode, `batch_pipeline` uses Task 2's exact Qwen and REBEL paths, disables Qwen download, enables Transformers local-only loading, and supplies bundled OCR paths. In source mode, all existing defaults remain unchanged.

- [ ] **Step 4: Verify focused and full tests**

Run the Step 2 command, then `python -m pytest -q`.

Expected: all tests pass.

- [ ] **Step 5: Commit local-only model loading**

```powershell
git add portable_runtime.py tests/test_portable_runtime.py local_qwen.py text_extractor.py pdf_ingestion.py batch_pipeline.py api_server.py tests
git commit -m "feat: load bundled models offline"
```

---

### Task 4: Add a cached integrity manifest for the portable bundle

**Files:**
- Create: `portable_manifest.py`
- Create: `tests/test_portable_manifest.py`
- Create: `packaging/model-lock.json`

**Interfaces:**
- Produces: `AssetRecord`, `BundleVerificationError`, `write_manifest()`, and `verify_manifest()`.
- Consumes: the portable application root and `data` path from Task 2.
- Used later by: launcher startup and package assembly.

- [ ] **Step 1: Write failing manifest tests**

Express the API with real temporary files:

```python
def test_verify_manifest_hashes_once_then_uses_matching_cache(tmp_path, monkeypatch):
    asset = write_asset(tmp_path, "models/qwen.gguf", b"model")
    manifest = write_test_manifest(tmp_path, asset)
    cache = tmp_path / "data" / ".verification-cache.json"
    verify_manifest(tmp_path, manifest, cache_path=cache)
    monkeypatch.setattr(portable_manifest, "sha256_file", fail_if_called)
    verify_manifest(tmp_path, manifest, cache_path=cache)


def test_verify_manifest_detects_changed_asset(tmp_path):
    asset = tmp_path / "models" / "qwen.gguf"
    asset.parent.mkdir()
    asset.write_bytes(b"model")
    manifest = tmp_path / "models" / "manifest.json"
    cache = tmp_path / "data" / ".verification-cache.json"
    write_manifest(tmp_path, (asset,), manifest)
    verify_manifest(tmp_path, manifest, cache_path=cache)
    asset.write_bytes(b"damaged")
    with pytest.raises(BundleVerificationError, match="models/qwen.gguf"):
        verify_manifest(tmp_path, manifest, cache_path=cache)


def test_force_hash_ignores_cached_verification(tmp_path, monkeypatch):
    asset = tmp_path / "models" / "qwen.gguf"
    asset.parent.mkdir()
    asset.write_bytes(b"model")
    manifest = tmp_path / "models" / "manifest.json"
    cache = tmp_path / "data" / ".verification-cache.json"
    write_manifest(tmp_path, (asset,), manifest)
    verify_manifest(tmp_path, manifest, cache_path=cache)
    original = portable_manifest.sha256_file
    hash_calls = []
    monkeypatch.setattr(
        portable_manifest,
        "sha256_file",
        lambda path: hash_calls.append(path) or original(path),
    )
    verify_manifest(tmp_path, manifest, cache_path=cache, force_hash=True)
    assert hash_calls == [asset]
```

- [ ] **Step 2: Run tests and verify the module is missing**

```powershell
python -m pytest tests/test_portable_manifest.py -q
```

Expected: import failure for `portable_manifest`.

- [ ] **Step 3: Implement validated JSON manifests and cache records**

Use immutable records:

```python
@dataclass(frozen=True)
class AssetRecord:
    path: str
    size: int
    sha256: str


class BundleVerificationError(RuntimeError):
    pass


def write_manifest(root: Path, assets: Sequence[Path], destination: Path) -> None:
    records = [
        AssetRecord(
            path=asset.resolve().relative_to(root.resolve()).as_posix(),
            size=asset.stat().st_size,
            sha256=sha256_file(asset),
        )
        for asset in assets
    ]
    payload = {"schema_version": 1, "files": [asdict(item) for item in records]}
    _atomic_json_write(destination, payload)
```

Implement `verify_manifest(root, manifest_path, *, cache_path=None, force_hash=False) -> tuple[AssetRecord, ...]`. It parses schema version 1 and rejects absolute paths, `..` traversal, duplicate normalized paths, negative sizes, malformed SHA-256 values, symlinks that resolve outside the bundle, missing files, size mismatches, and digest mismatches. Cache keys include manifest digest, relative path, expected digest, file size, and nanosecond modification time. It hashes when `force_hash` is true or a cache key is absent, writes cache JSON atomically, and returns the verified immutable records.

Create `packaging/model-lock.json` with these release inputs:

```json
{
  "schema_version": 1,
  "qwen": {
    "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
    "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
    "filename": "qwen2.5-3b-instruct-q5_k_m.gguf",
    "size": 2438740384,
    "sha256": "2c63dde5f2c9ab1fd64d47dee2d34dade6ba9ff62442d1d20b5342310c982081"
  },
  "rebel": {
    "repo_id": "Babelscape/rebel-large",
    "revision": "44eb6cb4585df284ce6c4d6a7013f83fe473c052",
    "allow_patterns": [
      "config.json",
      "model.safetensors",
      "added_tokens.json",
      "merges.txt",
      "special_tokens_map.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "vocab.json"
    ]
  },
  "tesseract": {
    "version": "5.4.0.20240606",
    "required_files": {
      "tesseract.exe": "babb405f4366b480d02cd8ff2bac8d497170f6c1711ce6f3d5d8bf0fb7fa6ed9",
      "tessdata/eng.traineddata": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
    }
  }
}
```

Asset preparation rejects `main`, any non-40-character lowercase hexadecimal model revision, wrong Qwen size/digest, a missing REBEL allow-pattern result, or mismatched Tesseract required-file digests.

- [ ] **Step 4: Run manifest and full tests**

Run the Step 2 command, then `python -m pytest -q`.

Expected: all tests pass.

- [ ] **Step 5: Commit bundle verification**

```powershell
git add portable_manifest.py tests/test_portable_manifest.py packaging/model-lock.json
git commit -m "feat: verify portable bundle integrity"
```

---

### Task 5: Add the console launcher and server startup checks

**Files:**
- Create: `portable_launcher.py`
- Create: `tests/test_portable_launcher.py`
- Modify: `api_server.py`
- Modify: `tests/test_api_server.py`

**Interfaces:**
- Produces: `parse_args()`, `probe_port()`, `run_launcher()`, and `main()`.
- Consumes: Task 2 paths, Task 3 offline/device configuration, Task 4 verification, and the existing FastAPI `app`.

- [ ] **Step 1: Write failing launcher tests**

Cover argument parsing, startup ordering, verification-only mode, healthy existing instance, foreign port owner, CUDA reporting, CPU fallback environment settings, and server invocation. Use dependency injection rather than starting a real server:

```python
def test_launcher_configures_offline_mode_before_loading_server(tmp_path):
    events = []
    result = run_launcher(
        parse_args([]),
        root=portable_fixture(tmp_path),
        configure_offline=lambda: events.append("offline"),
        load_server=lambda: events.append("server") or fake_app,
        run_server=lambda *_args, **_kwargs: events.append("run"),
    )
    assert result == 0
    assert events == ["offline", "server", "run"]


def test_verify_mode_hashes_assets_and_does_not_start_server(tmp_path):
    calls = []
    result = run_launcher(
        parse_args(["--verify"]),
        root=portable_fixture(tmp_path),
        verify_bundle=lambda force: calls.append(("verify", force)),
        detect_devices=lambda: gpu_status(),
        run_server=lambda *_args, **_kwargs: calls.append(("server", False)),
    )
    assert result == 0
    assert calls == [("verify", True)]
```

- [ ] **Step 2: Run tests and verify launcher APIs are absent**

```powershell
python -m pytest tests/test_portable_launcher.py tests/test_api_server.py -q
```

Expected: import or attribute failures for the launcher.

- [ ] **Step 3: Implement the minimal console launcher**

Arguments:

```python
parser.add_argument("--port", type=valid_port, default=8000)
parser.add_argument("--verify", action="store_true")
parser.add_argument("--version", action="store_true")
```

Startup order is fixed: resolve paths, create/write-test data, configure offline environment, verify bundle, detect acceleration, configure API storage/device settings, probe the port, import the API app, then call `uvicorn.run(app, host="127.0.0.1", port=args.port, reload=False)`.

When both GPU checks are true, configure Qwen `n_gpu_layers=-1` and REBEL `cuda`. Otherwise print a warning and configure `n_gpu_layers=0` and REBEL `cpu`. `--verify` forces all hashes and reports components/device state without starting Uvicorn. Fatal startup exceptions are written to `data/logs/startup-YYYYMMDD-HHMMSS.log` and produce exit code 1.

Probe `GET /health` when the port is occupied. Matching application/version returns an already-running message and exit code 0; any other response or listener produces a foreign-owner error.

- [ ] **Step 4: Run launcher, API, and full tests**

Run the Step 2 command, then `python -m pytest -q`.

Expected: all tests pass.

- [ ] **Step 5: Commit the executable entry point**

```powershell
git add portable_launcher.py tests/test_portable_launcher.py api_server.py tests/test_api_server.py
git commit -m "feat: add portable console launcher"
```

---

### Task 6: Add reproducible asset preparation and PyInstaller assembly

**Files:**
- Create: `packaging/prepare_assets.py`
- Create: `packaging/assemble_portable.py`
- Create: `packaging/ModuleToFlashcards.spec`
- Create: `packaging/build_portable.ps1`
- Create: `requirements-build.txt`
- Create: `tests/test_portable_packaging.py`

**Interfaces:**
- Consumes: `packaging/model-lock.json`, `portable_launcher.py`, `write_manifest()`, locally staged models, portable Tesseract, and a CUDA-enabled Python environment.
- Produces: `dist/ModuleToFlashcards/ModuleToFlashcards.exe`, adjacent assets, integrity manifest, license directory, and a versioned ZIP archive.

- [ ] **Step 1: Write failing tests for locked preparation and safe assembly**

Test the Python helpers without downloading or copying real models:

```python
def test_prepare_rejects_moving_rebel_revision(tmp_path):
    lock = valid_lock(rebel_revision="main")
    with pytest.raises(ValueError, match="immutable REBEL revision"):
        validate_model_lock(lock)


def test_assemble_copies_assets_and_writes_verifiable_manifest(tmp_path):
    result = assemble_portable(fake_frozen_app(tmp_path), fake_assets(tmp_path))
    verify_manifest(result, result / "models" / "manifest.json", force_hash=True)
    assert (result / "ModuleToFlashcards.exe").is_file()
    assert (result / "tesseract" / "tesseract.exe").is_file()


def test_assemble_rejects_output_outside_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="distribution directory"):
        assemble_portable(
            repo_root=repo,
            frozen_app=fake_frozen_app(repo),
            assets=fake_assets(repo),
            output_dir=tmp_path / "outside",
        )
```

- [ ] **Step 2: Run tests and verify preparation modules are absent**

```powershell
python -m pytest tests/test_portable_packaging.py -q
```

Expected: import failure for packaging helpers.

- [ ] **Step 3: Implement testable preparation and assembly**

`prepare_assets.py` validates the lock, downloads the exact Qwen filename and immutable REBEL snapshot only into `packaging/assets`, selects safetensors instead of the pickle weight duplicate, verifies required tokenizer/configuration files, and copies a caller-supplied Tesseract 5.4.0.20240606 directory after confirming the locked digests for `tesseract.exe` and `tessdata/eng.traineddata`.

`assemble_portable.py` accepts resolved source/frozen/asset/output paths, validates every target is inside the repository's named `dist` directory before replacement, copies only declared assets, creates empty `data` subdirectories, writes the runtime manifest, copies license notices, and verifies the result.

- [ ] **Step 4: Add the PyInstaller specification and PowerShell orchestrator**

The spec uses:

```python
a = Analysis(
    [str(PROJECT_ROOT / "portable_launcher.py")],
    pathex=[str(PROJECT_ROOT)],
    hiddenimports=collect_submodules("uvicorn")
        + collect_submodules("transformers")
        + collect_submodules("sentencepiece"),
    binaries=collect_dynamic_libs("torch") + collect_dynamic_libs("llama_cpp"),
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ModuleToFlashcards",
    console=True,
    contents_directory="runtime",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="ModuleToFlashcards",
)
```

`build_portable.ps1` validates that its build/dist/assets paths resolve under the repository, verifies CUDA capability unless `-SkipGpuPreflight` is supplied for CPU-hosted configuration checks, runs PyInstaller, runs assembly and `--verify`, then creates a versioned ZIP. It never removes paths selected by wildcard or unresolved environment variables.

Set `requirements-build.txt` to `PyInstaller==6.22.2`; release dependency versions are recorded in the assembled `licenses/build-versions.txt` from the active environment.

- [ ] **Step 5: Run packaging unit tests and syntax checks**

```powershell
python -m pytest tests/test_portable_packaging.py -q
python -m py_compile packaging/prepare_assets.py packaging/assemble_portable.py packaging/ModuleToFlashcards.spec portable_launcher.py
```

Expected: tests and compilation pass.

- [ ] **Step 6: Run a CPU-hosted PyInstaller smoke build when PyInstaller is available**

```powershell
powershell -ExecutionPolicy Bypass -File packaging/build_portable.ps1 -SkipGpuPreflight -SkipAssetPreparation
```

Expected on this workspace: the freeze/configuration phase succeeds; final offline asset verification may report the explicitly missing Q5/REBEL staging assets. Do not label that partial artifact GPU-ready.

- [ ] **Step 7: Commit packaging infrastructure**

```powershell
git add packaging requirements-build.txt tests/test_portable_packaging.py
git commit -m "build: add portable Windows distribution"
```

---

### Task 7: Document portable installation, layout, operation, and target verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: actual launcher flags, paths, build script parameters, and acceptance commands implemented in Tasks 2-6.
- Produces: user and maintainer instructions consistent with the executable.

- [ ] **Step 1: Add a portable Windows quick-start section**

Document these exact behaviors:

```text
1. Extract the complete ModuleToFlashcards directory to a writable location.
2. Keep ModuleToFlashcards.exe, runtime, models, tesseract, and licenses together.
3. Double-click ModuleToFlashcards.exe and leave its console open.
4. Wait for the http://127.0.0.1:8000 readiness message.
5. Use the existing browser extension; find results under data/pipeline_output.
6. Press Ctrl+C to stop the server.
```

State that copying only the EXE is unsupported, the release is approximately 8-14 GB, `.pptx` must be exported to PDF, and the final release runs offline.

- [ ] **Step 2: Add troubleshooting and verification commands**

Include:

```powershell
.\ModuleToFlashcards.exe --verify
.\ModuleToFlashcards.exe --port 8010
nvidia-smi
```

Explain GPU/CPU messages, writable-folder errors, port conflicts, model-integrity errors, log paths, and the requirement that target validation show CUDA for both Qwen and REBEL.

- [ ] **Step 3: Add maintainer build instructions**

Document asset preparation, Tesseract input, CUDA build prerequisites, the build command, output location, and the prohibition on committing `packaging/assets`, `build`, `dist`, or model/runtime binaries.

- [ ] **Step 4: Verify documentation matches implemented names**

Run:

```powershell
rg -n "ModuleToFlashcards.exe|--verify|--port|data/pipeline_output|build_portable" README.md portable_launcher.py packaging
git diff --check
```

Expected: documented commands and paths exactly match implementation; no whitespace errors.

- [ ] **Step 5: Commit README updates**

```powershell
git add README.md
git commit -m "docs: add portable executable guide"
```

---

### Task 8: Complete source verification and record target-machine acceptance steps

**Files:**
- Modify if needed: `README.md`
- Modify if needed: packaging/runtime files only when verification exposes a covered defect

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a source-verified feature branch and an explicit list of the remaining RTX 5070 acceptance commands.

- [ ] **Step 1: Run the complete automated test suite freshly**

```powershell
python -m pytest -q
```

Expected: all tests pass with no failures.

- [ ] **Step 2: Compile every Python entry point and packaging script**

```powershell
python -m py_compile api_server.py batch_pipeline.py local_qwen.py pdf_ingestion.py portable_launcher.py portable_manifest.py portable_paths.py portable_runtime.py text_extractor.py packaging/prepare_assets.py packaging/assemble_portable.py packaging/ModuleToFlashcards.spec
```

Expected: exit code 0.

- [ ] **Step 3: Run source-mode CLI smoke checks**

```powershell
python portable_launcher.py --version
python portable_launcher.py --help
python pipeline.py --help
python main.py --help
```

Expected: every command exits 0 without downloading or loading a model.

- [ ] **Step 4: Review the branch diff against the specification**

Check each completion criterion in the spec. Confirm no model, generated data, build output, environment directory, or user content is tracked.

- [ ] **Step 5: Run target RTX 5070 acceptance after transferring the branch**

On the target build machine:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch, llama_cpp; print(torch.cuda.is_available()); print(llama_cpp.llama_supports_gpu_offload())"
powershell -ExecutionPolicy Bypass -File packaging/build_portable.ps1 -TesseractDir "C:\Program Files\Tesseract-OCR"
.\dist\ModuleToFlashcards\ModuleToFlashcards.exe --verify
```

Disconnect networking, start the EXE, call `/health`, process a representative 20-slide PDF, parse the output as exactly 100 flashcards in two 50-row blocks, rerun the same PDF to verify resume, and record elapsed stage timings. These target results are required before labeling the archive GPU-ready.

- [ ] **Step 6: Commit only verification-driven corrections, if any**

Use a focused commit naming the verified defect. Do not create an empty verification commit.
