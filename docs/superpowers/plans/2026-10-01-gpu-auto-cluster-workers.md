# GPU-Aware Cluster Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cluster concurrency default to the largest conservative count supported by available GPU memory, with no fixed five-worker ceiling.

**Architecture:** Keep an integer worker count inside `FlashcardPipeline`, but let CLI and API accept `auto` or an explicit 1–20 override. `LocalQwenBackend` measures free VRAM around the primary model load; a pure policy function derives a safe count, while the parallel runner tolerates a later allocation failure by using the contexts that loaded successfully.

**Tech Stack:** Python, llama-cpp-python, PyTorch CUDA memory query, pytest; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-01-gpu-auto-workers-and-split-flashcard-csv-design.md` (worker-selection section).

## Global Constraints

- `auto` is the API, portable, and CLI default; manual worker counts remain available.
- The only hard worker ceiling is `CLUSTERS_PER_MODULE == 20`.
- CPU-only, partial-offload, or unmeasurable GPU runs use one worker by default.
- Preserve ordered clusters, full module validation, duplicate repair, cleanup, and progress/rate output.
- Record the policy `auto` in reuse manifests, not the momentary selected count.
- Do not claim a real speedup without a run on the target GPU.

## Review Focus

1. `test_auto_workers_without_cuda_uses_one`: CUDA absent must not probe VRAM or fork.
2. `test_auto_workers_with_no_measurable_footprint_uses_one`: a zero/negative memory delta must never imply unlimited workers.
3. `test_explicit_twelve_workers_survives_cli_api_and_manifest`: a requested count above five must remain intact.
4. `test_parallel_load_failure_uses_loaded_workers_and_closes_them`: one failed fork must not discard already loaded contexts or leak them.
5. `test_auto_worker_count_does_not_invalidate_reusable_manifest`: different free VRAM on a later run must not invalidate valid output.

---

### Task 1: GPU memory policy

**Files:**
- Create: `worker_budget.py`
- Test: `tests/test_worker_budget.py`

**Interfaces:**
- Produces: `parse_cluster_workers(value: str) -> int | str` (`"auto"` or integer 1–20).
- Produces: `choose_cluster_workers(setting: int | str, *, full_gpu: bool, before: tuple[int, int] | None, after: tuple[int, int] | None, cluster_count: int = CLUSTERS_PER_MODULE) -> tuple[int, str]` (count, explanation).
- Policy: for `auto`, require `full_gpu` and `before_free > after_free`; set `context_bytes = before_free - after_free`, `reserve = max(1 GiB, ceil(0.10 * after_total_vram))`, `per_extra = ceil(1.25 * context_bytes)`, then clamp `1 + floor(max(0, after_free - reserve) / per_extra)` to `1..cluster_count`. A manual integer bypasses GPU detection and is clamped only to the available cluster count.

- [ ] **Step 1: Write failing policy tests.** Assert 1 for CPU, partial offload, missing readings, and zero delta; assert >5 on ample measured VRAM; assert exact manual 12; assert invalid strings/0/21 raise `ValueError`.
- [ ] **Step 2: Run red tests.** ` .\.venv\Scripts\python.exe -m pytest -q tests/test_worker_budget.py -p no:cacheprovider --basetemp pipeline_temporary/pytest-worker-budget-red` must fail because the module/functions are absent.
- [ ] **Step 3: Implement the two pure functions.** Keep GPU probing out of this module so policy tests need no CUDA hardware.
- [ ] **Step 4: Run green tests.** The same file must pass.
- [ ] **Step 5: Run the full suite and commit.** Use the existing pytest command with a fresh `--basetemp`; commit only `worker_budget.py` and its tests.

### Task 2: Resolve `auto` at Qwen load and pass the selected count through entry points

**Files:**
- Modify: `local_qwen.py`, `main.py`, `pipeline.py`, `api_server.py`, `batch_pipeline.py`, `flashcard_pipeline.py`
- Test: `tests/test_local_qwen.py`, `tests/test_parallel_settings.py`, `tests/test_main.py`, `tests/test_batch_pipeline.py`

**Interfaces:**
- Consumes: Task 1 policy functions.
- Produces: `LocalQwenBackend.auto_cluster_workers: int` and `auto_cluster_workers_reason: str`; measurements occur immediately before/after constructing `Llama`, using `torch.cuda.mem_get_info(0)` only when PyTorch CUDA and llama.cpp GPU offload are available and `n_gpu_layers == -1`.
- Produces: `PipelineConfig.cluster_workers: int` within 1–20. `main.run` resolves a string `"auto"` from the backend property (fallback 1 for injected backends); explicit integers are unchanged.

- [ ] **Step 1: Write failing propagation tests.** Assert default CLI/API setting is `auto`, `--cluster-workers 12` and API env `12` propagate, model memory readings yield selected count, CPU fallback yields 1, and manifest stores `"auto"` rather than a measured number. Include Review Focus tests 1, 3, and 5.
- [ ] **Step 2: Run red tests.** Run the named tests in `tests/test_local_qwen.py tests/test_parallel_settings.py tests/test_main.py tests/test_batch_pipeline.py`; confirm feature-related failures.
- [ ] **Step 3: Implement load-time measurement and argument resolution.** Keep the first Qwen context's existing behavior; let the batch loader pass `n_threads=None` for `auto`, then budget threads for additional forks once a count is selected. Emit an explicit selection reason. Update validation/error text from 1–5 to 1–20.
- [ ] **Step 4: Run green tests and the full suite.** Both runs must pass; commit only this task's files and tests.

### Task 3: Graceful worker allocation and observable concurrency

**Files:**
- Modify: `flashcard_pipeline.py`, `README.md`
- Test: `tests/test_parallel_clusters.py`, `tests/test_flashcard_pipeline.py`

**Interfaces:**
- Consumes: `PipelineConfig.cluster_workers` resolved in Task 2.
- Produces: selected/actual worker progress messages and the existing ordered tuple of 20 validated clusters. If fork N fails, keep the primary plus forks 1..N-1, run all clusters with those contexts, and close every successful fork in `finally`.

- [ ] **Step 1: Write failing tests.** Assert 12 independent backends can be used concurrently and output order is stable; when loading worker 4 fails (fork index 3), assert three loaded contexts complete all 20 clusters, actual count is reported as 3, and each extra context closes exactly once. Include Review Focus test 4.
- [ ] **Step 2: Run red tests.** Run `tests/test_parallel_clusters.py tests/test_flashcard_pipeline.py` and confirm the new cases fail for the expected behavior.
- [ ] **Step 3: Implement the allocation fallback and progress wording.** Do not suppress validation/retries or change per-worker seeds.
- [ ] **Step 4: Run green tests, full suite, and `git diff --check`.** Update README with `auto`, manual 1–20, and the CPU/GPU caveat; commit the task.

## Verification handoff

Report the selected worker count and actual cluster/minute rate from a real run on the target GPU. The development machine has no CUDA-capable llama.cpp runtime, so unit tests here prove selection and concurrency behavior but not throughput improvement.
