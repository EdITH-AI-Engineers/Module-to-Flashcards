# Flashcard generation performance report

## Status on 2026-10-01

The prompt-reduction stages and optional 1/2/5-cluster batching code are implemented on `clusters`. The available coding laptop has not produced a GPU benchmark. The separate RTX 3060 12 GB run is still required before changing defaults or claiming a throughput gain. The three supplied module samples are the only available benchmark candidates and cannot establish performance or quality for all future modules.

The benchmark runner records prompt/completion tokens by task, truncation and context failures, retries, first-attempt cluster pass rate, and validated clusters per minute. Each module must finish as two validated 50-card CSVs. The benchmark inputs and model must stay fixed across comparisons; reports record graph hashes and settings so mismatches are visible.

| Stage | Commit | GPU result |
| --- | --- | --- |
| Baseline instrumentation | `bf3de20` | Pending |
| Compact payloads | `dce6b89` | Pending |
| Task-specific prompts | `268b43d` | Pending |
| Per-type schemas | `663691c` | Pending |
| Optional no-think and context preflight | `7c80302` | Pending |
| Partial retries | `a319f67` | Pending |
| Optional cluster batching | Current working branch | Pending |

No stage was reverted based on performance data because none has been measured on the 3060. The deterministic replacement for global duplicate review was deliberately **not** applied: its local near-duplicate rule excludes one-word substitutions that the existing review may catch.

## Required 3060 checks

1. Use the same frozen, locally available `benchmarks/modules.json` and graph files for baseline and every later run. Verify all three modules pass. Run the baseline at `bf3de20` with `--workers 1 --clusters-per-call 1` (the older runner implicitly uses size 1), then each stage at its listed commit with the same settings. Save the JSON reports under `benchmarks/`.
2. On the final `clusters` version, compare `--clusters-per-call 1`, `2`, and `5` with `--workers 1`, then repeat with a *fixed* worker count the GPU can load. Keep context and all other settings equal. See [benchmark instructions](benchmarks/README.md).
3. Measure the automatic GPU-worker policy separately with `--workers auto`, because its actual loaded workers may differ from a fixed-worker trial. Report both generation-only and end-to-end validated clusters per minute.
4. Recommend hard no-think and Qwen non-thinking sampling only after separate controlled trials. Both remain optional and off by default.

Acceptance requires at least 40% lower mean prompt tokens for cluster calls and 25% lower overall mean prompt tokens than baseline, zero length finishes and context errors, first-attempt validation pass rate no more than two percentage points lower, and no increase in total retries. A 4-cluster/minute result on the RTX 3060 remains an unverified target, not a claim. Size 1 remains the production default until a 3060 comparison shows a better size meeting these gates.
