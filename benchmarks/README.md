# Flashcard Generation Benchmark

This benchmark compares prompt cost, truncation, validation, and completed clusters on the same fixed graph inputs. It starts at graph facts and ends with both validated 50-card CSV parts. Graph inputs and the local `modules.json` manifest are ignored by Git because they can contain user-supplied course material; only aggregate reports and graph hashes are committed.

Create `benchmarks/modules.json` with at least three entries. A graph path is relative to the manifest directory unless absolute:

```json
{
  "modules": [
    {"graph": "inputs/course-a-m1.json", "course_code": "COURSE_A", "module_number": "1"},
    {"graph": "inputs/course-a-m2.json", "course_code": "COURSE_A", "module_number": "2"},
    {"graph": "inputs/course-b-m1.json", "course_code": "COURSE_B", "module_number": "1"}
  ]
}
```

Run the pinned local model from the repository root:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run --manifest benchmarks/modules.json --output benchmarks/baseline.json
```

The default is three modules, seed 42, 8,192 context tokens, one cluster worker, three attempts, and final review enabled. The pinned Ministral instruct model uses its embedded chat template, schema-constrained JSON, and temperature 0.05. To measure the RTX 3060's automatic worker policy separately, pass `--workers auto` on that computer and save a separate report. Keep model, graph hashes, seeds, context, worker setting, module order, and review setting identical when comparing two code stages. `valid_clusters_per_minute` includes planning, generation, review, and validated CSV writing but excludes model loading; `cluster_generation_per_minute` measures cluster generation only. First-attempt pass rate means accepted without another model response, including safe local repairs.

`chars_per_completion_token` compares visible response length with recorded completion tokens; `visible_think_markers` counts responses containing a literal `<think>` marker. These metrics can detect visible reasoning leakage, but cannot prove whether an engine internally spent unreported tokens on reasoning.

A failed module is recorded with its exception type and makes the command exit nonzero. Full failure details go to the console, not the JSON report, so card content is not captured in committed metrics. No benchmark report should be treated as representative of future course modules solely because it passed three available samples.

## Optional multi-cluster calls

`--clusters-per-call` accepts `1`, `2`, or `5`; `1` remains the default. For a controlled comparison on the RTX 3060, run all three sizes with the same graph manifest, model, context, worker count, and review setting. For example, use `--workers 1` to isolate the effect of batching, then repeat with a fixed worker count the 3060 can actually load:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run --manifest benchmarks/modules.json --workers 1 --clusters-per-call 1 --output benchmarks/after_stage6_batch1.json
.\.venv\Scripts\python.exe -m benchmarks.run --manifest benchmarks/modules.json --workers 1 --clusters-per-call 2 --output benchmarks/after_stage6_batch2.json
.\.venv\Scripts\python.exe -m benchmarks.run --manifest benchmarks/modules.json --workers 1 --clusters-per-call 5 --output benchmarks/after_stage6_batch5.json
```

Check each report's `settings`, top-level `first_attempt_cluster_pass_rate`, `metrics.overall.retry_count`, length/context error counts, and `cluster_generation_per_minute`. The generic `metrics.overall.first_attempt_pass_rate` is per model call, so it is not comparable across batch sizes. A five-cluster call may be split into smaller calls automatically if it cannot reserve its completion budget or its output is truncated. Count only completed, validated clusters when judging speed; a lower number of model calls alone is not a speedup. Keep size `1` in production unless the 3060 evidence meets the quality and truncation gates in `REPORT.md`.

For the end-to-end CLI, pass `--clusters-per-call 2` or `5` to `pipeline.py` or `main.py`. The API's corresponding environment variable is `MODULE_FLASHCARDS_CLUSTERS_PER_CALL`; it also accepts only `1`, `2`, or `5` and defaults to `1`. The API enables final grounding review when batching is requested, because one batch prompt contains multiple concepts' facts.
