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

The default is three modules, seed 42, 12,288 context tokens, one cluster worker, three attempts, and final review enabled. To measure the RTX 3060's automatic worker policy separately, pass `--workers auto` on that computer and save a separate report. Keep model, graph hashes, seeds, context, worker setting, module order, and review setting identical when comparing two code stages. `valid_clusters_per_minute` includes planning, generation, review, and validated CSV writing but excludes model loading; `cluster_generation_per_minute` measures cluster generation only. First-attempt pass rate means accepted without another model response, including safe local repairs.

A failed module is recorded with its exception type and makes the command exit nonzero. Full failure details go to the console, not the JSON report, so card content is not captured in committed metrics. No benchmark report should be treated as representative of future course modules solely because it passed three available samples.
