# GPU-Aware Cluster Workers and Split Flashcard CSVs

## Purpose and success criteria

Generation currently defaults to one cluster at a time, and one CSV contains both labeled 50-card blocks. The desired behavior is to use as many independent Qwen cluster workers as GPU memory can safely support by default, without a fixed five-worker ceiling, and to save the two halves as separate files named `<course>_M<module>-1.csv` and `<course>_M<module>-2.csv`. Each file must contain a normal CSV header and exactly 50 cards (10 complete clusters). The 100-card module-level validation, including cross-half duplicate checks, must remain in force. Faster wall-clock completion is a goal, but additional workers are not a throughput guarantee.

## Considered approaches

1. **Conservative VRAM budgeting (selected).** Observe usable GPU memory and the footprint of the loaded Qwen context, reserve headroom, and choose up to the 20 independent cluster jobs. Keep explicit worker-count override. This avoids knowingly overcommitting VRAM and does not assume that a particular GPU model has a particular capacity.
2. **Load workers until allocation fails.** This might fit one more worker, but a failed native GPU allocation can destabilize the process and lose the module. It is not a suitable default.
3. **Keep the combined CSV as the primary artifact and write two extra files.** This is less disruptive internally but leaves three competing outputs and does not fulfill the requested two-file contract clearly.

## Worker selection and generation

`auto` becomes the default worker setting for the API, portable launcher, and command-line entry points. A positive integer remains an explicit override, with a maximum of 20 because the module contains 20 clusters. Worker selection happens when Qwen is loaded, not merely when arguments are parsed, so the GPU-memory check reflects the available memory at generation time. It checks that CUDA is available, llama.cpp supports GPU offload, and Qwen is configured for full offload. Otherwise, `auto` uses one worker. If memory measurement is unavailable or inconclusive, use one worker and explain that choice in progress output.

For a GPU run, measure free VRAM before and after loading the first Qwen context, derive a conservative per-additional-context budget from the observed footprint, and retain explicit VRAM headroom for the runtime and other applications. Select the largest integer count that fits that budget, clamped to 1–20. Do not launch more workers than there are remaining clusters. Worker initialization occurs before cluster generation; if a predicted worker fails to load, release it and continue with the successfully loaded count, while reporting the reduction. Existing ordered result assembly, final validation, duplicate repair, and rate reporting remain. Progress explicitly reports the selected and actual worker counts so a sequence of completion messages is not mistaken for sequential execution.

The automatic policy, rather than the momentary selected count, is recorded in the reuse manifest. That prevents unrelated changes in available VRAM from invalidating already valid flashcards. Explicit count changes still invalidate cached output as they do now.

## Two-file output contract

The canonical artifact pair is `<course>_M<module>-1.csv` and `<course>_M<module>-2.csv` in the existing `flashcards/<course>/` directory. `--output custom.csv` similarly means `custom-1.csv` and `custom-2.csv`. The files contain ordinary CSV only: the existing 13-column header followed by 50 cards; the old `Module N.1`/`Module N.2` separator lines are not written into the new files. The first 10 complete clusters go to part 1 and the second 10 to part 2. Both parts are rendered from the same fully validated 100-card result, preserving cross-file uniqueness.

Write each part to a temporary file, validate the pair, and replace the canonical files only after both are ready. Since two filesystem replacements cannot be atomic as a unit, resume/reuse validation always requires both valid parts; a partial write is never reported as a completed module. Update artifact paths, invalidation, batch results, API response and status, CLI output, documentation, and tests to treat the pair as one module result. The API returns both paths in order; its existing singular status `output` can continue to identify part 1 for compatibility, with a new `outputs` list for both.

For a valid pre-change combined CSV, the next run converts its two validated blocks into the new pair without invoking Qwen. The original combined file is retained as a legacy backup rather than silently deleted. An invalid or mismatched legacy file cannot be migrated and follows normal regeneration. New runs do not produce a combined file.

## Testing and constraints

Use test-first coverage for worker selection with varied GPU-memory readings, CPU-only fallback, missing measurements, explicit overrides above five, and failed worker initialization. Test the two exact filenames, normal CSV structure, 50/50 split, pair-wide validation and uniqueness, partial-write recovery, valid legacy conversion, batch reuse and invalidation, API outputs/status, and CLI output. Run the full test suite. A real multi-GPU-context throughput result cannot be asserted on a CPU-only development machine; report the chosen count and observed throughput on the target GPU before claiming a speedup.
