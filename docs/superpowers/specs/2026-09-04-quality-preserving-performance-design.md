# Quality-Preserving Performance Design

Status: Approved in conversation; awaiting written-spec review

Date: 2026-09-04

## Context

The local pipeline must support both single-PDF and multi-PDF runs on a Ryzen 5
class machine with 12 GB of RAM. Input size ranges from roughly 20 to 600
slides. The pipeline currently preserves memory by processing three stages in
sequence and reusing one runtime inside each stage, but a cold run can still be
slow because it may perform OCR and one Q5 normalization completion for every
slide. A normalization failure near the end of a large document also loses the
successful per-slide work from that stage because only the completed module
artifact is resumable.

This design improves throughput and recovery time without changing the Q5
model, generation limits, sampling settings, REBEL beam settings, flashcard
review behavior, or validation rules.

## Goals

- Reduce cold-run time where work can safely overlap or reuse a common model
  prefix.
- Make interrupted or failed 20-600-slide normalization runs resume from
  validated per-slide checkpoints.
- Avoid repeating PDF extraction and OCR when the source PDF and extraction
  settings have not changed.
- Keep peak memory suitable for the 12 GB target device.
- Move the same repository, model files, generated artifacts, and caches to a
  stronger PC without editing source code or machine-specific paths.
- Preserve output ordering and every existing deterministic quality check.
- Record useful stage timings and cache-hit counts without changing the public
  API response shape.

## Non-goals

- Changing from Qwen2.5 3B Q5_K_M to a smaller or lower-quality model.
- Reducing Q5 output limits, validation attempts, or deterministic checks.
- Skipping reviews that are enabled by the selected CLI or API mode.
- Reducing REBEL beam count below the value selected by the caller.
- Running Q5 and REBEL concurrently or keeping both resident in memory.
- Combining several slides or flashcard clusters into a single generation
  request in this phase.
- Claiming a hardware speedup from synthetic sleep-based tests.

## Architecture

The existing staged batch runner remains the top-level coordinator:

1. Resolve the requested hardware profile on the current machine.
2. Extract PDF pages, using a validated extraction cache when possible.
3. Normalize pages with Q5, using validated per-slide checkpoints and a bounded
   in-memory prompt cache.
4. Release Q5 completely.
5. Extract graph relations with REBEL, using a memory-aware batch size.
6. Release REBEL completely.
7. Load Q5 for flashcard generation and retain all current quality checks.

The performance features are optimizations around existing boundaries. They do
not create an alternate output path and they cannot bypass validation.

## Components

### Portable hardware profiles

A small `runtime_profile` component will resolve resource controls once per
process or command. Three profiles are supported:

- `auto` is the default. It detects logical CPU count and currently available
  physical memory on the machine where the command runs. With less than 4 GiB
  available it disables the Q5 RAM cache, uses one OCR worker, and uses REBEL
  batch size one. From 4 GiB to less than 12 GiB available it uses a 256 MiB Q5
  cache, two OCR workers, and REBEL batch size two. At 12 GiB or more available
  it uses a 512 MiB Q5 cache, up to four OCR workers, and REBEL batch size four.
- `low-memory` is the predictable 12 GB laptop fallback: a 256 MiB Q5 cache,
  two OCR workers, REBEL batch size one, and strictly sequential model
  residency.
- `performance` targets a better PC with at least 24 GB RAM or a supported
  accelerator: a 512 MiB Q5 cache, up to four OCR workers, REBEL batch size
  four, and automatic Q5/REBEL GPU selection. Q5 and REBEL remain sequential by
  default so the profile is safe on GPUs with limited VRAM.

Worker counts never exceed the detected logical CPU count. When hardware
detection fails, `auto` resolves to `low-memory`. Explicit values for OCR
workers, Q5 cache capacity, REBEL batch size, Q5 GPU layers, or REBEL device
override the profile field-by-field.

The selected profile changes only resource scheduling and caching. It does not
change prompts, model files, sampling, token limits, beams, retries, reviews,
or validators. CLI entry points expose `--profile auto|low-memory|performance`.
The local API reads the same setting from `MODULE_FLASHCARDS_PROFILE`, defaulting
to `auto`, so a copied deployment adapts without code edits. New CLI overrides
are `--ocr-workers` and `--qwen-cache-mb`; the existing `--kg-batch-size`,
`--n-gpu-layers`, and `--kg-device` remain authoritative. API deployments may
override the same controls through `MODULE_FLASHCARDS_OCR_WORKERS`,
`MODULE_FLASHCARDS_QWEN_CACHE_MB`, `MODULE_FLASHCARDS_KG_BATCH_SIZE`,
`MODULE_FLASHCARDS_N_GPU_LAYERS`, and `MODULE_FLASHCARDS_KG_DEVICE`.

### Normalization cache

A small cache module will own cache keys, JSON serialization, validation, and
atomic writes. For an output such as `structured_module.txt`, its private cache
directory will be a sibling named `.structured_module.normalization-cache`.
This keeps standalone outputs separate even when several files share an output
directory and naturally places batch caches inside the existing per-module
workspace.

The extraction cache will contain:

- a cache format version;
- the SHA-256 digest of the source PDF;
- OCR minimum-character and DPI settings;
- the ordered page number, extraction method, and extracted text for every
  page.

The extraction cache is reusable only when every identity and setting field
matches and the cached page structure validates. It is written atomically as
soon as extraction succeeds, before Q5 normalization begins.

Each successful slide normalization receives a separate checkpoint. Its key
includes:

- normalization cache and prompt versions;
- source PDF digest;
- slide number, extracted text digest, and extraction method;
- Q5 repository and filename;
- context size, maximum output tokens, seed, and temperature.

Cache and manifest model identity uses the model repository, exact filename,
and exact byte size—not the absolute `models` directory path. This avoids
re-hashing a multi-gigabyte GGUF on every run. Cache paths
remain relative to generated workspace directories. Copying the repository and
model to another drive therefore does not invalidate reusable work solely
because its absolute path changed.

The checkpoint stores the successfully parsed JSON value rather than accepting
an unvalidated model response. A cache hit is parsed again through the current
normalization validator before reuse. Invalid, corrupt, or mismatched entries
are ignored and recomputed. Successful checkpoints survive a later slide
failure. `--force` bypasses cache reads but may replace entries with newly
validated results.

Cache write failures are reported as performance warnings and do not invalidate
otherwise correct generated output. Final stage artifacts continue to use the
existing strict manifest and atomic-write rules.

### Bounded Q5 prompt cache

`LocalQwenBackend` will optionally attach a `llama_cpp.LlamaRAMCache`. The
`low-memory` profile caps it at 256 MiB. This reuses tokenized/evaluated common
prompt prefixes across repeated normalization and flashcard requests; it does
not reuse generated answers or change sampling.

If the installed llama-cpp runtime does not expose a compatible cache, cache
construction fails, or a cache state cannot fit within the cap, inference
continues without this optimization. The cache is released with the owning Q5
runtime and is never shared with REBEL.

The cache size will be configurable so devices with less available memory can
set it to zero. Cache capacity is intentionally excluded from artifact and
checkpoint identity because it changes performance only, not generated output.

### Bounded OCR concurrency

Direct PDF text extraction and PyMuPDF document access remain on the caller
thread. Only pages below the direct-text threshold are rendered for OCR. Page
image bytes are handed to the resolved OCR pool, with no more rendered pages in
flight than active workers. The 12 GB profile therefore retains at most two;
the performance profile may retain at most four. This prevents a 600-slide
document from accumulating an unbounded image queue.

OCR results are reassembled by page number, so concurrency cannot reorder the
module. Tesseract discovery/configuration happens once before workers start.
Any OCR failure retains the existing fail-fast behavior, cancels pending work,
and reports the affected page. The DPI, preprocessing, and OCR engine remain
unchanged.

The worker count is profile-controlled and explicitly configurable. Two is the
Ryzen 5/12 GB setting; `performance` may use up to four, bounded by detected CPU
count and the in-flight image limit.

### Memory-aware REBEL batching

Explicit CLI batch sizes continue to win. Otherwise the resolved hardware
profile supplies REBEL batch size one, two, or four. `auto` rechecks available
physical memory immediately before REBEL loads and may lower—but never
increase—the resolved value when memory pressure has grown. If memory detection
is unavailable, it uses one.

Only the number of independent input sequences processed together changes.
Chunk size, overlap, output limit, beam count, model weights, parsing, and graph
validation do not change. The resolved batch size is stored in the reuse
manifest.

### Timing and observability

The batch runner records elapsed time for extraction/normalization, graph
generation, and flashcard generation, plus extraction-cache hits and
slide-checkpoint hits. CLI runs print a concise summary to standard error. API
runs log the same summary server-side while retaining the existing response
keys.

Timing is observational only. It is excluded from cache identity and cannot
change control flow except for the existing user-selected timeout.

## Data flow

For each module:

1. Resolve the portable hardware profile and any explicit overrides.
2. Resolve the private normalization-cache directory.
3. Hash the PDF once and attempt to load its extraction cache.
4. On a miss, extract direct text, run bounded OCR where required, validate the
   ordered pages, and atomically store them.
5. For each readable page in order, compute its slide key and try a validated
   checkpoint.
6. Send only cache misses to Q5. Atomically checkpoint each slide immediately
   after successful parsing.
7. Assemble and validate the complete structured module, then atomically write
   the existing structured text artifact.
8. Continue through the existing staged REBEL and flashcard flow.
9. Write the existing completed-module manifest only after the final
   flashcard artifact passes strict validation.

No cache entry is treated as proof that a final artifact is complete. The
existing artifact validators and completed-module manifest remain authoritative.

## Error handling

- Corrupt cache JSON: ignore that entry, report a cache miss, and recompute.
- Identity or setting mismatch: leave the old entry untouched and write a new
  content-addressed entry after successful recomputation.
- Cache directory or write failure: warn and continue without caching.
- Cached value fails current validation: ignore and recompute with Q5.
- OCR worker failure: cancel pending OCR work and raise a page-specific
  `PdfExtractionError`.
- Q5 failure on slide N: retain checkpoints for slides before N and return the
  existing slide-specific normalization error.
- Low or unreported available memory: choose REBEL batch size one.
- Unknown profile name: reject it before model or OCR work begins.
- Prompt-cache setup failure: close any partial cache state and continue with
  ordinary Q5 inference.

## Quality invariants

- Qwen2.5 3B Q5_K_M remains the exact Q5 model.
- Q5 temperature, seed, context, and maximum output-token values remain the
  selected caller values.
- Every live or cached slide passes `_parse_page` before inclusion.
- Extraction uses the same direct-text threshold, OCR DPI, and Tesseract output.
- REBEL uses the same chunks, overlap, beams, generation limit, parser, and
  graph validator.
- Flashcard planning, 20 cluster generations, retries, optional reviews,
  regeneration, duplicate checks, and final 100-card validation remain intact.
- Cached and uncached executions produce artifacts accepted by the same
  validators and identity checks.

## Testing

### Cache tests

- A validated extraction cache prevents repeated extraction/OCR.
- A changed PDF digest, OCR threshold, or DPI forces extraction.
- A successful slide checkpoint prevents a repeated backend call.
- A failed slide followed by a second invocation reuses earlier slides and
  retries only the missing slide onward.
- Model, prompt, source text, extraction method, token, seed, temperature, or
  context changes invalidate the applicable slide checkpoint.
- Corrupt and structurally invalid cache files are ignored.
- Cache writes are atomic and a write failure does not corrupt final output.
- `--force` bypasses both extraction and slide cache reads.

### OCR tests

- Direct-text pages never enter the OCR pool.
- Active OCR operations and retained rendered pages never exceed the resolved
  worker limit; the low-memory test fixes that limit at two.
- Mixed direct-text and OCR pages retain source order.
- A worker error reports its page and cancels pending work.
- One-worker mode preserves existing sequential behavior.

### Runtime tests

- Profile parsing accepts only `auto`, `low-memory`, and `performance`.
- Hardware detection maps boundary memory values to the documented controls and
  safely falls back to `low-memory` when detection is unavailable.
- Explicit resource settings override only their matching profile fields.
- Worker counts are capped by detected CPU count.
- The configured Q5 prompt cache is attached when supported and omitted when
  disabled or unavailable.
- Prompt-cache failure does not prevent inference and owned resources close.
- Automatic REBEL batching selects two above the memory threshold and one
  below it or when detection is unavailable.
- An explicit batch size is never overridden.
- Q5 and REBEL loader barriers remain sequential.
- Cache and manifest identity remains reusable when only the absolute project or
  model directory changes, while a different model file still invalidates it.

### Integration and benchmark tests

- Single-file and multi-file staged runs preserve input/output order and
  per-file failure isolation.
- Resume behavior reduces deterministic extraction and Q5 call counts.
- All existing structured-module, graph, and 100-card validators stay green.
- The complete model-free test suite remains gating.
- Any wall-clock benchmark using real Q5/REBEL models is opt-in and reports the
  machine, cold/warm state, slide count, stage durations, and cache hits. Tests
  do not download model files, and synthetic sleep timings are not used as
  hardware performance claims.

## Rollout and compatibility

New cache directories live under generated output locations and are not source
artifacts. Existing completed outputs remain reusable through the current
manifest rules. A missing cache behaves exactly like the current implementation.
Users can disable Q5 caching, set OCR workers to one, or explicitly select the
REBEL batch size without changing output formats or API response keys.

For migration, users may copy the repository, the exact GGUF/model files, and
generated output directories to the new PC. Caches are optional: copied caches
are validated through content and model identities, while omitted caches rebuild
automatically. Environment setup remains machine-specific—particularly the
GPU-enabled llama-cpp/PyTorch builds and drivers—but no source or artifact
format conversion is required. On first launch, `auto` resolves controls for the
new hardware; users may select `performance` explicitly after dependencies are
installed.

The first implementation phase ends after the conservative mechanisms above
are verified. Multi-slide or multi-cluster Q5 requests require a separate
quality comparison and are deliberately excluded from this design.
