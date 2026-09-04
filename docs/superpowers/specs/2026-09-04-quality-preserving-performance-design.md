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

1. Extract PDF pages, using a validated extraction cache when possible.
2. Normalize pages with Q5, using validated per-slide checkpoints and a bounded
   in-memory prompt cache.
3. Release Q5 completely.
4. Extract graph relations with REBEL, using a memory-aware batch size.
5. Release REBEL completely.
6. Load Q5 for flashcard generation and retain all current quality checks.

The performance features are optimizations around existing boundaries. They do
not create an alternate output path and they cannot bypass validation.

## Components

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
default target profile caps it at 256 MiB. This reuses tokenized/evaluated common
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
image bytes are handed to a two-worker OCR pool, with no more than two rendered
pages in flight at once. This prevents a 600-slide document from retaining a
large image queue.

OCR results are reassembled by page number, so concurrency cannot reorder the
module. Tesseract discovery/configuration happens once before workers start.
Any OCR failure retains the existing fail-fast behavior, cancels pending work,
and reports the affected page. The DPI, preprocessing, and OCR engine remain
unchanged.

The worker count will be configurable from one to two, with two as the Ryzen
5/12 GB profile and one as the conservative fallback.

### Memory-aware REBEL batching

Explicit CLI batch sizes continue to win. API/default automatic selection uses
batch size two only when the platform can report at least 4 GiB of currently
available physical memory immediately before REBEL loads; otherwise it uses
batch size one. If memory detection is unavailable, it uses one.

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

1. Resolve the private normalization-cache directory.
2. Hash the PDF once and attempt to load its extraction cache.
3. On a miss, extract direct text, run bounded OCR where required, validate the
   ordered pages, and atomically store them.
4. For each readable page in order, compute its slide key and try a validated
   checkpoint.
5. Send only cache misses to Q5. Atomically checkpoint each slide immediately
   after successful parsing.
6. Assemble and validate the complete structured module, then atomically write
   the existing structured text artifact.
7. Continue through the existing staged REBEL and flashcard flow.
8. Write the existing completed-module manifest only after the final
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
- At most two OCR operations are active and at most two rendered pages are
  retained.
- Mixed direct-text and OCR pages retain source order.
- A worker error reports its page and cancels pending work.
- One-worker mode preserves existing sequential behavior.

### Runtime tests

- The configured Q5 prompt cache is attached when supported and omitted when
  disabled or unavailable.
- Prompt-cache failure does not prevent inference and owned resources close.
- Automatic REBEL batching selects two above the memory threshold and one
  below it or when detection is unavailable.
- An explicit batch size is never overridden.
- Q5 and REBEL loader barriers remain sequential.

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

The first implementation phase ends after the conservative mechanisms above
are verified. Multi-slide or multi-cluster Q5 requests require a separate
quality comparison and are deliberately excluded from this design.
