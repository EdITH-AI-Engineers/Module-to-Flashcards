# Low-Resource Batch Performance Design

## Goal

Make multi-PDF processing substantially faster and more reliable on lower-end
devices while continuing to produce exactly 100 deterministically validated
flashcards per successful module.

## Current Behavior and Root Cause

The HTTP endpoint accepts multiple PDF uploads, but it loops over them and runs
the complete three-stage subprocess pipeline separately for every file. The
knowledge-graph stage batches chunks within one document, but file-level work is
not a resource-aware batch.

Each module currently loads Qwen once for slide normalization, unloads it, loads
REBEL for graph extraction, unloads it, and then loads Qwen again for flashcard
generation. A multi-file request repeats those expensive loads for every PDF.
The default Qwen checkpoint is a 3.6 GB Q8 model with a 32,768-token context,
REBEL uses three-beam generation, final flashcard review adds six Qwen calls,
and the API imposes a five-minute deadline that is impractical for CPU-only
machines.

## Processing Model

The API will use a staged, sequential batch:

1. Save and validate every upload, recording failures per file.
2. Load Qwen Q5_K_M once and normalize every valid PDF that needs stage 1.
3. Release Qwen, load REBEL once, and build every graph that needs stage 2.
4. Release REBEL, load Qwen Q5_K_M once, and generate flashcards for every
   module that needs stage 3.
5. Return successful output paths and per-file errors in the existing response
   shape.

Files remain sequential within each heavy stage. Running large-model inference
for several PDFs concurrently would increase memory pressure and cause paging
on the devices this change targets. Staging removes repeated model loads without
requiring multiple models in memory at the same time.

The command-line single-PDF pipeline remains supported and resumable. The API
batch runner and command-line runner will share stage functions and artifact
validators so their behavior cannot drift.

## Low-Resource Defaults

- Qwen repository: `Qwen/Qwen2.5-3B-Instruct-GGUF`
- Qwen file: `qwen2.5-3b-instruct-q5_k_m.gguf`
- Qwen context: 8,192 tokens
- Qwen GPU layers: `-1`, allowing a compatible build to offload all possible
  layers while CPU-only builds continue on CPU
- REBEL batch size: 1 in API fast mode, limiting peak activation memory
- REBEL beams: 1 in API fast mode, avoiding three-way beam-search work
- Final model-assisted flashcard review: disabled in API fast mode
- Deterministic schema, grounding, duplication, UUID, cluster-count, and
  100-question validation: always enabled

The command-line tools keep explicit switches for users who want larger
contexts, larger REBEL batches, multiple beams, or final model-assisted review.
The fast path changes optional semantic review, not the output contract.

## Reusable Stage Interfaces

Stage code will accept already-loaded dependencies:

- PDF normalization accepts a shared `LocalQwenBackend`.
- Knowledge-graph extraction accepts a shared REBEL tokenizer, model, and
  selected device.
- Flashcard generation accepts a shared `LocalQwenBackend`.

When no dependency is supplied, each command-line entry point loads what it
needs exactly as it does today. This keeps standalone commands simple while
allowing the batch runner to control resource lifetimes.

Because the Qwen stage-1 and stage-3 prompts use the same checkpoint and
backend configuration, the batch runner creates one backend for each Qwen
phase. It explicitly releases the backend before loading REBEL, and releases
REBEL before the final Qwen phase.

## Resume and Failure Handling

Before each stage, the batch runner validates its expected artifact. A valid
artifact is reused unless force mode is enabled. Recomputing an upstream stage
invalidates reuse of downstream artifacts for that module.

An error in one module records the filename and focused error, marks that module
inactive for later stages, and allows remaining modules to continue. Uploads are
always closed. Partial output files are not reported as successful.

## Timeouts

The five-minute wall-clock deadline is removed from the default low-resource
API path because it terminates valid CPU-only work before completion. Timeout
remains configurable. When a positive timeout is supplied, it measures active
processing time for each module rather than time spent waiting behind other
files in the request.

Standalone command-line execution retains `--timeout`, with `0` meaning no
deadline. Stage failures continue to include the stage name.

## Compatibility

The existing endpoint, request fields, response keys, output locations,
filename-derived module numbers, and resumable artifacts remain unchanged.
Existing Q8 files are not deleted. The first optimized run downloads Q5_K_M;
later runs reuse it.

## Testing and Verification

Automated tests will prove that:

- Q5_K_M is selected and an existing Q5 file is reused.
- Qwen defaults use an 8,192-token context.
- multiple modules reuse one loaded dependency per batch stage.
- heavy stages remain sequential.
- valid artifacts skip their stages.
- one module's failure does not stop other modules.
- fast mode uses one REBEL beam, batch size one, and skips optional final review.
- disabled and positive timeouts behave as documented.
- the existing complete test suite remains green.

A dependency-injected benchmark will compare load counts and orchestration time
for a multi-file request without downloading models. A local smoke test will
verify Q5 model loading when the checkpoint is available; full model inference
will not be required for the automated suite.
