# Local AI Model Alternatives for Flashcard Generation

This guide compares practical local models for this project's structured
school-module-to-flashcard workflow. It is written for the intended machine:
an NVIDIA GeForce RTX 3060 with 12 GB of VRAM on Windows.

Last reviewed: 2026-10-02.

The short answer is:

- **Try LFM2.5-2.6B and Granite 4.0 H Micro first.** They are the most promising
  alternatives when the goal is to run two or more workers without moving too
  far below the current model's expected quality.
- **Try Granite 3.3 2B when worker count matters more.** It is smaller, but its
  reasoning-style output needs careful testing for unwanted `<think>` text.
- **Use LFM2.5-1.2B only as a speed experiment.** It has the best chance of
  fitting several workers, but its accuracy may be too low for assessment
  content.
- **Do not assume that two workers means twice the speed.** Workers share one
  GPU. The only reliable answer comes from the included benchmark on this
  exact computer.

The current packaged model, Ministral-3-3B-Instruct-2512 Q4_K_M, remains the
reference model until an alternative beats it on both validated throughput and
content quality.

## What the important words mean

You do not need an AI background to use this comparison.

**Model** means the AI file that reads the module facts and writes flashcard
JSON.

**Parameter count**, such as 2B or 3B, is a rough measure of model size. A
larger number can improve ability, but architecture and training matter too. A
3B model is not automatically better than every 2B model.

**GGUF** is the local model-file format used by `llama.cpp` and this project.

**Quantization**, such as Q4_K_M or Q5_K_M, compresses a model. Q4 is usually
smaller and faster to load. Q5 uses more memory and may preserve a little more
quality. Q2 and Q3 are not recommended here because this is an
accuracy-sensitive educational task.

**Context size** is the amount of prompt and answer text a worker can hold at
once. This project currently uses 8,192 tokens. A model that supports only
4,096 tokens is not a safe replacement.

**VRAM** is the memory on the graphics card. The GGUF file size is only the
starting point. Each loaded worker also needs context memory and runtime
overhead, so a 2 GB file does not mean that five workers will fit into 12 GB.

**Schema-constrained JSON** means the model is restricted to the exact JSON
shape the validator expects. This improves structure, but it does not prove
that a question is factually correct or educationally good.

## What a worker means in this application

A worker is one independent copy of the local model context. One worker can
generate one cluster while another worker generates a different cluster.

More workers can help only when all of the following are true:

1. The extra model contexts fit in available VRAM.
2. The GPU has enough unused compute and memory bandwidth to run them
   efficiently.
3. Parallel requests do not slow each other down more than the concurrency
   helps.
4. Output quality and validation success remain acceptable.

The application's `auto` mode measures free VRAM before and after the first
model context is loaded. It keeps at least 1 GiB, or 10% of total VRAM,
whichever is larger, as headroom. It estimates every additional context at
125% of the first measured footprint. This is deliberately conservative.

The earlier Qwen3-4B Q5_K_M test is an important warning. On this exact RTX
3060, `auto` selected only one worker. At 8,192 context and two clusters per
call, it produced 2.24 valid clusters per minute with a 100% first-attempt
validation rate. A smaller model may make two or more workers possible, but it
must still beat that real result.

Reaching 10 clusters per minute from 2.24 requires about **4.5 times** the
validated throughput. Even an impossible-to-guarantee perfect doubling from
two workers would reach only about 4.5 clusters per minute. The target will
probably require a model that is faster by itself *and* useful parallelism,
not just a second worker.

The worker counts below therefore mean **the order in which counts should be
tested**, not a promise that the count will fit or be faster.

## Recommended test order

| Order | Model | Recommended GGUF | File size | Worker tests on 12 GB RTX 3060 | Why test it |
|---:|---|---|---:|---|---|
| Reference | [Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF) | Q4_K_M | 2.15 GB | 1, then 2, then `auto` | Current packaged baseline; do not replace it without a measured win. |
| 1 | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) | Q4_K_M | 1.67 GB | 1 → 2 → 3 → `auto` | Best first speed/size experiment; designed for on-device use. |
| 2 | [Granite 4.0 H Micro](https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF) | Q4_K_M | 1.94 GB | 1 → 2 → 3 → `auto` | Strong instruction and extraction focus with a permissive license. |
| 3 | [Granite 3.3 2B Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF) | Q4_K_M | 1.55 GB | 1 → 2 → 3 → 4 → `auto` | Smaller mature fallback; potentially more workers, but reasoning output is a risk. |
| 4 | [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | [Q4_K_M conversion](https://huggingface.co/tensorblock/Phi-4-mini-instruct-GGUF) | 2.49 GB | 1 → 2 → `auto` | Quality-oriented 3.8B candidate; less likely to support many workers. |
| 5 | [SmolLM3-3B](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF) | Q4_K_M | 1.92 GB | 1 → 2 → 3 → `auto` | Compact and easy to test in llama.cpp; quality is more uncertain for this task. |
| 6 | [EXAONE 3.5 2.4B Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF) | Q4_K_M | 1.64 GB | 1 → 2 → 3 → `auto` | Good size for concurrency; custom license and English-task quality need review. |
| 7 | [Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct-GGUF) | Q4_K_M | 2.01 GB | 1 → 2 → 3 → `auto` | Viable older fallback with a 32K context window. |
| 8 | [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) | Q4_K_M | 0.73 GB | 1 → 2 → 3 → 4 → `auto` | Best chance of several workers; highest risk of weaker questions and distractors. |
| 9 | [Gemma 3 4B IT QAT](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | Q4_0 | 3.16 GB | 1 → 2 only if VRAM allows | Quality-first option, but a poor match for the multi-worker goal. |
| 10 | [Gemma 4 E2B IT QAT](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf) | Q4_0 | 3.35 GB | 1 → 2 only if VRAM allows | Newer quality experiment; larger and needs a runtime-compatibility check. |
| 11 | [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [Community Q4 conversion](https://huggingface.co/professorf/Llama-3.2-3B-Instruct-gguf) | about 2.02 GB | 1 → 2 → 3 → `auto` | Widely understood fallback, but older and not the strongest first choice. |

File sizes are the published model-file sizes, not total VRAM use. “About” is
used where the linked GGUF is a community conversion rather than an official
publisher artifact.

No public general-purpose score can prove that an alternative is “as accurate
as Qwen3-4B” at this particular flashcard task. The fixed three-module test and
human review are the evidence that matters here.

## The best candidates in plain language

### 1. LFM2.5-2.6B: best first multi-worker experiment

This is the first model to benchmark if the main goal is two or more workers.
Its Q4_K_M file is roughly 480 MB smaller than the current Ministral file. It
is intended for on-device use and has enough stated context capacity for this
project.

Start with one worker to learn its true single-worker speed and quality. Then
test two and three. Do not jump directly to the highest count because that
would hide whether the model itself or concurrency caused a change.

Important cautions:

- It uses Liquid AI's custom LFM license rather than Apache 2.0.
- Its hybrid architecture must be tested with this project's pinned
  `llama-cpp-python` build.
- A smaller file makes more workers plausible, not guaranteed.

### 2. Granite 4.0 H Micro: best balanced alternative

Granite 4.0 H Micro is attractive for structured educational work because its
official description emphasizes instruction following, extraction,
classification, summarization, and tool use. Those abilities are closer to
this workflow than creative conversation is. It is Apache 2.0 licensed.

Its Q4_K_M file is only modestly smaller than Ministral, so two workers are a
reasonable test. Three workers are an experiment, not an expectation. Its
hybrid attention/Mamba architecture can have different performance from a
standard transformer, so only the application benchmark can settle the speed
question.

### 3. Granite 3.3 2B: best smaller conservative fallback

At 1.55 GB for Q4_K_M, Granite 3.3 2B has a stronger chance of fitting three
or possibly four contexts. It is Apache 2.0 licensed and supports long
contexts.

However, the model can produce reasoning inside `<think>` tags. This project
needs compact JSON, not a long visible reasoning trace. Reject it if the
benchmark shows think markers, wasted output tokens, truncated JSON, or slower
validated throughput.

### 4. Phi-4-mini: quality first, worker count second

Phi-4-mini-instruct is a 3.8B model whose official card emphasizes precise
instruction following and function calling. That makes it a sensible quality
candidate. The Q4_K_M GGUF is about 2.49 GB, so it is less likely than the 2B
models to run several useful workers on this GPU.

The linked GGUF is a third-party conversion, not an official Microsoft GGUF.
Before packaging it, pin the exact revision, byte size, and SHA-256 just as the
current model is pinned.

### 5. LFM2.5-1.2B: maximum concurrency experiment

This is the smallest serious candidate in the list. Several contexts may fit,
but that is not the same as producing accurate assessment questions. Small
models are more likely to create weak distractors, repeat concepts, miss
instructions, or make statements that pass JSON validation but are
educationally wrong.

Use it to discover the upper speed limit of the pipeline. Do not make it the
default unless a human review confirms that its cards are comparable to the
reference model.

## Secondary candidates

**SmolLM3-3B** is easy to test with the llama.cpp-published GGUF and has an
Apache 2.0 license. Its own published general benchmarks place it below
Qwen3-4B on several instruction and knowledge measures. That does not prove it
will fail this specialized task, but it makes it a higher-risk replacement.

**EXAONE 3.5 2.4B Instruct** is small enough to make multiple workers
plausible. It was designed primarily for English and Korean and uses the
EXAONE license, so both English flashcard quality and deployment terms need to
be checked.

**Falcon3-3B-Instruct** fits the context requirement and has a reasonable Q4
size. It is an older candidate with a custom Falcon license, so the newer LFM
and Granite options should be tested first.

**Gemma 3 4B IT QAT** may preserve good quality at Q4 because its quantization
was considered during training. Its 3.16 GB file makes it a weak choice when
the specific goal is more workers. Access also requires accepting Google's
Gemma terms.

**Gemma 4 E2B IT QAT** is another quality-oriented option, but its text-only
GGUF is about 3.35 GB. Do not download the optional vision projector for this
text-only application. Because this model family is newer, confirm support in
the pinned runtime before judging its output.

**Llama 3.2 3B Instruct** is a familiar general-purpose fallback, but the
listed GGUF is a community conversion and the model uses Meta's custom Llama
license. It offers no clear reason to test it before the top candidates.

## Models deliberately left out

This is a screened catalogue, not a list of every file on Hugging Face. A
literal list would become outdated quickly and would include many unsafe or
irrelevant choices.

The following are intentionally excluded:

- **[NVIDIA Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct):**
  its 4,096-token context is below the application's current 8,192-token
  requirement.
- **Models larger than about 7B:** even when one quantized copy fits, multiple
  8K contexts are not a realistic goal on this 12 GB card.
- **Models below 1B:** their worker count may look impressive, but factual and
  instructional quality is too risky for the main shortlist.
- **Base models:** choose an Instruct or IT checkpoint. Base models are not
  trained to follow the application's conversation and JSON instructions.
- **Reasoning-first variants:** they often spend tokens on hidden or visible
  reasoning that does not belong in the JSON response.
- **Q2 and Q3 quantizations:** the memory saving is not worth making the
  educational-quality question harder.
- **Unverified fine-tunes, “abliterated” models, and anonymous conversions:**
  they add provenance, safety, and reproducibility risks.
- **Vision projector files:** the input is structured text, so multimodal
  components consume storage or memory without helping this workflow.
- **Other Qwen checkpoints:** Qwen3-4B is retained only as a measured historical
  reference because this search is specifically for a non-Qwen replacement.

## How to decide whether more workers actually help

Run the same three-module benchmark for every candidate. Keep the seed,
context, cluster batch size, retry count, review setting, module order, and
input graph hashes unchanged.

First test a single worker:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run `
  --manifest benchmarks\modules.json `
  --output benchmarks\candidate-workers-1.json `
  --n-ctx 8192 `
  --workers 1 `
  --clusters-per-call 2
```

Then change only the worker count and output filename:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.run `
  --manifest benchmarks\modules.json `
  --output benchmarks\candidate-workers-2.json `
  --n-ctx 8192 `
  --workers 2 `
  --clusters-per-call 2
```

Repeat with three or four workers only for models whose table row recommends
it. Finally, run `--workers auto` as a separate test. Close browsers, games,
video tools, and other GPU-heavy programs first. The captured Qwen run showed
about 6.6 GiB of the 12 GiB total in use, leaving about 5.6 GiB; desktop apps
reduce that remaining space further.

For each result, compare:

- `valid_clusters_per_minute`: the main speed result;
- `actual_workers`: how many contexts really loaded;
- `first_attempt_cluster_pass_rate`: whether quality became less reliable;
- `retry_count`: retries consume time and can hide weak generation;
- `length_finishes` and `context_errors`: both should remain zero;
- `visible_think_markers`: should remain zero;
- the final 100 cards, reviewed by a person for factual accuracy, clear
  wording, answer leakage, distractor quality, and duplicates.

A candidate should not become the default unless all three modules pass, no
context or length failures occur, the first-attempt pass rate remains within
two percentage points of the reference, retries do not increase, and human
review accepts the cards.

The target is 10 valid clusters per minute. Compare that target with measured
validated throughput, not raw tokens per second. A model that is twice as fast
but needs frequent retries may finish the real job more slowly.

## Reading the worker result

Use these interpretations:

- **Two workers load and throughput rises substantially:** keep testing two.
- **Two workers load but throughput barely changes:** the GPU is saturated;
  use one worker.
- **Two workers are slower:** the contexts are competing for GPU resources;
  use one worker.
- **Two workers fail to load:** return to one or test a smaller model. Do not
  reduce context below the measured prompt-plus-answer requirement merely to
  force a second worker.
- **More workers are faster but quality falls:** reject the setting. The goal
  is valid, accurate clusters per minute, not the largest worker number.
- **`auto` selects one while a manual two-worker test succeeds:** compare
  stability over repeated runs before changing the conservative memory rule.

## How a model replacement is packaged

Dropping a new GGUF into `models/` is not enough. The application intentionally
locks one reproducible model. A real migration must update the model repository,
filename, immutable revision, expected byte size, SHA-256, model label,
packaging lock, tests, and documentation. It must also confirm the model's chat
template and whether it emits reasoning markers.

Use an official publisher GGUF when one exists. If only a community conversion
exists, inspect its provenance, pin it immutably, hash it, and treat that extra
supply-chain risk as part of the decision.

## Recommended decision

Keep Ministral as the reference while running this sequence:

1. Benchmark LFM2.5-2.6B Q4_K_M with one, two, three, and automatic workers.
2. Benchmark Granite 4.0 H Micro Q4_K_M with the same sequence.
3. If neither reaches the required speed, benchmark Granite 3.3 2B Q4_K_M.
4. Use LFM2.5-1.2B only to test whether very small models can approach the
   10-cluster-per-minute target.
5. Use Phi-4-mini or Gemma only if quality matters more than adding workers.

The best model is the one that produces the most **human-acceptable, validated
clusters per minute** on this RTX 3060. Model size and worker count are only
clues.

## Primary model sources

- [Mistral AI: Ministral-3-3B-Instruct-2512 GGUF](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF)
- [Liquid AI: LFM2.5-2.6B GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- [Liquid AI: LFM2.5-1.2B-Instruct GGUF](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)
- [IBM: Granite 4.0 H Micro GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF)
- [IBM: Granite 3.3 2B Instruct GGUF](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF)
- [Microsoft: Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [ggml-org: SmolLM3-3B GGUF](https://huggingface.co/ggml-org/SmolLM3-3B-GGUF)
- [LG AI Research: EXAONE 3.5 2.4B Instruct GGUF](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF)
- [Technology Innovation Institute: Falcon3-3B-Instruct GGUF](https://huggingface.co/tiiuae/Falcon3-3B-Instruct-GGUF)
- [Google: Gemma 3 4B IT QAT Q4 GGUF](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)
- [Google: Gemma 4 E2B IT QAT Q4 GGUF](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [Meta: Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Model publishers can update model cards and repository files. Before any future
migration, re-check the license, exact artifact, runtime support, file size,
revision, and checksum.
