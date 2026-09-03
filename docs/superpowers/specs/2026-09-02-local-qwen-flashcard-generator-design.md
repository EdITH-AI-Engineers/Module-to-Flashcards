# Local Qwen Flashcard Generator Design

Date: 2026-09-02
Status: Approved in conversation; awaiting written-spec review

## Summary

Extend the existing knowledge-graph extractor with a local flashcard-generation pipeline. The pipeline downloads the official `Qwen/Qwen2.5-3B-Instruct-GGUF` checkpoint file `qwen2.5-3b-instruct-q8_0.gguf` from Hugging Face, runs it with `llama-cpp-python`, and generates assessment CSV from a supplied knowledge-graph JSON file.

The application will not ask the 3B model to generate all 100 questions in one completion. It will first select 20 supported concepts, generate one five-question cluster per request, validate each cluster deterministically, run a final model-assisted grounding and duplication review, and write output only when the complete module passes validation.

## Goals

- Generate exactly 100 college-level assessment questions for each module when the graph supports 20 distinct concepts.
- Use exactly 20 concepts, five questions per concept, and one valid UUID cluster per concept.
- Support only `multiple-choice`, `identification`, and `true-false` questions.
- Produce two copy-paste-ready, 50-row CSV blocks without splitting a cluster.
- Ground all assessed facts in the supplied graph and avoid external knowledge.
- Make local generation reliable enough for a 3B model through narrow prompts, deterministic validation, targeted retries, and final review.
- Run locally after the initial model download.

## Non-goals

- Replacing the existing REBEL graph extractor.
- Training, fine-tuning, or modifying the Qwen checkpoint.
- Providing a graphical interface.
- Supporting arbitrary assessment schemas or question types.
- Silently padding insufficient source material with duplicate or invented concepts.
- Guaranteeing perfect semantic judgment from a small language model; the design reduces and surfaces risk but still requires human academic review before publication.

## Model and Runtime

- Hugging Face repository: `Qwen/Qwen2.5-3B-Instruct-GGUF`
- Filename: `qwen2.5-3b-instruct-q8_0.gguf`
- Runtime: `llama-cpp-python`
- Model directory: `models/`, excluded from version control
- Default context window: 32,768 tokens
- Default temperature: low, initially `0.2`, to favor instruction adherence
- Default GPU offload: `n_gpu_layers=-1`, which requests all possible layers from a GPU-enabled build; CPU-only builds remain valid and execute on the CPU

`huggingface_hub.hf_hub_download` will fetch the exact repository and filename into the project model directory if the file is absent. Existing model files will be reused. The application will not download alternative checkpoints automatically.

## Command-Line Interface

The primary command will have this form:

```text
python main.py output/knowledge_graph.json --course-code CPE0021 --module-number 1
```

Required inputs:

- Positional graph JSON path.
- `--course-code`, unless an exact value exists in graph metadata.
- `--module-number`, unless an exact value exists in graph metadata.

Important optional inputs:

- `--output`, defaulting to `flashcards/module_<module-number>.txt`.
- `--model-dir`, defaulting to `models/`.
- `--max-retries`, defaulting to `3`.
- `--seed`, for reproducible sampling where the runtime supports it.
- `--n-gpu-layers`, defaulting to `-1` to request maximum available offload.
- `--skip-final-review`, an explicit speed-over-quality override; final review is enabled by default.

Course codes and module numbers are treated as strings and preserved exactly. Command-line values count as user-provided values. The precedence rules are:

- Course code: command-line value first, then graph metadata.
- Module number: graph metadata first, then command-line value.

If either value remains missing or is genuinely ambiguous, the command exits with a focused message naming only the missing value.

## Components

### `main.py`

Coordinates argument parsing, graph loading, model setup, concept planning, cluster generation, final review, CSV assembly, and output writing. It will contain orchestration rather than prompt or validation details.

### `local_qwen.py`

Owns the model repository constants, exact-file download, local-path resolution, `llama-cpp-python` initialization, chat-template calls, sampling configuration, and extraction of assistant content. It exposes a small completion interface so tests can replace the real model with a fake backend.

### `flashcard_prompt.py`

Contains the stable system prompt and focused user-prompt builders for:

1. Concept planning.
2. Five-card cluster generation.
3. Retry correction using concrete validator errors.
4. Final grounding and duplication review.

The prompt will preserve the user's educational, source, type, direct-stem, difficulty, explanation, hint, and equation requirements. It will instruct Qwen to return JSON only for internal stages. CSV formatting is deliberately delegated to Python.

### `flashcard_validator.py`

Defines the internal card schema, parses model JSON, performs deterministic validation, detects exact and near duplicates, validates module-wide invariants, and returns actionable error messages. It does not call the model.

### CSV assembler

The CSV assembler may live in `flashcard_validator.py` initially because it consumes validated card records and is small. It uses Python's `csv` module, fixes the exact 13-column order, writes one row per question, and creates two labeled 50-row blocks. If the module grows beyond a focused size, it will be separated into `flashcard_csv.py` without changing public behavior.

## Internal Data Contracts

### Concept plan

Each planned concept contains:

- A concise unique concept name.
- A list of relevant graph node or edge identifiers.
- A compact set of verbatim graph-supported facts used as generation evidence.
- Five distinct assessment approaches selected from recall, comparison, classification, application, scenario analysis, cause/effect, misconception detection, conditions, consequences, and reversed reasoning.

The plan must contain exactly 20 semantically distinct concepts. Evidence is retained internally and never copied as provenance into question text or CSV fields.

### Flashcard

The model returns these semantic fields for each card:

- `type`
- `question`
- `correct_option`
- `wrong_option_1`
- `wrong_option_2`
- `wrong_option_3`
- `is_true`
- `expalanation`
- `hint`
- `difficulty`
- `assessment_approach`

`assessment_approach` is used for validation and is not emitted to CSV. Python supplies `cluster`, `course code`, and `module number` so the model cannot corrupt them.

### Review result

The final reviewer returns a list of cluster UUIDs that require regeneration, with one or more concise reasons chosen from unsupported claim, semantic duplication, answer leakage, invalid distractor, misleading explanation, or insufficient variation. An empty list means the model found no remaining issues.

## Prompt and Generation Protocol

### Stage 1: Concept planning

Qwen receives the graph's factual content and the module identity. It must select exactly 20 explicitly supported, non-overlapping concepts and attach supporting graph facts to each. It is told to report insufficient content instead of inventing a plan when fewer than 20 concepts are supported.

The deterministic validator checks count, required fields, empty evidence, repeated identifiers, repeated normalized names, and obvious concept duplication. Concept-planning failures are retried with the precise problems found.

### Stage 2: Cluster generation

The application creates a UUID for the concept, then sends Qwen only:

- The stable assessment rules.
- The selected concept name.
- Its approved supporting facts.
- Its five required assessment approaches.
- Any graph relationships needed to distinguish correct answers from distractors.

Qwen returns exactly five JSON card objects. Every cluster must contain at least one multiple-choice, one identification, and one true-false card. The remaining two types vary rather than following one fixed distribution across all clusters.

The model is instructed to check its own five cards against the evidence and rules before returning JSON, but deterministic Python validation remains authoritative for structural rules.

### Stage 3: Targeted retries

If parsing or validation fails, the retry prompt includes:

- The same concept and evidence.
- The prior candidate output.
- Only the concrete validation errors.
- A requirement to return a complete replacement cluster, not a patch or commentary.

Each cluster gets at most the configured number of attempts. A permanently invalid cluster stops the run and reports its concept and unresolved errors.

### Stage 4: Final review

After 20 clusters pass deterministic validation, Qwen performs two bounded reviews. Five grounding-review calls each receive four complete clusters and their supporting evidence; they check unsupported claims, answer leakage, invalid distractors, misleading explanations, and insufficient within-cluster variation. One global-duplication call then receives all 100 question stems with their concept names and cluster UUIDs, but no answer content, and flags semantic duplicates across the entire module.

Each flagged cluster is regenerated once using all applicable review reasons and then deterministically validated with the normal retry limit. The application runs one final deterministic module validation after all replacements. It does not repeat the model-assisted review or enter an unbounded review loop.

### Stage 5: CSV assembly

The first ten complete clusters become `Module <n>.1`, and the remaining ten become `Module <n>.2`. Each block includes the exact header and 50 data rows. The exact header is:

```text
Type,Question,Correct Option,Wrong Option 1,Wrong Option 2,Wrong Option 3,is_true,expalanation,hint,difficulty,cluster,course code,module number
```

The misspelled `expalanation` field is preserved because it is part of the required contract. The output file contains only the module labels and CSV blocks specified by the user's output contract.

## Deterministic Validation

### Module invariants

- Exactly 20 concepts.
- Exactly 20 unique valid UUID clusters.
- Exactly five rows per cluster.
- Exactly 100 rows in total.
- Exactly ten complete clusters and 50 rows per CSV block.
- Exact course code and module number on every row.

### Cluster invariants

- Exactly five cards.
- At least one card of each allowed type.
- Five distinct `assessment_approach` values.
- No exact or near-duplicate question stems.
- No repeated multiple-choice option within a question.

### Type-specific invariants

- Multiple-choice: one concise correct option, three non-empty distinct wrong options, and empty `is_true`.
- Identification: one concise identifiable correct option; all wrong options and `is_true` empty.
- True-false: declarative statement only; answer-option fields empty; `is_true` exactly integer `0` or `1`.

### Text-quality invariants

- Difficulty is integer `1`, `2`, or `3`.
- Question, explanation, and hint are non-empty.
- Identification answers are constrained to a short phrase rather than a sentence.
- Questions do not contain question numbers, answer labels, choices, or line breaks.
- True-false stems do not contain evaluation instructions.
- Direct question stems are used where the type expects a question.
- Banned wrapper and source-framing phrases are rejected case-insensitively.
- Provenance and presentation terms are rejected when they expose a graph, source, slide, file, chunk, citation, URL, or module wrapper.
- Hints are rejected when they contain a normalized copy of the correct answer.

### Duplicate detection

Exact duplicates are detected after Unicode normalization, case folding, punctuation removal, and whitespace normalization. Two different stems are treated as near duplicates when their normalized token-set Jaccard similarity is at least `0.85` and their normalized character-sequence similarity is at least `0.88`. Module-wide comparison covers all 100 questions. Tests will lock these thresholds against representative duplicate and non-duplicate cases. Model-assisted final review supplements the heuristics for semantic duplicates that wording-based checks cannot reliably identify.

## Error Handling

- Missing or unreadable graph: exit before loading the model.
- Invalid graph shape: report the missing structural field.
- Missing identity value: ask only for the course code or module number that is missing.
- Model download failure: retain no false success state and show the repository, filename, and underlying error.
- Model load failure: show the model path and actionable runtime error.
- Malformed JSON: retry with a JSON-format error.
- Insufficient supported concepts: stop with a message that more content is required.
- Exhausted generation retries: stop and name the failed concept and remaining validation errors.
- Final-review failure: do not write partial CSV unless the user later requests a dedicated diagnostic option.

Output is assembled fully in memory and written atomically only after final validation, preventing a failed run from appearing complete.

## Testing Strategy

Automated tests do not load or download the model.

### Unit tests

- Prompt builders include the required constraints and only the intended evidence.
- Model JSON parsing accepts the contract and rejects surrounding prose or invalid shapes.
- Every module, cluster, type-specific, text-quality, and duplicate rule has passing and failing cases.
- UUID assignment produces one valid UUID per concept and reuses it for five rows.
- CSV escaping handles commas, quotation marks, Unicode, and empty fields.
- Output splitting produces two headers, two labels, 50 rows per block, and no split cluster.
- Retry feedback contains actionable errors and does not loosen constraints.

### Pipeline tests

A fake model backend supplies planned responses, malformed responses, corrected retries, insufficient-content responses, and reviewer flags. End-to-end tests verify orchestration without inference cost.

### Optional smoke test

An explicit command performs a small real-model request against the downloaded GGUF and validates the response contract. It is excluded from the default test suite.

## Dependencies and Project Files

Expected files:

- `main.py`
- `local_qwen.py`
- `flashcard_prompt.py`
- `flashcard_validator.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `tests/`

Expected direct Python packages:

- `llama-cpp-python`
- `huggingface-hub`
- `pytest` for development and tests

The standard library provides argument parsing, JSON, CSV, UUIDs, temporary-file replacement, paths, and similarity primitives.

## Operational Considerations

- The Q8 model file is several gigabytes, so the first run requires sufficient disk space and network access.
- CPU-only inference is supported but may be slow for concept planning, 20 cluster calls, retries, and final review.
- GPU acceleration depends on installing a compatible `llama-cpp-python` build; model behavior must not depend on GPU availability.
- The graph and generated assessment remain local after the checkpoint download.
- Fixed seeds improve repeatability but do not guarantee byte-identical output across runtime versions or hardware backends.

## Success Criteria

The implementation is complete when:

1. A fresh setup can obtain the exact Q8 checkpoint from the official Hugging Face repository.
2. The CLI can generate assessment output from a valid graph using the local model.
3. The final output has exactly the requested 13 columns, two 50-row blocks, 20 clusters, and 100 validated cards.
4. Structurally defective model output is corrected through bounded retries or fails without writing partial results.
5. Insufficient source content is reported rather than padded.
6. All model-free automated tests pass.
7. The real-model smoke command is documented and can be run separately.
