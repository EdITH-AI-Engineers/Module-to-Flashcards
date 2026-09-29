# Validation and Retry Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make flashcard generation preserve meaningful notation, explain exact validation collisions, avoid false cross-type duplicates, repair repeated invalid cards directly, keep truncation retries independent, and reliably fit a grounded 20-concept plan inside the existing 8,192-token context.

**Architecture:** Keep the current pipeline and schemas, but make validation comparisons type-aware and notation-aware. Extend the shared completion loop with independent counters and an optional repeated-valid-JSON handler; cluster generation supplies a targeted single-card repair queue while other stages retain ordinary retries. Use a dedicated compact concept-planning prompt and cap each concept's selected fact IDs without making facts exclusive.

**Tech Stack:** Python 3.11+, standard library, pytest, llama.cpp-compatible structured JSON generation, argparse, PowerShell

**Spec:** `docs/superpowers/specs/2026-09-29-validation-retry-resilience-design.md`

## Global Constraints

- Preserve exactly 20 concepts, 5 cards per concept, and 100 cards per module.
- Preserve all existing card types, grounding rules, final-review behavior, CSV columns, and output paths.
- Keep the default Qwen context at 8,192 tokens; do not solve planning by increasing memory requirements.
- Keep production rules course-neutral: no HCI-, programming-, engineering-, or multimedia-specific vocabulary.
- Preserve punctuation that changes code, equations, units, logical expressions, or symbolic meaning.
- Apply fuzzy question comparison only when both card types are known and equal; exact normalized equality remains invalid across types.
- Treat prior course-corpus question strings as type-unknown and therefore exact-only.
- Never invoke targeted card repair for truncated, empty, or unparseable output.
- Repair the later card in a pairwise duplicate and keep the earlier card as the stable reference.
- Validate the complete reconstructed cluster after every targeted repair queue.
- Preserve every fact ID and full statement in planner input; compaction may remove optional metadata, not evidence.
- `max_retries` counts parseable rejected candidates; `max_truncation_retries` independently counts completion truncations.
- Write a failing regression test before each production change and keep each task independently passing.
- Preserve unrelated user changes. Stage only the files named by the current task.

## File Structure

- Modify `flashcard_validator.py`: notation-aware option equivalence, exact option-collision reporting, and card-type-aware duplicate helpers.
- Modify `flashcard_pipeline.py`: independent retry accounting, repeated-candidate fingerprints, targeted repair queue, and compact planner system-prompt selection.
- Modify `flashcard_prompt.py`: targeted validation-repair prompts and compact concept-plan prompts.
- Modify `flashcard_schema.py`: limit each concept's `fact_ids` array to eight.
- Modify `flashcard_contract.py`: define the shared maximum fact IDs per concept.
- Modify `main.py`: expose the independent truncation retry allowance.
- Modify `pipeline.py`: forward the new CLI option to the flashcard stage.
- Modify `README.md`: explain validation versus truncation attempts.
- Modify focused tests in `tests/test_flashcard_validator.py`, `tests/test_module_validation.py`, `tests/test_flashcard_pipeline.py`, `tests/test_flashcard_prompt.py`, `tests/test_main.py`, and `tests/test_pipeline_runner.py`.

## Review Focus

- A mixed prose/notation value such as `Version 2.0.` must remain prose-like; a decimal point alone must not preserve harmless final punctuation.
- When three or four multiple-choice fields share a value, report every colliding field pair and each original JSON-quoted value.
- Repeated malformed or truncated responses must not call the targeted card repair path.
- If targeted repair creates a new collision or duplicate, complete-cluster validation must reject it.
- Long Unicode fact statements must survive prompt compaction byte-for-byte, and prompt-size tests must use deterministic bounds rather than pretending characters equal model tokens.

---

### Task 0: Preserve and Checkpoint the Existing Provenance Fixes

**Files:**
- Modify only if tests expose an incomplete prerequisite: `flashcard_pipeline.py`
- Modify only if tests expose an incomplete prerequisite: `flashcard_prompt.py`
- Modify only if tests expose an incomplete prerequisite: `flashcard_validator.py`
- Test: `tests/test_flashcard_pipeline.py`
- Test: `tests/test_flashcard_validator.py`

**Interfaces:**
- Preserves the current uncommitted exact provenance-trigger reporting, source-definition cleanup, retry feedback, and hint-leak compatibility.
- Produces a clean prerequisite commit before resilience work begins.

- [ ] **Step 1: Inspect the current diff without changing it**

Run: `git diff -- flashcard_pipeline.py flashcard_prompt.py flashcard_validator.py tests/test_flashcard_pipeline.py tests/test_flashcard_validator.py`

Expected: only the previously approved provenance-trigger and sentence-cleanup changes are present.

- [ ] **Step 2: Run the prerequisite focused tests**

Run: `python -m pytest tests/test_flashcard_validator.py tests/test_flashcard_pipeline.py -q`

Expected: PASS. If a failure is caused by the prerequisite patch itself, fix it with a failing regression test before proceeding; do not mix any of Tasks 1-5 into this checkpoint.

- [ ] **Step 3: Commit only the prerequisite files**

```powershell
git add flashcard_pipeline.py flashcard_prompt.py flashcard_validator.py tests/test_flashcard_pipeline.py tests/test_flashcard_validator.py
git commit -m "fix: preserve provenance validation context"
```

Expected: the five prerequisite files are committed; unrelated files remain untouched.

---

### Task 1: Preserve Meaningful Option Punctuation and Report Exact Collisions

**Files:**
- Modify: `flashcard_validator.py`
- Modify: `tests/test_flashcard_validator.py`

**Interfaces:**
- Add `_is_notation_option(value: str) -> bool`.
- Keep `_canonical_option(value: str) -> str` as the shared equality key, but make terminal punctuation handling conditional on notation.
- Add `_option_collisions(named_options: Sequence[tuple[str, str]]) -> tuple[tuple[str, str, str, str], ...]`.
- `validate_cluster(...)` emits one detailed error per colliding field pair.

- [ ] **Step 1: Add failing notation and prose-equivalence tests**

Add parameterized tests covering:

```python
(
    "cout << value;",
    "cout << value,",
    "x = y + 1;",
    "x = y + 1,",
)
```

as four distinct symbolic options, plus equations, comparisons, ratios, percentages, number-unit expressions, bracketed expressions, and quoted literals. Add prose cases proving `A stable process.` and `A stable process` still collide, and `Version 2.0.` is treated as prose despite its decimal point.

Run: `python -m pytest tests/test_flashcard_validator.py -k "notation or formatting_variants" -v`

Expected: FAIL because terminal comma and semicolon are currently removed unconditionally.

- [ ] **Step 2: Add failing exact collision-message tests**

Build a multiple-choice card where `correct_option`, `wrong_option_1`, and `wrong_option_3` contain the same logical value with harmless formatting differences. Assert that validation returns all three pairs and includes exact field names plus `json.dumps(original_value, ensure_ascii=False)` values.

Run: `python -m pytest tests/test_flashcard_validator.py -k "duplicate_option" -v`

Expected: FAIL with the generic `options must be distinct` message.

- [ ] **Step 3: Implement content-based notation detection and pairwise collision reporting**

Use syntax signals rather than course vocabulary: multi-character operators, assignment/comparison operators, arithmetic/logical symbols, brackets, quoted literals, identifier-like expressions containing syntax, and number-unit expressions. Keep final `.`, `?`, and `!` harmless; retain terminal `,` and `;` only for notation-like values. Do not classify a decimal point by itself as notation.

In `validate_cluster`, pass named option fields in stable order and append an error shaped like:

```text
card 1 multiple-choice duplicate options: correct_option "..." and wrong_option_2 "..."
```

Emit every pair, not only the first collision.

- [ ] **Step 4: Run focused validator tests**

Run: `python -m pytest tests/test_flashcard_validator.py -v`

Expected: PASS, including existing answer-leak and provenance tests.

- [ ] **Step 5: Commit the option-validation change**

```powershell
git add flashcard_validator.py tests/test_flashcard_validator.py
git commit -m "fix: preserve symbolic option distinctions"
```

---

### Task 2: Make Question Duplication Card-Type Aware

**Files:**
- Modify: `flashcard_validator.py`
- Modify: `flashcard_pipeline.py`
- Modify: `tests/test_module_validation.py`
- Modify: `tests/test_flashcard_pipeline.py`

**Interfaces:**
- Add `are_card_questions_duplicates(left: FlashcardDraft, right: FlashcardDraft) -> bool`.
- Add an exact normalized helper for type-unknown prior strings, such as `is_exact_question_duplicate(question: str, prior_question: str) -> bool`.
- Preserve `are_near_duplicates(left: str, right: str) -> bool` for same-type fuzzy comparison and focused low-level tests.

- [ ] **Step 1: Write failing module-level duplicate-policy tests**

Add tests proving:

- a multiple-choice question and a true-false statement with overlapping subject wording are accepted;
- byte-different but normalization-identical question text is rejected across card types;
- same-type one-word near duplicates remain rejected; and
- polarity variants remain distinct under the existing policy.

Run: `python -m pytest tests/test_module_validation.py -k "cross_type or same_type or exact" -v`

Expected: the cross-type fuzzy-overlap case FAILS under the current untyped comparison.

- [ ] **Step 2: Write failing cluster and prior-corpus policy tests**

In `tests/test_flashcard_pipeline.py`, cover `_duplicate_errors(...)` with typed cards and prior question strings. Assert same-type fuzzy overlap is rejected, cross-type fuzzy overlap is accepted, and prior strings reject exact normalized equality only.

Run: `python -m pytest tests/test_flashcard_pipeline.py -k "duplicate_errors or prior_question" -v`

Expected: FAIL because prior strings and typed generated cards currently share the fuzzy path.

- [ ] **Step 3: Implement the shared typed policy**

Implement one policy used by both `validate_module(...)` and `FlashcardPipeline._duplicate_errors(...)`:

```python
def are_card_questions_duplicates(
    left: FlashcardDraft,
    right: FlashcardDraft,
) -> bool:
    if normalize_stem(left.question) == normalize_stem(right.question):
        return True
    if left.type != right.type:
        return False
    return are_near_duplicates(left.question, right.question)
```

Retain the separate polarity check. For historical prior-question strings, use exact normalized equality only because type metadata is unavailable.

- [ ] **Step 4: Run focused duplicate tests**

Run: `python -m pytest tests/test_module_validation.py tests/test_flashcard_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the typed duplicate policy**

```powershell
git add flashcard_validator.py flashcard_pipeline.py tests/test_module_validation.py tests/test_flashcard_pipeline.py
git commit -m "fix: compare duplicate questions by card type"
```

---

### Task 3: Give Truncation an Independent Retry Allowance

**Files:**
- Modify: `flashcard_pipeline.py`
- Modify: `main.py`
- Modify: `pipeline.py`
- Modify if the API namespace requires an explicit value: `api_server.py`
- Modify: `tests/test_flashcard_pipeline.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_pipeline_runner.py`
- Modify only if needed: `tests/test_api_server.py`

**Interfaces:**
- Extend `PipelineConfig` with `max_truncation_retries: int = 2` and positive/non-negative validation appropriate to its meaning.
- Add CLI flag `--max-truncation-retries` with default `2`.
- Forward the flag from `pipeline.py` to the flashcard generation subprocess.
- `_complete_with_retries(...)` uses separate `validation_attempts` and `truncation_failures` counters.

- [ ] **Step 1: Write failing retry-accounting tests**

Use a scripted backend to return:

1. `CompletionTruncatedError`;
2. `CompletionTruncatedError`;
3. parseable invalid JSON content;
4. a corrected valid response.

Configure `max_retries=2` and `max_truncation_retries=2`. Assert the call succeeds after four backend calls, with two validation slots still available after truncations.

Add separate tests that exhaust each budget and assert the final message names `truncation` or `validation` respectively. Also assert a pre-generation context-window configuration error is not retried.

Run: `python -m pytest tests/test_flashcard_pipeline.py -k "truncation or retry_budget" -v`

Expected: FAIL because one shared `for attempt` loop consumes all attempts.

- [ ] **Step 2: Implement bounded independent counters**

Replace the single attempt loop with a bounded loop whose maximum calls are `max_retries + max_truncation_retries`. Increment the truncation counter only for `CompletionTruncatedError`; increment validation attempts only after receiving a candidate that the parser rejects. Log each counter with its own denominator. Preserve immediate propagation for context-window-overflow/configuration errors.

Malformed JSON remains a validation rejection because a non-truncated candidate was received.

- [ ] **Step 3: Write and run failing CLI propagation tests**

Assert:

- `main.parse_args([]).max_truncation_retries == 2`;
- an explicit value reaches `PipelineConfig`;
- `pipeline.build_stage_commands(...)` forwards `--max-truncation-retries` to the flashcard command; and
- batch/API construction continues to use the default unless explicitly configured.

Run: `python -m pytest tests/test_main.py tests/test_pipeline_runner.py tests/test_api_server.py -k "truncation_retries or pipeline_args or stage_commands" -v`

Expected: FAIL before the new option is wired.

- [ ] **Step 4: Thread the configuration through entry points**

Add the argument to `main.py` and the wrapper in `pipeline.py`. Do not reinterpret the existing batch `attempts` setting as a truncation allowance. Where callers create an `argparse.Namespace` directly, use the default or `getattr(..., 2)` for backward compatibility.

- [ ] **Step 5: Run focused retry and entry-point tests**

Run: `python -m pytest tests/test_flashcard_pipeline.py tests/test_main.py tests/test_pipeline_runner.py tests/test_api_server.py -q`

Expected: all tests affected by retry configuration PASS. Record any already-known unrelated API collision failures separately; do not weaken those tests.

- [ ] **Step 6: Commit independent retry accounting**

```powershell
git add flashcard_pipeline.py main.py pipeline.py tests/test_flashcard_pipeline.py tests/test_main.py tests/test_pipeline_runner.py
git add api_server.py tests/test_api_server.py  # only if changed
git commit -m "fix: separate truncation and validation retries"
```

---

### Task 4: Switch Repeated Invalid Clusters to Targeted Card Repair

**Files:**
- Modify: `flashcard_pipeline.py`
- Modify: `flashcard_prompt.py`
- Modify: `tests/test_flashcard_pipeline.py`
- Modify: `tests/test_flashcard_prompt.py`

**Interfaces:**
- Add a private rejected-candidate fingerprint helper based on stripped exact response text or SHA-256 of that text.
- Add a private `_RepairableClusterValidationError(ValidationError)` carrying the already parsed five cards. Raise it only after `parse_cards(...)` succeeds and later cluster/duplicate validation fails.
- Extend `_complete_with_retries(...)` with an optional typed callback, for example:

```python
def _complete_with_retries(
    ...,
    repeated_candidate_handler: (
        Callable[[str, Sequence[str]], Parsed] | None
    ) = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Parsed:
```

- Add `_validation_repair_targets(errors: Sequence[str]) -> tuple[int, ...]` or an equivalent structured result that maps every error to a zero-based card position.
- Add `_repair_validation_cards(...) -> tuple[FlashcardDraft, ...]` and reuse the existing `build_single_card_schema(...)` path.
- Add `build_validation_card_repair_prompt(...)` and a focused retry builder.

- [ ] **Step 1: Write failing repeated-response routing tests**

Script two identical parseable invalid five-card responses, then a valid single-card repair response. Assert:

- the second identical invalid response triggers repair;
- the backend receives two cluster-schema calls and one single-card-schema call;
- no third full-cluster generation occurs; and
- the repaired card returns to its original array position.

Add negative cases for repeated truncated responses and repeated malformed JSON. They must use ordinary retry accounting and never invoke the repair callback.

Run: `python -m pytest tests/test_flashcard_pipeline.py -k "repeated_candidate or targeted_repair" -v`

Expected: FAIL because the completion loop does not fingerprint rejected candidates.

- [ ] **Step 2: Write failing repair-target extraction tests**

Cover these generic error forms:

- `card 3 ...` maps to card index 2;
- `cards 2 and 5 are near duplicates` maps only to later card index 4;
- multiple option-collision errors on card 1 coalesce into one target;
- multiple affected cards are returned in deterministic order; and
- an unknown cluster-wide error returns no target instead of guessing.

Run: `python -m pytest tests/test_flashcard_pipeline.py -k "repair_targets" -v`

Expected: FAIL because no target extractor exists.

- [ ] **Step 3: Add prompt tests for precise targeted context**

Assert the validation-repair prompt contains the exact original card JSON, only that card's relevant errors, conflicting question text where applicable, exact duplicate option fields/values, concept evidence, and an instruction to preserve unrelated fields. Assert it contains no course-specific sample text.

Run: `python -m pytest tests/test_flashcard_prompt.py -k "validation_card_repair" -v`

Expected: FAIL because the prompt builder does not exist.

- [ ] **Step 4: Implement candidate fingerprints and safe callback dispatch**

Fingerprint every non-truncated rejected response for repetition diagnostics, but dispatch the card-repair callback only when the latest exception is `_RepairableClusterValidationError`. `parse_and_validate(...)` must raise that subtype only after it has parsed exactly five cards and collected card-addressable validation errors. When the same fingerprint produces that subtype again, call the handler immediately with the repeated raw candidate and latest structured errors. Do not dispatch for `CompletionTruncatedError`, backend exceptions, empty output, malformed JSON, a wrong card count, or validation errors whose card positions cannot be recovered safely.

If no handler is supplied, retain ordinary retry behavior for non-card stages.

- [ ] **Step 5: Implement the deterministic card repair queue**

Parse the repeated five-card candidate, map errors to targets, and stop with a precise error when none can be mapped safely. For each target:

1. build one single-card prompt;
2. preserve `type`, `is_true`, `difficulty`, and `assessment_approach` through `build_single_card_schema` and parser checks;
3. preserve card fields unrelated to the reported error;
4. replace the card at the exact original index; and
5. continue until the queue is empty.

After the queue finishes, call `validate_cluster(...)` and `_duplicate_errors(...)` on the full reconstructed cluster. Add a regression where a repaired question conflicts with a different card; full-cluster validation must reject it rather than returning the partial repair.

Reuse or extract the mechanics of `_repair_duplicate_stack(...)`; do not create a second inconsistent single-card validation policy.

- [ ] **Step 6: Run focused prompt and pipeline tests**

Run: `python -m pytest tests/test_flashcard_prompt.py tests/test_flashcard_pipeline.py -q`

Expected: PASS, including existing final duplicate-stack behavior.

- [ ] **Step 7: Commit repeated-candidate targeted repair**

```powershell
git add flashcard_pipeline.py flashcard_prompt.py tests/test_flashcard_pipeline.py tests/test_flashcard_prompt.py
git commit -m "fix: repair repeated invalid cards directly"
```

---

### Task 5: Compact 20-Concept Planning Without Dropping Evidence

**Files:**
- Modify: `flashcard_contract.py`
- Modify: `flashcard_schema.py`
- Modify: `flashcard_prompt.py`
- Modify: `flashcard_pipeline.py`
- Modify: `tests/test_flashcard_prompt.py`
- Modify: `tests/test_flashcard_pipeline.py`
- Create or modify: `tests/test_flashcard_schema.py`

**Interfaces:**
- Add `MAX_FACT_IDS_PER_CONCEPT = 8` to `flashcard_contract.py`.
- `build_concept_plan_schema(...)` sets `maxItems` from that shared constant while retaining `minItems: 1` and allowing fact sharing across concepts.
- Add a concise `CONCEPT_PLAN_SYSTEM_PROMPT` containing only source authority, JSON-only output, exact concept count, and non-invention rules.
- `_complete_with_retries(...)` accepts a per-stage `system_prompt`, defaulting to the existing `SYSTEM_PROMPT`.
- `build_concept_plan_prompt(...)` and its truncation retry serialize planner facts as only `{fact_id, statement}`.

- [ ] **Step 1: Write failing schema-bound tests**

Assert the concept schema has `maxItems == MAX_FACT_IDS_PER_CONCEPT`, accepts a 20-concept plan with up to eight IDs per concept, permits the same fact in more than one concept, and rejects nine IDs on one concept.

Run: `python -m pytest tests/test_flashcard_schema.py tests/test_flashcard_pipeline.py -k "fact_ids or concept_plan_schema" -v`

Expected: FAIL because `fact_ids` has no maximum.

- [ ] **Step 2: Write failing compact-prompt tests**

Create the maximum selected fact set with long Unicode statements. Assert:

- every exact `fact_id` and statement occurs in the prompt;
- `topic` and `slides` keys do not occur;
- the literal numbered `1`-through-`20` concept checklist is absent;
- assessment approaches appear in one compact rule block;
- the truncation retry does not embed rejected bulk output; and
- both initial and retry prompts remain below explicit character bounds derived from the current verbose prompt fixture.

The bound is a regression threshold, not a claimed token count.

Run: `python -m pytest tests/test_flashcard_prompt.py -k "compact or concept_plan" -v`

Expected: FAIL because the planner prompt repeats instructions and optional metadata.

- [ ] **Step 3: Implement the shared fact-ID cap and concise prompts**

Use the shared constant in both schema and prompt text. Keep all input facts but tell the model to choose the smallest sufficient set, up to eight IDs, for each concept. Do not restore the removed exclusive-anchor rule: fact sharing is allowed when useful.

Add a dedicated compact planning system prompt. Pass it only for concept planning; card generation and reviews retain their existing system prompt.

Use an especially short retry after truncation: restate the JSON contract and evidence once, omit rejected output, and avoid optional topic/slide metadata.

- [ ] **Step 4: Add an end-to-end planning retry test**

Script two 8,192-token-style truncations followed by one complete 20-concept response whose concepts contain no more than eight fact IDs. Assert planning succeeds, all facts were present in the prompts, and the call used the concise planning system prompt each time.

Run: `python -m pytest tests/test_flashcard_pipeline.py -k "compact_plan or planning_truncation" -v`

Expected: PASS after Tasks 3 and 5 are integrated.

- [ ] **Step 5: Run focused schema, prompt, and planning tests**

Run: `python -m pytest tests/test_flashcard_schema.py tests/test_flashcard_prompt.py tests/test_flashcard_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit compact planning**

```powershell
git add flashcard_contract.py flashcard_schema.py flashcard_prompt.py flashcard_pipeline.py tests/test_flashcard_schema.py tests/test_flashcard_prompt.py tests/test_flashcard_pipeline.py
git commit -m "fix: reserve context for complete concept plans"
```

---

### Task 6: Document and Verify the Integrated Behavior

**Files:**
- Modify: `README.md`
- Test: `tests/test_flashcard_validator.py`
- Test: `tests/test_module_validation.py`
- Test: `tests/test_flashcard_prompt.py`
- Test: `tests/test_flashcard_schema.py`
- Test: `tests/test_flashcard_pipeline.py`
- Test: `tests/test_main.py`
- Test: `tests/test_pipeline_runner.py`
- Test: `tests/test_batch_pipeline.py`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Documents `--max-retries` as parseable validation attempts.
- Documents `--max-truncation-retries` as an independent completion-length allowance with default `2`.
- States that the default context remains 8,192 tokens and output CSV behavior is unchanged.

- [ ] **Step 1: Update user-facing retry documentation**

Keep the explanation course-neutral and short. Include one example showing the two flags together. Do not promise that every model response can be repaired; explain that repeated valid five-card JSON enters targeted card repair only when the validator can identify card positions safely.

- [ ] **Step 2: Run the complete focused resilience suite**

Run:

```powershell
python -m pytest tests/test_flashcard_validator.py tests/test_module_validation.py tests/test_flashcard_schema.py tests/test_flashcard_prompt.py tests/test_flashcard_pipeline.py tests/test_main.py tests/test_pipeline_runner.py tests/test_batch_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full non-API suite**

Run: `python -m pytest --ignore=tests/test_api_server.py -q`

Expected: PASS.

- [ ] **Step 4: Run the full suite and classify any failures**

Run: `python -m pytest -q`

Expected: all changes introduced by this plan PASS. If the six previously observed API upload/module-number collision tests still fail unchanged, capture their names and failure signatures as pre-existing; do not conceal them or change their assertions as part of this work.

- [ ] **Step 5: Inspect the final diff for accidental tuning or output changes**

Run: `git diff HEAD~5 -- flashcard_validator.py flashcard_pipeline.py flashcard_prompt.py flashcard_schema.py flashcard_contract.py main.py pipeline.py README.md tests`

Verify:

- no subject-specific terms appear in new production rules or prompts;
- CSV schema and output paths are untouched;
- no exclusive fact-ownership rule was reintroduced;
- exact duplicate option values are present in feedback;
- cross-type fuzzy comparisons are absent;
- truncation and validation counters are separate; and
- planner facts retain full statements.

- [ ] **Step 6: Commit documentation or final integration adjustments**

```powershell
git add README.md
git add <only-files-changed-by-final-integration-fixes>
git commit -m "docs: explain resilient flashcard retries"
```

- [ ] **Step 7: Request final code review before integration**

Use `superpowers:requesting-code-review` against the implementation range. Address only verified findings, rerun the relevant focused tests, then use `superpowers:verification-before-completion` before claiming success.
