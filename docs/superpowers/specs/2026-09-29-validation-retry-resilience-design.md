# Validation and Retry Resilience Design

> Later correction: the eight-fact-per-concept cap described below was removed after user feedback. The current concept schema and parser permit any number of known fact IDs; prompt compaction remains in place.

## Purpose

Make flashcard generation recover reliably from validation failures across
programming, mathematics, engineering, multimedia arts, and prose-heavy
courses. The solution must preserve meaning-bearing notation, identify the
exact rejected content, avoid false cross-type duplicate flags, stop wasting
calls on byte-identical responses, keep truncation from consuming validation
attempts, and fit a complete 20-concept plan inside the existing 8,192-token
low-memory context window.

## Scope

This change covers:

- option-equivalence validation;
- validation error details for colliding options;
- question-duplicate comparison within generated modules;
- repeated-candidate detection and targeted card repair;
- independent truncation and validation retry accounting;
- concept-plan prompt and output compaction; and
- regression coverage for the complete flow.

It does not change the required 20 concepts, five cards per concept, allowed
card types, grounding rules, final CSV schema, knowledge-graph checker, or
default Qwen model. It also does not increase the default context window,
because doing so would raise memory requirements on low-resource machines.

## Existing Failure Modes

### Symbolic options collapse during comparison

`_canonical_option` removes terminal commas and semicolons as if they were
always sentence formatting. In code, however, `cout << value;` and
`cout << value,` are different expressions. The same risk exists for
equations, ratios, units, operators, and other notation-heavy answers.

The current error only says that options must be distinct. It does not tell
Qwen which fields collided or show the values involved.

### Cross-type questions are compared as if they had the same form

The near-duplicate check removes interrogative framing and permits a one-word
edit. A multiple-choice question and a true-false proposition about the same
subject can therefore be rejected even when they are different assessment
acts.

### Whole-cluster retries can stall

When Qwen returns the same invalid five-card JSON repeatedly, the pipeline
keeps asking for another full cluster. Different seeds do not guarantee a
different response. The retry budget is exhausted without changing the
flagged card.

### Truncation consumes semantic correction attempts

The current retry loop counts every model call against one shared maximum.
Two context-limit truncations can leave only one parseable response. If that
response has a repairable validation error, generation fails immediately.

### The planner leaves too little completion space

The general system prompt, repeated planning instructions, optional fact
metadata, and unrestricted fact-id arrays consume too much of an 8,192-token
context. A 20-concept response then reaches the hard context limit.

## Design

### 1. Notation-aware option equivalence

Option comparison will use two normalization paths:

1. All options receive Unicode normalization, case folding, whitespace
   normalization, and spacing normalization around technical symbols.
2. Prose-like options ignore terminal sentence punctuation.
3. Notation-like options retain terminal commas and semicolons when the value
   contains code, mathematical, logical, structural, or unit syntax.

Notation detection will be content-based, not course-based. Signals include
multi-character operators, assignment/comparison operators, arithmetic and
logical symbols, brackets, quoted literals, identifier-like expressions, and
number-unit expressions. No programming-language names or module-specific
terms will be used.

Periods, question marks, and exclamation marks at the end remain sentence
punctuation. Decimal points and punctuation inside an expression remain
meaningful.

The multiple-choice validator will compare named fields rather than an
anonymous tuple. For every collision it will report both field names and
JSON-quoted original values. Example:

```text
card 1 multiple-choice duplicate options: correct_option "cout << value;"
and wrong_option_2 "cout << value;"
```

If more than one pair collides, each pair receives its own error. This gives
the retry prompt enough information to change the right field.

### 2. Card-type-aware question duplication

Question comparison will distinguish exact duplicates from fuzzy overlap:

- Exact normalized question text is a duplicate regardless of card type.
- Fuzzy near-duplicate comparison applies only when both cards have the same
  type.
- Questions with different negation or polarity markers remain distinct.
- Cross-type cards that merely share subject wording are accepted.

Within a duplicate pair, the later card is the repair target and the earlier
card is the stable reference. Module-level and cluster-level checks will use
the same policy.

Historical course-corpus entries currently store only question strings. For
those legacy entries, only exact normalized duplication will be enforced,
because their original card type is unavailable. The on-disk corpus format
will remain backward compatible.

### 3. Repeated-response detection and targeted repair

`_complete_with_retries` will fingerprint every rejected candidate using its
exact normalized JSON text. If a later validation attempt returns the same
candidate, the pipeline will stop full-cluster regeneration immediately.

For card-cluster generation, a repeated-candidate callback will:

1. parse the rejected five-card candidate;
2. derive repair targets from validation errors;
3. choose the later card for pairwise duplicate errors;
4. group each target card's relevant errors;
5. push the targets into a deterministic repair queue;
6. repair one card at a time with the existing single-card JSON schema;
7. restore each repaired card to its original array position; and
8. validate the reconstructed cluster after all queued repairs.

The targeted repair prompt will include:

- the exact rejected card;
- only the errors relevant to that card;
- conflicting question text for duplicate errors;
- duplicated option field names and values;
- the concept evidence; and
- explicit instructions to preserve fields unrelated to the error.

The single-card schema will continue to lock type, truth value, difficulty,
and assessment approach. Identification and true-false option constraints
remain enforced. If targeted repair exhausts its own bounded attempts, the
pipeline returns a precise targeted-repair failure instead of silently
falling back to another identical cluster generation.

Non-card generators may detect repeated candidates for diagnostics but will
continue using their stage-specific retry prompts; targeted card repair is
only valid for five-card cluster output.

### 4. Independent retry accounting

`PipelineConfig.max_retries` remains the number of parseable validation
attempts, preserving the public configuration name. A new
`max_truncation_retries` setting defaults to `2`.

The retry loop will maintain separate counters:

- truncation failures increment only the truncation counter;
- malformed or validation-rejected responses increment the validation
  counter; and
- successful output ends both loops.

The maximum model calls for one stage therefore become the configured
validation attempts plus the configured truncation allowance. Progress logs
will name the relevant counter so users can distinguish a context failure
from a content-validation failure.

Context-window-exceeded errors that occur before generation remain immediate
configuration errors; they are not completion truncations and will not be
retried blindly.

### 5. Compact concept planning within 8,192 tokens

Planning will use a dedicated concise system prompt rather than the full
card-writing system prompt. It will retain source authority, JSON-only output,
and the requirement not to invent facts, but omit card-specific rules.

The initial and retry planning prompts will:

- include every selected fact ID and statement;
- omit topic and slide metadata that the planner does not need;
- state each structural rule once;
- replace the verbose numbered 1-to-20 checklist with an exact array-length
  requirement;
- include the allowed assessment approaches once;
- omit rejected bulk output after truncation; and
- use an especially compact prompt after a truncation.

Concepts will select the smallest sufficient fact set. The JSON schema will
cap `fact_ids` at eight per concept. This is not a uniqueness or ownership
rule: facts may still be shared where genuinely useful. The cap prevents the
planner from copying most of the graph into every concept and materially
reduces completion length without discarding source facts from the input.

The existing `insufficient_content` response remains available when the
module cannot support 20 distinct concepts.

### 6. Prompt-size and completion-reserve checks

Tests will measure the actual prompt text generated from the maximum planning
fact set. The compact prompt must remain materially below the previous size
and leave enough room for a complete schema-constrained 20-concept response
under the 8,192-token configuration.

Because the production tokenizer is owned by llama.cpp, unit tests will use
deterministic prompt-length and response-shape bounds rather than pretending a
character count is an exact token count. Integration logs will continue to
record real prompt and completion token usage when Qwen reports it.

## Data Flow

```text
Qwen candidate
    |
    v
parse + validate
    |
    +-- completion truncated --> compact prompt --> truncation counter
    |
    +-- valid --> return result
    |
    +-- invalid, new candidate --> normal retry --> validation counter
    |
    +-- invalid, identical candidate
            |
            v
       derive repair targets
            |
            v
       single-card repair queue
            |
            v
       reconstruct + revalidate cluster
```

## Error Handling

- Duplicate-option errors always identify exact fields and original values.
- Repeated candidates are logged with a message explaining that targeted
  repair is replacing full regeneration.
- A repair target that cannot be mapped safely causes a clear validation
  failure; the pipeline will not guess a card position.
- Truncation exhaustion reports its independent limit and the latest token
  usage.
- Validation exhaustion reports the latest validation errors, not an earlier
  truncation.
- Planner compaction never truncates individual fact statements silently.

## Testing Strategy

Tests will be written before production changes and will cover:

1. C++ statements that differ only by terminal comma versus semicolon.
2. Equations, logical operators, ratios, percentages, units, and quoted
   literals.
3. Prose options that differ only by harmless sentence punctuation.
4. Duplicate-option messages containing both fields and values.
5. Cross-type subject overlap accepted while exact cross-type text remains
   rejected.
6. Same-type one-word near duplicates still rejected.
7. Two byte-identical invalid cluster responses switching to single-card
   repair without a third full-cluster call.
8. Correct queue placement and reconstruction when several cards are flagged.
9. Two truncations followed by a validation rejection and then a corrected
   response succeeding within independent budgets.
10. Truncation-budget exhaustion and validation-budget exhaustion reporting
    the correct final cause.
11. A complete 20-concept plan using compact prompts and bounded fact-id
    arrays.
12. Existing provenance, grounding, duplicate-review, CSV, and batch behavior
    remaining unchanged.

The full project suite will be run after the focused validator, prompt,
pipeline, schema, and backend tests. Any unrelated pre-existing failures will
be reported separately rather than concealed.

## Compatibility and Migration

- `max_retries` retains its name and validation meaning.
- `max_truncation_retries` is optional and defaults to two.
- Existing course-corpus JSON files containing string questions remain
  readable.
- Existing error consumers that match the leading validation category will
  continue to work; regex consumers will be updated to accept the appended
  field/value detail.
- The output CSV format and locations do not change.
- The default Qwen context remains 8,192 tokens.

## Success Criteria

The change is complete when:

- symbolic options that differ meaningfully are accepted;
- actual duplicate option pairs are rejected with exact field/value context;
- fuzzy cross-type overlap is accepted;
- identical invalid clusters enter targeted repair after the first repeated
  response;
- truncations do not reduce the available validation attempts;
- the 20-concept planner has adequate output space at the default context;
- focused regression tests and all otherwise-passing project tests pass; and
- no production rule contains course- or module-specific vocabulary.
