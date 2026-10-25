# Module to Flashcards

Generate validated assessment CSV files from course folders containing module
PDFs using local Tesseract OCR and the Mistral hosted API. Mistral handles relationship extraction, graph review,
concept planning, flashcard generation, and final reviews. No local model,
CUDA, or GPU is required. Report content is sent to Mistral over HTTPS.

Current release: 1.1.0

## Course folders and PDF input

Choose a course folder, or a parent folder containing several course folders.
Each PDF is one module:

```text
Courses/
  CS0003/
    Module 1.pdf
    Module 2 - Algorithms.pdf
  EE101/
    M1.pdf
    02 - Circuits.pdf
```

The PDF's immediate parent folder supplies the course code. The filename
supplies the module number (`Module 1.pdf`, `M1.pdf`, `01 - Title.pdf`, and
`CS0003-M1.pdf` are accepted). Individual PDFs can use the entered course code
when their filename does not include one. Ambiguous module names produce a
visible error rather than a guessed identity. Original PDFs are preserved.

Selectable PDF text is read locally with PyMuPDF. Tesseract OCR reads pages
with sparse native text or large embedded images, including a scanned body
under a selectable title. Choose **Every page** to force OCR. The default is
English (`eng`); other language codes, such as `eng+fil`, require their
Tesseract language data to be installed. Page rendering is bounded to limit
memory use, and each OCR invocation has a 120-second timeout.

Mistral classifies extracted text blocks and removes only clearly decorative
or administrative blocks: logos, repeated institutional branding, page
numbers, copyright footers, navigation, and thank-you text. Mixed or uncertain
blocks are preserved. Retained source text is not rewritten, so lesson
statements, formulas, examples, tables, and diagram labels remain available.
Mistral also selects groups of retained block indexes so bullet-heavy slides
can supply lesson facts. The app reconstructs study passages directly from
those source blocks; Mistral does not copy or rewrite them. This keeps cleanup
responses small (typically a few hundred tokens of budget rather than a fixed
4,096-token response). Only whitespace changes are allowed in reconstructed text. OCR reads visible labels; this pipeline
does not infer unseen diagram meaning.
A large block that cannot safely fit a filtering request is preserved.

If Mistral marks a block as both decorative and useful for study, the app keeps
that source block. Study groups may reuse a heading, and repeated indexes are
deduplicated locally. Invalid source indexes trigger a retry with validation
feedback; a response cut short gets a larger output budget, capped at 1,024
tokens. If cleanup still cannot be validated, the affected source text is kept
for review and processing continues. This fallback applies to response
validation; API access and limit errors still stop or retry as described below.
Unfinished, empty, or malformed model answers use the same recovery path.
Their actual finish reason and request ID are recorded when available;
incomplete output is never accepted as a completed answer. Relationship and
flashcard generation also retry unfinished answers with repair feedback.
Explicit provider content-filter refusals remain errors.

Each module workspace contains `pdf_extraction.json`, recording raw text,
page numbers, text/OCR method, block positions, keep/remove decisions, and
source-verified study passages. Pages needing review include `filter_status`
and `filter_warnings`; their warnings also appear in the server log and progress.
`structured_module.txt` contains the filtered lesson content used by the
remaining Mistral stages. A PDF with no readable learning content fails
visibly. Invalid or password-protected PDFs are rejected.

Existing UTF-8 TXT slide reports and version-1 `[MODULE]` / `[SLIDE n]`
reports remain supported through the individual-file input and API. TXT
filename identities of the form `CS0003-M2.txt` retain their earlier priority;
otherwise the course field and report metadata supply the identity.

## Start the app

In Windows PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Install Tesseract OCR with English language data if it is not already installed.
python api_server.py
```

Open http://localhost:8000 in your browser. Choose your course folder, enter
one or more Mistral API keys, and click **Generate flashcards**. Use the
arrow buttons to reorder keys. The default model is `ministral-14b-2512`,
hosted on Mistral's servers;
you can enter another Mistral model ID that supports structured JSON output.
Download the two CSV files from each completed module. The UI checks whether
Tesseract is available. Before uploading modules, it makes one small structured
JSON request to check that your key can actually use the selected model.
Subscription or model-limit errors appear before the PDFs are uploaded.
The app finds Tesseract on PATH, in the usual Windows install
location, or beside a portable executable. Set `TESSERACT_CMD` to an executable
path if it is installed elsewhere. OCR installation guidance is available in
the [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).

Keys are kept in server memory only. They are never saved in generated files,
returned by status endpoints, or stored in browser storage. Close a completed
run to clear its keys. Restarting the server clears all keys and in-memory
run status. Idle runs expire after 12 hours; up to 64 runs can be retained.
Run files remain on disk after closing a run. Keep this local server bound to
127.0.0.1, as in the supplied launch commands.

Refreshing or reopening the browser page reconnects to submitted runs on the
same running server, including completed downloads. It does not upload files
or repeat ingestion. Status polling continues through a dropped browser
connection and reconnects when the page becomes visible after sleep. Run
references are retrieved from the server; no API keys are saved in browser
storage. A server restart still clears the in-memory run list and keys.

To continue an interrupted run, choose **Saved graphs or structured modules** in the
**Start from** menu. Select the saved run, enter your Mistral keys, and click
**Continue to flashcards**. The app discovers graphs and `structured_module.txt`
files under `runs/` even after a server restart. It copies the checkpoint into
a new isolated run, keeps
completed CSV pairs, and generates only the remaining modules. Each module
uses its furthest usable checkpoint. Structured text skips PDF extraction,
OCR, and filtering, and starts at graph extraction. Checked graphs are reused;
unchecked graphs are reviewed before generating cards. Original PDFs are not
required, and modules with graphs do not need structured text. The original
run stays on disk. Unusable saved modules are reported individually while
other modules continue.

## Key queue and limits

Each run uses one active key at a time. Other keys remain on standby even
when several generation workers are configured. A complete request owns the
active key through its waits and retries; queued workers keep their own data
and wait for their turn.

Temporary HTTP 429 responses wait on the same key, respecting `Retry-After`
and sharing the pause with other runs in this server process. Generic rate-limit
responses never disable a key or trigger a sweep through backups. Invalid keys
(HTTP 401), payment failures (HTTP 402), and explicit exhausted monthly/daily
quota errors advance to the next key in order. Model permission errors stop
with the provider's explanation. Connection and service errors retry the active
key with backoff; certificate errors stop immediately rather than trying keys.

Request timing follows the PDF-to-SRT web automation:
requests are spaced by at least five seconds, with a five-second pause after
each response. Temporary HTTP 429 responses wait 90 seconds when Mistral does
not supply `Retry-After`. This pacing and shared backoff apply across runs in
one server process; separate server processes do not coordinate pacing.
Each completion allows up to eight temporary attempts within a 30-minute retry
window, enough for the waits plus individual HTTP attempts of up to 120 seconds.
Waiting for another generation worker's turn does not spend that retry
window. If a wait is longer than the retry window, or temporary attempts still
fail, the module receives a specific error; standby keys remain untouched.

A zero `x-ratelimit-limit-req-minute` allowance on an HTTP 429 is a model
configuration error rather than a temporary cooldown. The app stops without
retrying or rotating keys. A subscription denial (HTTP 403), zero model
allowance, unavailable key queue, or certificate failure stops the remaining
modules in that run instead of repeating the same request for each PDF.
Previously completed modules are preserved. Choose a model available to your
subscription; using the same model ID as another tool does not grant access.

The UI distinguishes active, waiting, standby, invalid, and exhausted keys.
**Mistral request details** shows the recent provider status, explanation, and
request ID for troubleshooting. API key values are redacted from diagnostics,
logs, status, and errors. There is no generic claim that keys are exhausted
merely because temporary retries failed. Keys must be re-entered in a new run
if an invalid or genuinely exhausted key needs to be replaced.

Rotation does not increase the provider's shared quotas. Mistral can enforce
limits across an organization or workspace, so several keys may reach a
shared limit together. See [Mistral usage limits](https://docs.mistral.ai/admin/billing-usage/usage-limits).

## Concurrent runs and data isolation

Start another run or open another browser tab while processing continues.
Each run has an independent key pool, model selection, input files, prompts,
course corpus, intermediate artifacts, output files, and status. Every chat
completion contains only that worker's system and user messages; no provider
conversation or shared message history is used. Workers share the credential
scheduler within their run, not loaded report data. The default is one
generation worker; choose 1–20 in the UI. Requests in one run are serialized
on its active key, so additional workers wait rather than using standby keys.
Modules within one batch run sequentially. Separate runs can transact with
Mistral concurrently, with process-wide request pacing and rate-limit waits.

Files are stored under `runs/<random-run-id>/uploads/`,
`runs/<random-run-id>/pipeline_output/`, and
`runs/<random-run-id>/flashcards/`. Even simultaneous runs with the same
course and module names have separate output locations. Several server
processes can serve UI runs on distinct ports; use one Uvicorn worker per
server process because keys and run status are process-local. Separate
server processes do not coordinate shared API quotas.

The older extension endpoint `POST /process` still accepts `courseCode` and
one or more `files` multipart fields. PDF uploads can carry relative filenames
such as `Courses/CS0003/Module 1.pdf`; folder-derived identities work without
a course field. Without a `session_id`, it keeps the
original serialized batch behavior, shared course corpus, and canonical
`pipeline_uploads/`, `pipeline_output/`, and `flashcards/` locations. Configure
its keys with `MISTRAL_API_KEY` or `MISTRAL_API_KEYS` before starting the
server. The latter accepts newline-separated or comma-separated keys.
Restart the server to reload environment keys. Use the UI session workflow
for concurrent isolated runs. Avoid running several legacy endpoint or CLI
processes against the same output directories.

Session API clients can create a run with `POST /api/sessions` and a JSON
body containing `apiKeys`, `model`, and `workers`. Call
`POST /api/sessions/<sessionId>/check` to verify model access with a small
structured JSON request before uploading files. On success, upload to
`POST /process?session_id=<sessionId>`. Poll `GET /api/sessions/<sessionId>`
for run-specific progress and key availability; `DELETE` that same URL
clears an idle run and its credentials. The opaque session ID controls
access to its status and CSV downloads. `GET /status` retains aggregate
progress for older clients, and `GET /health` reports server readiness.

For recovery, `GET /api/saved-runs` lists saved run IDs, course codes,
graph/structured checkpoint counts (`graphCount` and `structuredCount`),
and completed/pending module counts. Create a new session, check model access,
then call `POST /api/sessions/<sessionId>/resume` with
`{"sourceRunId":"<saved-run-id>"}`. Progress and downloads use the same session
endpoints. Runs still processing cannot be used as recovery sources.

## Command-line pipeline

Set keys in the current PowerShell session, then run:

```powershell
$env:MISTRAL_API_KEY = "your-api-key"
# A course folder or a parent folder containing courses:
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\Courses" `
  --mistral-model ministral-14b-2512 --cluster-workers 1

# One module PDF, with optional explicit identity:
.\.venv\Scripts\python.exe pipeline.py "C:\path\to\EE101\Module 9.pdf" `
  --course-code EE101 --module-number 9 --ocr-mode always
```

`MISTRAL_MODEL` sets the model default when `--mistral-model` is omitted.
Other useful options are `--output-root`, `--ocr-language`, `--ocr-dpi`,
`--n-ctx`, `--timeout`, and `--force`. TXT CLI reports still require
`--course-code` and `--module-number`. Folder runs report per-PDF errors while
allowing valid modules to continue.
The default context budget is 32,768 tokens, using a conservative UTF-8 byte
bound for prompt sizing rather than an exact provider tokenizer. Lower it
for a model with a smaller window. The module timeout defaults to disabled;
individual HTTP attempts retain a 120-second timeout, with a separate
30-minute window for temporary retry waits. Provider and model
settings are recorded in reuse manifests, never credentials.

Run only graph extraction with `text-extractor.py`, or generate from an
existing graph with `main.py`. Both now use Mistral. Legacy GPU/REBEL flags
remain accepted for older callers but do not select local inference.

Every successful module produces two CSV files with 50 rows each: 20 concept
clusters, five questions per cluster, and exactly 100 questions. Existing
schema checks, source-grounding checks, deduplication, full-module validation,
and paired-output receipts remain in place. The UI enables final model review.

## Portable Windows release

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File packaging/build_portable.ps1
```

The build creates a Windows x64 client with the Python runtime, browser UI,
PyMuPDF, and Tesseract including installed language data. Install Tesseract on
the build machine, or pass `-TesseractDir` pointing to its installation folder.
It does not download or bundle local language models or CUDA libraries. Keep the full
extracted directory together. Run `ModuleToFlashcards.exe`, open the reported
localhost URL, and enter keys in the UI. Internet access is required for
inference. `--port 8010` selects another port; `--verify` checks bundle files
without contacting Mistral. Historical local-model asset preparation utilities
and benchmarks remain available for legacy experiments, not the default app.

## Development checks

```powershell
# Additional dependency used by historical local-model unit tests:
python -m pip install huggingface-hub
.\.venv\Scripts\python.exe -m pytest -q
```

`requirements-local.txt` contains optional dependencies for running legacy
local-model experiments. Generated models, uploads, run files, outputs,
portable assets, temporary data, and logs stay outside version control.
