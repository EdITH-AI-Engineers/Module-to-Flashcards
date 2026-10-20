import io
import json
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError

import pytest

import mistral_backend as mb
from flashcard_types import CompletionTruncatedError, ContextWindowExceededError, IncompleteCompletionError


def response(content='{"ok":true}', finish="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}]}).encode())


def http_error(request, status, message, *, code="", retry=None, request_limit=None):
    headers = {"x-request-id": "request-123"}
    if retry is not None:
        headers["Retry-After"] = str(retry)
    if request_limit is not None:
        headers["x-ratelimit-limit-req-minute"] = str(request_limit)
        headers["x-ratelimit-remaining-req-minute"] = "0"
    return HTTPError(request.full_url, status, "provider error", headers,
                     io.BytesIO(json.dumps({"message": message, "code": code}).encode()))


@pytest.fixture(autouse=True)
def isolated_pacer(monkeypatch):
    monkeypatch.setattr(mb, "_REQUEST_PACER", mb.RequestPacer(interval=0))


@pytest.fixture
def fast_clock(monkeypatch):
    class Clock:
        value = 1000.0
        def now(self):
            return self.value
        def advance(self, seconds):
            self.value += seconds
    clock = Clock()
    monkeypatch.setattr(mb.time, "monotonic", clock.now)
    pacer = mb.RequestPacer()
    monkeypatch.setattr(pacer._condition, "wait", clock.advance)
    monkeypatch.setattr(mb, "_REQUEST_PACER", pacer)
    def pool(keys):
        result = mb.ApiKeyPool(keys)
        monkeypatch.setattr(result._condition, "wait", clock.advance)
        return result
    return clock, pool


@pytest.mark.parametrize("recovers", [True, False])
def test_default_model_policy_allows_eight_slow_attempts_without_using_backups(monkeypatch, fast_clock, recovers):
    clock, make_pool = fast_clock
    pool = make_pool(["primary", "backup-2", "backup-3", "backup-4", "backup-5"])
    calls = []
    def send(request, **kwargs):
        body = json.loads(request.data)
        assert body["model"] == "ministral-14b-2512"
        assert body["temperature"] == 0
        assert "reasoning_effort" not in body
        assert kwargs["timeout"] == 120
        calls.append((request.headers["Authorization"], clock.now()))
        clock.advance(120)  # A slow response must not consume the later attempts.
        if recovers and len(calls) == 8:
            return response()
        raise http_error(request, 429, "Rate limit exceeded")
    monkeypatch.setattr(mb, "urlopen", send)
    backend = mb.MistralBackend(pool)
    if recovers:
        assert json.loads(backend.complete("system", "module A", max_tokens=32)) == {"ok": True}
    else:
        with pytest.raises(mb.MistralRequestError, match="after 8 attempts") as error:
            backend.complete("system", "module A", max_tokens=32)
        assert error.value.status == 429
    assert calls == [("Bearer primary", 1000 + attempt * 210) for attempt in range(8)]
    assert [key["state"] for key in pool.snapshot()][1:] == ["standby"] * 4


def test_five_second_pause_follows_slow_responses_across_runs(monkeypatch, fast_clock):
    clock, make_pool = fast_clock
    starts = []
    def send(*args, **kwargs):
        starts.append(clock.now())
        clock.advance(12)
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    for key in ["one", "two"]:
        mb.MistralBackend(make_pool([key])).complete("s", "u", max_tokens=32)
    assert starts == [1000, 1017]


@pytest.mark.parametrize("model", ["mistral-small-latest", "mistral-small-2603", "mistral-small-4-custom"])
def test_small_model_override_still_disables_reasoning(monkeypatch, model):
    calls = []
    monkeypatch.setattr(mb, "urlopen", lambda request, **kwargs: calls.append(json.loads(request.data)) or response())
    mb.MistralBackend(mb.ApiKeyPool(["key"]), model=model).complete("s", "u", max_tokens=32)
    assert calls[0]["model"] == model
    assert calls[0]["reasoning_effort"] == "none"


@pytest.mark.parametrize("request_limit", ["0", "0.0"])
def test_zero_model_allowance_is_not_retried_or_treated_as_exhausted_key(monkeypatch, request_limit):
    pool = mb.ApiKeyPool(["primary", "backup"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        raise http_error(request, 429, "Rate limit exceeded", code="1300", request_limit=request_limit)
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError, match="0 per minute") as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert error.value.category == "model_unavailable" and error.value.status == 429
    assert calls == ["Bearer primary"]
    assert [key["state"] for key in pool.snapshot()] == ["active", "standby"]


def test_used_up_minute_window_with_nonzero_allowance_is_still_temporary(monkeypatch, fast_clock):
    _, make_pool = fast_clock
    pool = make_pool(["primary", "backup"])
    def send(request, **kwargs):
        raise http_error(request, 429, "Rate limit exceeded", code="1300", request_limit=30)
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError) as error:
        mb.MistralBackend(pool, max_retries=1).complete("s", "u", max_tokens=32)
    assert error.value.category == "rate_limited"
    assert [key["state"] for key in pool.snapshot()] == ["waiting", "standby"]


def test_temporary_rate_limit_retries_same_key_then_keeps_using_it(monkeypatch, fast_clock):
    clock, make_pool = fast_clock
    pool = make_pool(["primary", "backup", "primary"])
    calls = []
    def send(request, **kwargs):
        calls.append((request.headers["Authorization"], clock.now()))
        if len(calls) == 1:
            raise http_error(request, 429, "Rate limit exceeded", retry=60)
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    backend = mb.MistralBackend(pool)
    backend.complete("system", "report A", max_tokens=32)
    backend.fork(1).complete("system", "report B", max_tokens=32)
    assert [key for key, _ in calls] == ["Bearer primary"] * 3
    assert calls[1][1] - calls[0][1] >= 60
    assert [key["state"] for key in pool.snapshot()] == ["active", "standby"]
    assert pool.diagnostics()[0]["status"] == 429
    assert "primary" not in json.dumps(pool.snapshot())


@pytest.mark.parametrize("code,reason", [(401, "invalid key"), (402, "quota exhausted")])
def test_confirmed_invalid_or_exhausted_key_advances_once(monkeypatch, code, reason):
    pool = mb.ApiKeyPool(["bad-key", "good-key"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        if calls[-1] == "Bearer bad-key":
            raise http_error(request, code, "No access for this key")
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert calls == ["Bearer bad-key", "Bearer good-key"]
    assert pool.snapshot()[0]["state"] == reason
    assert pool.snapshot()[1]["state"] == "active"


def test_explicit_monthly_quota_error_advances_to_backup(monkeypatch):
    pool = mb.ApiKeyPool(["primary", "backup"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        if len(calls) == 1:
            raise http_error(request, 429, "Monthly token quota exhausted", code="insufficient_quota")
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert calls == ["Bearer primary", "Bearer backup"]
    assert pool.snapshot()[0]["state"] == "quota exhausted"


@pytest.mark.parametrize("message,code", [
    ("Token quota exceeded per minute", "quota_exceeded"),
    ("Rate limit exceeded", "1300"),
    ("Rate limit exceeded", ""),
    ("Temporary rate limit. Monthly quota still has allowance remaining.", "rate_limited"),
])
def test_all_five_keys_are_not_cycled_on_temporary_or_unknown_429(monkeypatch, fast_clock, message, code):
    clock, make_pool = fast_clock
    pool = make_pool(["primary", "backup-2", "backup-3", "backup-4", "backup-5"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        raise http_error(request, 429, message, code=code, retry=60)
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError) as error:
        mb.MistralBackend(pool, max_retries=3).complete("s", "u", max_tokens=32)
    assert calls == ["Bearer primary"] * 3
    assert error.value.status == 429 and error.value.category == "rate_limited"
    assert "Backup keys remain on standby" in str(error.value)
    assert [key["state"] for key in pool.snapshot()] == ["waiting"] + ["standby"] * 4
    assert clock.now() >= 1120


def test_long_retry_after_preserves_reason_without_spending_backup_attempts(monkeypatch, fast_clock):
    _, make_pool = fast_clock
    pool = make_pool(["primary", "backup"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        raise http_error(request, 429, "Tokens per minute limit reached", code="1300", retry=3600)
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError) as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert calls == ["Bearer primary"]
    assert error.value.status == 429 and "Tokens per minute" in str(error.value)
    assert pool.snapshot()[1]["state"] == "standby"


def test_cooldown_waits_then_wakes_without_holding_condition():
    pool = mb.ApiKeyPool(["key"])
    index, _ = pool.acquire(time.monotonic() + 1)
    pool.release(index, cooldown=0.04)
    start = time.monotonic()
    index, _ = pool.acquire(start + 1)
    assert time.monotonic() - start >= 0.03
    pool.release(index)


def test_queued_worker_waits_for_active_key_and_preserves_its_own_prompt(monkeypatch):
    started, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def send(request, **kwargs):
        payload = json.loads(request.data)
        calls.append((request.headers["Authorization"], payload))
        if len(calls) == 1:
            started.set()
            assert release.wait(timeout=3)
        return response(json.dumps({"report": payload["messages"][1]["content"]}))
    monkeypatch.setattr(mb, "urlopen", send)
    pool = mb.ApiKeyPool(["primary", "backup"])
    backend = mb.MistralBackend(pool)
    def second():
        second_entered.set()
        return backend.fork(1).complete("s", "B only", max_tokens=40)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(backend.complete, "s", "A only", max_tokens=40)
        try:
            assert started.wait(timeout=2)
            next_worker = executor.submit(second)
            assert second_entered.wait(timeout=2)
            assert len(calls) == 1
            assert pool.snapshot()[1]["state"] == "standby"
        finally:
            release.set()
        assert json.loads(first.result())["report"] == "A only"
        assert json.loads(next_worker.result())["report"] == "B only"
    assert [key for key, _ in calls] == ["Bearer primary", "Bearer primary"]
    assert {payload["random_seed"] for _, payload in calls} == {42, 100042}
    assert all(len(payload["messages"]) == 2 for _, payload in calls)
    assert all(payload["model"] == "ministral-14b-2512" for _, payload in calls)
    assert all("reasoning_effort" not in payload for _, payload in calls)


def test_request_starts_are_paced_across_independent_runs(monkeypatch, fast_clock):
    clock, make_pool = fast_clock
    starts = []
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: starts.append(clock.now()) or response())
    for key in ["one", "two", "three"]:
        mb.MistralBackend(make_pool([key])).complete("s", "u", max_tokens=32)
    assert starts == [1000, 1005, 1010]


def test_shared_rate_limit_pause_applies_to_other_runs(monkeypatch, fast_clock):
    clock, make_pool = fast_clock
    starts = []
    def send(request, **kwargs):
        starts.append(clock.now())
        if len(starts) == 1:
            raise http_error(request, 429, "Rate limit exceeded", retry=60)
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError):
        mb.MistralBackend(make_pool(["one"]), max_retries=1).complete("s", "u", max_tokens=32)
    mb.MistralBackend(make_pool(["two"])).complete("s", "u", max_tokens=32)
    assert starts == [1000, 1060]


@pytest.mark.parametrize("finish", ["length", "model_length"])
def test_truncated_completions_remain_retryable_by_pipeline(monkeypatch, finish):
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: response('{"partial":', finish))
    with pytest.raises(CompletionTruncatedError) as error:
        mb.MistralBackend(mb.ApiKeyPool(["secret"])).complete("s", "u", max_tokens=32)
    assert error.value.partial_content == '{"partial":'


@pytest.mark.parametrize("finish", ["error", "tool_calls", None, "unexpected_finish"])
def test_unfinished_completion_preserves_reason_and_keeps_backup_idle(monkeypatch, finish):
    pool = mb.ApiKeyPool(["primary", "backup"])
    calls = []
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: calls.append(True) or response('{"ok":true}', finish))
    with pytest.raises(IncompleteCompletionError) as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert error.value.finish_reason == (finish or "missing")
    assert len(calls) == 1  # The calling stage owns response repair attempts.
    assert pool.diagnostics()[0]["category"] == "incomplete_response"
    assert pool.diagnostics()[0]["code"] == (finish or "missing")
    assert [key["state"] for key in pool.snapshot()] == ["active", "standby"]


@pytest.mark.parametrize("content", [None, ""])
def test_empty_length_limited_reply_is_truncated_not_a_fatal_empty_reply(monkeypatch, content):
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: response(content, "length"))
    with pytest.raises(CompletionTruncatedError) as error:
        mb.MistralBackend(mb.ApiKeyPool(["key"])).complete("s", "u", max_tokens=32)
    assert error.value.partial_content == ""


@pytest.mark.parametrize("body", [b'not json', b'{}', b'{"choices":[]}',
    b'{"choices":[{"finish_reason":"stop","message":{"content":""}}]}'])
def test_malformed_or_empty_reply_is_recoverable_and_does_not_exhaust_keys(monkeypatch, body):
    pool = mb.ApiKeyPool(["primary", "backup"])
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: io.BytesIO(body))
    with pytest.raises(IncompleteCompletionError):
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert pool.diagnostics()[0]["category"] == "incomplete_response"
    assert [key["state"] for key in pool.snapshot()] == ["active", "standby"]


def test_unfinished_response_diagnostics_redact_reason_and_request_id(monkeypatch, caplog):
    body = {"id": "req-private-api-key", "choices": [{"finish_reason": "error private-api-key",
        "message": {"content": '{"partial":'}}]}
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(body).encode()))
    pool = mb.ApiKeyPool(["private-api-key", "backup-api-key"])
    with pytest.raises(IncompleteCompletionError) as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    captured = str(error.value) + error.value.request_id + json.dumps(pool.diagnostics()) + caplog.text
    assert "private-api-key" not in captured and "backup-api-key" not in captured
    assert pool.diagnostics()[0]["requestId"] == "req-[redacted]"


def test_content_filter_is_reported_without_accepting_or_retrying_partial_text(monkeypatch):
    pool = mb.ApiKeyPool(["primary", "backup"])
    calls = []
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: calls.append(True) or response('{"ok":true}', "content_filter"))
    with pytest.raises(mb.MistralRequestError) as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert error.value.category == "content_filtered" and calls == [True]
    assert pool.snapshot()[1]["state"] == "standby"


@pytest.mark.parametrize("recovers", [True, False])
def test_pdf_handles_unfinished_http_200_without_losing_source(tmp_path, monkeypatch, recovers):
    import pdf_ingestion
    from tests.test_pdf_ingestion import module_bytes
    calls = []
    def send(request, **kwargs):
        body = json.loads(request.data)
        calls.append((request.headers["Authorization"], body))
        if len(calls) == 1 or not recovers:
            return response('{"remove":[]}', "error")
        blocks = json.loads(body["messages"][1]["content"])["blocks"]
        remove = ["DECORATIVE" in block["text"] for block in blocks]
        return response(json.dumps({"remove": remove, "study_groups": [[i] for i, removed in enumerate(remove) if not removed]}))
    monkeypatch.setattr(mb, "urlopen", send)
    pool = mb.ApiKeyPool(["primary", "backup"])
    source, target = tmp_path / "Module 5.pdf", tmp_path / "structured_module.txt"
    original = module_bytes()
    source.write_bytes(original)
    pdf_ingestion.convert_pdf(source, target, course_code="CE0055", module_number="5",
        backend=mb.MistralBackend(pool), attempts=2)
    text = target.read_text(encoding="utf-8")
    assert source.read_bytes() == original and "Resistance opposes current flow" in text
    assert len(calls) == 2 and all(key == "Bearer primary" for key, _ in calls)
    assert "finish_reason: error" in calls[1][1]["messages"][0]["content"]
    audit = json.loads(target.with_name("pdf_extraction.json").read_text(encoding="utf-8"))
    assert audit[0]["filter_status"] == ("complete" if recovers else "review_needed")
    assert pool.snapshot()[1]["state"] == "standby"


def test_prompt_overflow_is_rejected_before_network(monkeypatch):
    monkeypatch.setattr(mb, "urlopen", lambda *a, **k: pytest.fail("network must not run"))
    with pytest.raises(ContextWindowExceededError):
        mb.MistralBackend(mb.ApiKeyPool(["key"]), n_ctx=100).complete("s", "u" * 100, max_tokens=32)


@pytest.mark.parametrize("status", [400, 403, 404, 422])
def test_provider_details_are_preserved_but_secrets_are_redacted(monkeypatch, caplog, status):
    pool = mb.ApiKeyPool(["private-api-key", "backup-api-key"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        raise http_error(request, status, "Model access denied private-api-key backup-api-key", code="model_not_available")
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError) as error:
        mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert len(calls) == 1 and "Model access denied" in str(error.value)
    assert error.value.request_id == "request-123"
    captured = str(error.value) + json.dumps(pool.snapshot()) + json.dumps(pool.diagnostics()) + caplog.text
    assert "private-api-key" not in captured and "backup-api-key" not in captured
    assert pool.snapshot()[1]["state"] == "standby"


@pytest.mark.parametrize("status", [None, 500, 503])
def test_network_and_service_retries_stay_on_active_key(monkeypatch, fast_clock, status):
    _, make_pool = fast_clock
    pool = make_pool(["primary", "backup"])
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        if len(calls) == 1:
            if status:
                raise http_error(request, status, "Service unavailable")
            raise URLError("connection refused")
        return response()
    monkeypatch.setattr(mb, "urlopen", send)
    mb.MistralBackend(pool).complete("s", "u", max_tokens=32)
    assert calls == ["Bearer primary", "Bearer primary"]
    assert pool.snapshot()[1]["state"] == "standby"


def test_tls_failure_is_reported_without_retrying_other_keys(monkeypatch):
    calls = []
    def send(request, **kwargs):
        calls.append(request.headers["Authorization"])
        raise URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
    monkeypatch.setattr(mb, "urlopen", send)
    with pytest.raises(mb.MistralRequestError, match="certificate verify failed") as error:
        mb.MistralBackend(mb.ApiKeyPool(["primary", "backup"])).complete("s", "u", max_tokens=32)
    assert error.value.category == "tls_error" and calls == ["Bearer primary"]


def test_first_pdf_page_recovers_from_429_without_touching_four_backups(tmp_path, monkeypatch, fast_clock):
    import pdf_ingestion
    from structured_module import extract_lesson_facts
    from tests.test_pdf_ingestion import module_bytes
    clock, make_pool = fast_clock
    pool = make_pool(["primary", "backup-2", "backup-3", "backup-4", "backup-5"])
    calls = []
    def send(request, **kwargs):
        calls.append((request.headers["Authorization"], clock.now()))
        body = json.loads(request.data)
        assert body["max_tokens"] <= 576
        if len(calls) == 1:
            raise http_error(request, 429, "Token rate limit reached", retry=60)
        blocks = json.loads(body["messages"][1]["content"])["blocks"]
        remove = ["DECORATIVE" in block["text"] for block in blocks]
        return response(json.dumps({"remove": remove, "study_groups": [[i] for i, discarded in enumerate(remove) if not discarded]}))
    monkeypatch.setattr(mb, "urlopen", send)
    source, target = tmp_path / "Module 1.pdf", tmp_path / "structured_module.txt"
    source.write_bytes(module_bytes())
    pdf_ingestion.convert_pdf(source, target, course_code="CE0055", module_number="1", backend=mb.MistralBackend(pool))
    assert [key for key, _ in calls] == ["Bearer primary", "Bearer primary"]
    assert calls[1][1] - calls[0][1] >= 60
    assert [key["state"] for key in pool.snapshot()] == ["active"] + ["standby"] * 4
    assert extract_lesson_facts(target.read_text(encoding="utf-8"))
