"""Stateless Mistral requests with ordered failover and paced retries."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import os
import re
import ssl
import threading
import time
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flashcard_types import CompletionTruncatedError, ContextWindowExceededError, IncompleteCompletionError

DEFAULT_MODEL = "ministral-14b-2512"
DEFAULT_N_CTX = 32768
DEFAULT_REQUEST_INTERVAL = 5.0
DEFAULT_RATE_LIMIT_WAIT = 90.0
DEFAULT_MAX_RETRIES = 8
DEFAULT_RETRY_WAIT_TIMEOUT = 1800.0
API_URL = "https://api.mistral.ai/v1/chat/completions"
_LOGGER = logging.getLogger(__name__)


class MistralRequestError(RuntimeError):
    def __init__(self, message: str, *, category: str, status: int | None = None,
                 provider_code: str = "", request_id: str = "") -> None:
        super().__init__(message)
        self.category = category
        self.status = status
        self.provider_code = provider_code
        self.request_id = request_id


def retry_after_seconds(value: str | None, default: float = DEFAULT_RATE_LIMIT_WAIT) -> float:
    try:
        seconds = float(value)
        return max(1, seconds) if math.isfinite(seconds) else default
    except (ValueError, TypeError):
        try:
            target = parsedate_to_datetime(value)
            return max(1, (target - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return default


class RequestPacer:
    """Space requests and share temporary backoff across local runs."""
    def __init__(self, interval: float = DEFAULT_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self._condition = threading.Condition()
        self._next_start = 0.0
        self._blocked_until = 0.0
        self._clock = time.monotonic

    def acquire(self, deadline: float) -> None:
        with self._condition:
            while True:
                now = self._clock()
                ready = max(self._next_start, self._blocked_until)
                if now >= deadline or ready > deadline:
                    raise MistralRequestError(
                        "Mistral's temporary rate-limit wait exceeds this request's retry window. "
                        "Retry after the displayed reset; backup keys remain on standby.", category="rate_limited")
                if now >= ready:
                    self._next_start = now + self.interval
                    return
                self._condition.wait(min(ready - now, deadline - now))

    def defer(self, seconds: float) -> None:
        with self._condition:
            self._blocked_until = max(self._blocked_until, self._clock() + seconds)
            self._condition.notify_all()


_REQUEST_PACER = RequestPacer()


@dataclass
class _Key:
    secret: str = field(repr=False)
    cooldown_until: float = 0
    disabled: bool = False
    busy: bool = False
    disabled_reason: str = ""
    last_error: dict[str, object] | None = None


class ApiKeyPool:
    """Keep backups idle until the active key is invalid or quota is exhausted."""
    def __init__(self, keys: list[str]) -> None:
        unique = list(dict.fromkeys(key.strip() for key in keys if key.strip()))
        if not unique or len(unique) > 100:
            raise ValueError("Enter between 1 and 100 Mistral API keys.")
        if any(len(key) > 512 or any(ch.isspace() for ch in key) for key in unique):
            raise ValueError("API keys cannot contain whitespace or exceed 512 characters.")
        self._keys = [_Key(secret=key) for key in unique]
        self._condition = threading.Condition()
        # One logical completion owns the queue through waits and retries.
        # Additional cluster workers wait here instead of taking backup keys.
        self.request_lock = threading.Lock()
        self._current = 0
        self._clock = time.monotonic
        self._diagnostics: deque[dict[str, object]] = deque(maxlen=20)

    def redact(self, value: object, limit: int = 400) -> str:
        text = str(value or "")
        for key in sorted(self._keys, key=lambda item: len(item.secret), reverse=True):
            text = text.replace(key.secret, "[redacted]")
        text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", text)
        return re.sub(r"[\x00-\x1f\x7f]+", " ", text).strip()[:limit]

    def acquire(self, deadline: float) -> tuple[int, str]:
        with self._condition:
            while True:
                while self._current < len(self._keys) and self._keys[self._current].disabled:
                    self._current += 1
                if self._current >= len(self._keys):
                    reasons = "; ".join(f"Key {i + 1}: {key.disabled_reason}" for i, key in enumerate(self._keys))
                    raise MistralRequestError("No usable Mistral keys remain. " + reasons,
                                              category="keys_unavailable")
                index, now = self._current, self._clock()
                key = self._keys[index]
                if now >= deadline or key.cooldown_until > deadline:
                    last = key.last_error or {}
                    if not last:
                        raise MistralRequestError(
                            f"The retry window ended before Key {index + 1} could be used. "
                            "Untried backup keys have not been marked exhausted.", category="retry_timeout")
                    raise MistralRequestError(
                        f"Key {index + 1} is still waiting after a temporary Mistral error. "
                        f"{last.get('message', '')} Retry after the displayed reset. "
                        "This does not establish that the key's quota is exhausted.",
                        category=str(last.get("category", "rate_limited")),
                        status=last.get("status"), provider_code=str(last.get("code", "")),
                        request_id=str(last.get("requestId", "")))
                if not key.busy and key.cooldown_until <= now:
                    key.busy = True
                    return index, key.secret
                delay = key.cooldown_until - now if not key.busy else deadline - now
                self._condition.wait(min(max(0.01, delay), deadline - now))

    def record_error(self, index: int, error: MistralRequestError) -> None:
        """Record a redacted response error without changing key availability."""
        with self._condition:
            detail = {"key": f"Key {index + 1}", "status": error.status,
                      "category": error.category, "message": self.redact(error),
                      "code": self.redact(error.provider_code, 100),
                      "requestId": self.redact(error.request_id, 100)}
            self._keys[index].last_error = detail
            self._diagnostics.append(detail)
            _LOGGER.warning("Mistral %s: %s", detail["key"], detail["message"])

    def release(self, index: int, *, cooldown: float = 0, disable: bool = False,
                error: MistralRequestError | None = None) -> None:
        with self._condition:
            key = self._keys[index]
            key.busy = False
            key.cooldown_until = max(key.cooldown_until, self._clock() + cooldown)
            if error is not None:
                self.record_error(index, error)
            elif not cooldown:
                key.last_error = None
            if disable:
                key.disabled = True
                key.disabled_reason = error.category.replace("_", " ") if error else "disabled"
                if self._current == index:
                    self._current += 1
            self._condition.notify_all()

    def snapshot(self) -> list[dict[str, object]]:
        with self._condition:
            now = self._clock()
            result = []
            for index, key in enumerate(self._keys):
                state = "standby"
                if key.disabled:
                    state = key.disabled_reason
                elif index == self._current:
                    state = "waiting" if key.cooldown_until > now else "busy" if key.busy else "active"
                result.append({"label": f"Key {index + 1}", "state": state,
                    "retryAfterSeconds": max(0, math.ceil(key.cooldown_until - now)),
                    "lastError": dict(key.last_error) if key.last_error else None})
            return result

    def diagnostics(self) -> list[dict[str, object]]:
        with self._condition:
            return [dict(item) for item in self._diagnostics]


_default_lock = threading.Lock()
_default_pool: ApiKeyPool | None = None


def default_key_pool() -> ApiKeyPool:
    global _default_pool
    with _default_lock:
        if _default_pool is None:
            raw = os.environ.get("MISTRAL_API_KEYS") or os.environ.get("MISTRAL_API_KEY", "")
            _default_pool = ApiKeyPool(raw.replace(",", "\n").splitlines())
        return _default_pool


def _quota_exhausted(message: str, code: str) -> bool:
    value = (code + " " + message).lower().replace("_", " ").replace("-", " ")
    if re.search(r"per (?:second|minute)|tokens?/(?:s|m)\b|tokens? per minute|\brps\b|\btpm\b", value):
        return False
    explicit = re.search(
        r"insufficient (?:quota|credit|balance)|quota (?:exhausted|depleted)|"
        r"(?:credit|balance).{0,25}(?:exhausted|depleted|insufficient)|"
        r"(?:no credits|payment required)|hard limit.{0,20}(?:reached|exceeded)", value)
    scoped = re.search(r"(?:monthly|daily|billing|spend(?:ing)?).{0,40}(?:limit|quota|cap)", value)
    reached = re.search(r"exceed|exhaust|deplet|reached|insufficient", value)
    return bool(explicit or (scoped and reached))


def _http_failure(exc: HTTPError, pool: ApiKeyPool, index: int) -> tuple[MistralRequestError, str | None]:
    retry = exc.headers.get("Retry-After") if exc.headers else None
    request_id = (exc.headers.get("x-request-id") or exc.headers.get("x-correlation-id") or "") if exc.headers else ""
    try:
        zero_request_allowance = float(exc.headers.get("x-ratelimit-limit-req-minute")) == 0
    except (AttributeError, TypeError, ValueError):
        zero_request_allowance = False
    code, message = "", ""
    try:
        data = json.loads(exc.read(16384))
        if isinstance(data, dict):
            error = data.get("error", data)
            if isinstance(error, dict):
                message = error.get("message", "")
                code = error.get("code") or error.get("type") or ""
    except (ValueError, OSError, TypeError):
        pass
    status = exc.code
    exc.close()
    category = "request_rejected"
    if status == 401:
        category = "invalid_key"
    elif status == 402 or (status == 429 and _quota_exhausted(str(message), str(code))):
        category = "quota_exhausted"
    elif status == 429 and zero_request_allowance:
        category = "model_unavailable"
        message = (str(message) + " This model's request allowance is 0 per minute for this key. "
                   "Choose a model allowed by your subscription. Waiting or retrying cannot increase that allowance.")
    elif status == 429:
        category = "rate_limited"
    elif status in (408, 500, 502, 503, 504):
        category = "service_error"
    elif status == 403:
        category = "access_denied"
    message = pool.redact(message) or {
        "rate_limited": "Temporary request or token rate limit reached.",
        "quota_exhausted": "Billing or usage allowance is unavailable.",
        "invalid_key": "The API key is invalid or expired.",
        "access_denied": "This key cannot access the requested model or service.",
        "service_error": "Mistral returned a temporary service error.",
    }.get(category, "Check the model and request settings.")
    return MistralRequestError(f"Mistral {category.replace('_', ' ')} (HTTP {status}, Key {index + 1}): {message}",
        category=category, status=status, provider_code=pool.redact(code, 100),
        request_id=pool.redact(request_id, 100)), retry


class MistralBackend:
    auto_cluster_workers = 1
    auto_cluster_workers_reason = "Ordered Mistral key queue: one active request at a time."

    def __init__(self, pool: ApiKeyPool, *, model: str = DEFAULT_MODEL,
                 n_ctx: int = DEFAULT_N_CTX, seed: int = 42, timeout: float = 120,
                 retry_wait_timeout: float = DEFAULT_RETRY_WAIT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self.pool = pool
        self.model_name = model
        self.context_window = n_ctx
        self.seed = seed
        self.timeout = timeout
        self.retry_wait_timeout = retry_wait_timeout
        self.max_retries = max_retries

    def fork(self, worker_index: int) -> MistralBackend:
        return MistralBackend(self.pool, model=self.model_name, n_ctx=self.context_window,
            seed=self.seed + worker_index * 100000, timeout=self.timeout,
            retry_wait_timeout=self.retry_wait_timeout, max_retries=self.max_retries)

    def close(self) -> None:
        pass

    def count_prompt_tokens(self, system: str, user: str) -> int:
        return len((system + user).encode("utf-8")) + 64

    def _completion_text(self, result: object, request_id: str) -> str:
        finish = "missing"
        try:
            if not isinstance(result, dict):
                raise ValueError("response is not an object")
            request_id = request_id or self.pool.redact(result.get("id", ""), 100)
            choice = result["choices"][0]
            if not isinstance(choice, dict):
                raise ValueError("choice is not an object")
            finish = self.pool.redact(choice.get("finish_reason") or "missing", 80)
            message = choice.get("message")
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content
                                  if isinstance(part, dict) and part.get("type") == "text")
            if finish in ("length", "model_length"):
                usage = result.get("usage") or {}
                usage = usage if isinstance(usage, dict) else {}
                raise CompletionTruncatedError(
                    f"Mistral output was cut short (finish_reason: {finish}).",
                    partial_content=content if isinstance(content, str) else "",
                    prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"))
            if finish in ("content_filter", "refusal") or (isinstance(message, dict) and message.get("refusal")):
                raise MistralRequestError("Mistral declined this response through its content filter.",
                    category="content_filtered", provider_code=finish, request_id=request_id)
            if finish != "stop":
                raise IncompleteCompletionError(
                    f"Mistral returned an unfinished response (finish_reason: {finish}).",
                    finish_reason=finish, request_id=request_id)
            if not isinstance(content, str) or not content.strip():
                raise IncompleteCompletionError("Mistral returned an empty text response (finish_reason: stop).",
                    finish_reason=finish, request_id=request_id)
            return content
        except (KeyError, IndexError, TypeError, ValueError):
            raise IncompleteCompletionError("Mistral returned a malformed completion response.",
                finish_reason=finish, request_id=request_id) from None

    def complete(self, system: str, user: str, *, max_tokens: int,
                 schema: Mapping[str, object] | None = None, **_options: object) -> str:
        if self.count_prompt_tokens(system, user) + max_tokens > self.context_window:
            raise ContextWindowExceededError("Prompt exceeds the configured Mistral context budget.")
        response_format = {"type": "json_object"} if schema is None else {
            "type": "json_schema", "json_schema": {"name": "result", "schema": dict(schema), "strict": True}}
        body = {"model": self.model_name,
            "messages": [{"role": "system", "content": system + "\nReturn JSON only."},
                         {"role": "user", "content": user}],
            "temperature": 0, "random_seed": self.seed, "max_tokens": max_tokens,
            "response_format": response_format, "stream": False}
        if self.model_name in {"mistral-small-latest", "mistral-small-2603"} or self.model_name.startswith("mistral-small-4"):
            body["reasoning_effort"] = "none"
        payload = json.dumps(body).encode("utf-8")
        with self.pool.request_lock:
            # Queued workers don't spend their retry window waiting for a turn.
            deadline = time.monotonic() + self.retry_wait_timeout
            temporary_failures = 0
            while True:
                index, secret = self.pool.acquire(deadline)
                failure, cooldown, disable = None, 0.0, False
                response_error = None
                request_id = ""
                try:
                    _REQUEST_PACER.acquire(deadline)
                    request = Request(API_URL, data=payload, headers={
                        "Authorization": f"Bearer {secret}", "Content-Type": "application/json"})
                    with urlopen(request, timeout=min(self.timeout, max(1, deadline - time.monotonic()))) as response:
                        headers = getattr(response, "headers", None) or {}
                        request_id = self.pool.redact(headers.get("x-request-id") or headers.get("x-correlation-id") or "", 100)
                        result = json.load(response)
                except HTTPError as exc:
                    failure, retry = _http_failure(exc, self.pool, index)
                    disable = failure.category in {"invalid_key", "quota_exhausted"}
                    if not disable and failure.category in {"rate_limited", "service_error"}:
                        temporary_failures += 1
                        if failure.category == "rate_limited":
                            cooldown = retry_after_seconds(retry)
                            _REQUEST_PACER.defer(cooldown)
                        else:
                            cooldown = retry_after_seconds(retry, default=min(30, 2 ** (temporary_failures - 1)))
                except (URLError, TimeoutError, OSError) as exc:
                    reason = getattr(exc, "reason", exc)
                    category = "tls_error" if isinstance(reason, ssl.SSLError) else "connection_error"
                    failure = MistralRequestError("Could not connect to Mistral: " + self.pool.redact(reason), category=category)
                    if category != "tls_error":
                        temporary_failures += 1
                        cooldown = min(30, 2 ** (temporary_failures - 1))
                except ValueError:
                    response_error = IncompleteCompletionError("Mistral returned invalid response JSON.",
                        finish_reason="invalid_json", request_id=request_id)
                    failure = MistralRequestError(str(response_error), category="incomplete_response",
                        provider_code="invalid_json", request_id=request_id)
                except MistralRequestError as exc:
                    failure = exc
                finally:
                    # Like the PDF automation, pause after a response as well as
                    # spacing starts. Long requests must not erase the pause.
                    _REQUEST_PACER.defer(_REQUEST_PACER.interval)
                    self.pool.release(index, cooldown=cooldown, disable=disable, error=failure)
                if response_error is not None:
                    raise response_error from None
                if failure is not None:
                    if disable:
                        continue
                    if cooldown and temporary_failures < self.max_retries:
                        continue
                    if cooldown:
                        raise MistralRequestError(str(failure) +
                            f" Still failing after {temporary_failures} attempts on the active key. "
                            "Backup keys remain on standby because exhausted quota was not confirmed.",
                            category=failure.category, status=failure.status,
                            provider_code=failure.provider_code, request_id=failure.request_id) from None
                    raise failure from None
                try:
                    return self._completion_text(result, request_id)
                except IncompleteCompletionError as exc:
                    self.pool.record_error(index, MistralRequestError(str(exc), category="incomplete_response",
                        provider_code=exc.finish_reason, request_id=exc.request_id))
                    raise
                except MistralRequestError as exc:
                    self.pool.record_error(index, exc)
                    raise


def backend_from_args(args: object) -> MistralBackend:
    return MistralBackend(getattr(args, "_mistral_pool", None) or default_key_pool(),
        model=getattr(args, "mistral_model", None) or os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL),
        n_ctx=getattr(args, "n_ctx", DEFAULT_N_CTX), seed=getattr(args, "seed", 42))
