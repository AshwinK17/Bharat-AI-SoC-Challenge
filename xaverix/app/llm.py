"""Client for the on-device Llama endpoint hosted by ExecuServe.

ExecuServe is an Android app that runs a model through **ExecuTorch** on the
phone's Arm CPU and exposes an OpenAI-compatible HTTP server on loopback. That
is where the brief's "Llama deployed on-device via ExecuTorch" requirement is
actually satisfied: the ExecuTorch runtime and the INT4/XNNPACK .pte live inside
ExecuServe, and this module is the client.

Constraints that shaped this file:

* ``urllib.request`` only. Termux cannot build ``requests``, and the whole
  point is to depend on nothing but the standard library, numpy and Pillow.
* Loopback only. A non-loopback URL is refused here as well as by
  ``offline_guard``, so a misconfigured endpoint cannot phone home.
* A cold 1B model on a phone can take 10-60 s to produce its first token, so
  a background warm-up runs at startup and the first request waits only a
  bounded time before falling back.
* Every failure is typed and degrades to the knowledge-base advisory. Nothing
  from this module is allowed to reach the HTTP handler as an exception.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

from app import config
from app.offline_guard import _is_local

# Never let a system proxy intercept a loopback call.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

REASONS = (
    "disabled",
    "not_loopback",
    "not_running",
    "timeout",
    "bad_key",
    "bad_model",
    "bad_response",
    "cold",
)


class LLMUnavailable(Exception):
    """The endpoint could not produce a usable answer. Carries a reason code."""

    def __init__(self, reason, detail=""):
        super().__init__("%s: %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


def _host_of(url):
    without_scheme = url.split("://", 1)[-1]
    host = without_scheme.split("/", 1)[0]
    if host.startswith("["):  # IPv6 literal
        return host.split("]", 1)[0].strip("[]")
    return host.split(":", 1)[0]


class ExecuServeClient:
    """Thin, defensive client for one local OpenAI-compatible endpoint."""

    def __init__(self, url=None, key=None, model=None, enabled=None):
        self.base = (url if url is not None else config.LLM_URL).rstrip("/")
        self.key = key if key is not None else config.LLM_KEY
        self.model = model if model is not None else config.LLM_MODEL
        self.enabled = config.LLM_ENABLED if enabled is None else enabled

        self.host = _host_of(self.base)
        self.loopback_only = _is_local(self.host)

        self._warm = threading.Event()
        self._warming = False
        self._lock = threading.Lock()
        self._probe_cache = (0.0, None)
        self._last_error = None

    # ------------------------------------------------------------ internals
    def _require_usable(self):
        if not self.enabled:
            raise LLMUnavailable("disabled", "XAVERIX_LLM is not enabled")
        if not self.loopback_only:
            raise LLMUnavailable(
                "not_loopback",
                "refusing endpoint %r: Xaverix only talks to loopback" % self.base,
            )

    def _request(self, path, payload, timeout):
        self._require_usable()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers=headers,
            method="GET" if payload is None else "POST",
        )
        try:
            with _OPENER.open(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read()[:300].decode("utf-8", "replace")
            except Exception:
                pass
            reason = {401: "bad_key", 403: "bad_key", 404: "bad_model"}.get(
                exc.code, "not_running"
            )
            self._last_error = "HTTP %s %s" % (exc.code, body)
            raise LLMUnavailable(reason, self._last_error)
        except urllib.error.URLError as exc:
            inner = getattr(exc, "reason", exc)
            reason = "timeout" if isinstance(inner, socket.timeout) else "not_running"
            self._last_error = repr(inner)
            raise LLMUnavailable(reason, self._last_error)
        except socket.timeout:
            self._last_error = "read timeout after %.0fs" % timeout
            raise LLMUnavailable("timeout", self._last_error)
        except (ValueError, KeyError, TypeError) as exc:
            self._last_error = repr(exc)
            raise LLMUnavailable("bad_response", self._last_error)

    # ------------------------------------------------------------- probing
    def probe(self, ttl=None):
        """Is the endpoint answering? Cached, because the UI asks often."""
        ttl = config.LLM_PROBE_TTL if ttl is None else ttl
        now = time.time()
        stamp, cached = self._probe_cache
        if cached is not None and now - stamp < ttl:
            return cached

        try:
            body = self._request("/models", None, config.LLM_PROBE_TIMEOUT)
            names = []
            for item in (body or {}).get("data") or []:
                if isinstance(item, dict) and item.get("id"):
                    names.append(item["id"])
            result = {"reachable": True, "models": names, "reason": None}
        except LLMUnavailable as exc:
            result = {"reachable": False, "models": [], "reason": exc.reason}

        self._probe_cache = (now, result)
        return result

    # ------------------------------------------------------------ warm-up
    def warm_up_async(self):
        """Load the model in the background so request #1 is not the cold one."""
        with self._lock:
            if self._warming or self._warm.is_set() or not self.enabled:
                return
            self._warming = True

        def run():
            try:
                self._request(
                    "/chat/completions",
                    {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                    config.LLM_WARMUP_TIMEOUT,
                )
                self._warm.set()
            except LLMUnavailable:
                pass  # info() reports it; the advisory falls back cleanly
            finally:
                self._warming = False

        threading.Thread(target=run, name="xaverix-llm-warmup", daemon=True).start()

    def wait_warm(self, timeout=None):
        timeout = config.LLM_COLD_WAIT if timeout is None else timeout
        return self._warm.wait(timeout)

    def is_warm(self):
        return self._warm.is_set()

    # ---------------------------------------------------------- generation
    def chat(self, messages, max_tokens=None, temperature=None, timeout=None):
        """One completion. Returns (text, elapsed_ms). Raises LLMUnavailable."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
            "temperature": (
                config.LLM_TEMPERATURE if temperature is None else temperature
            ),
            "top_p": config.LLM_TOP_P,
            "stream": False,
        }
        started = time.perf_counter()
        body = self._request(
            "/chat/completions",
            payload,
            config.LLM_GEN_TIMEOUT if timeout is None else timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        try:
            choice = body["choices"][0]
            text = choice.get("message", {}).get("content")
            if text is None:
                text = choice.get("text", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable("bad_response", "unexpected shape: %r" % (exc,))

        if not (text or "").strip():
            raise LLMUnavailable("bad_response", "empty completion")

        self._warm.set()
        return text.strip(), elapsed_ms

    # ---------------------------------------------------------------- info
    def info(self):
        """Surfaced at /api/config so `"usable": true` is verifiable by curl."""
        if not self.enabled:
            return {
                "configured": False,
                "reachable": False,
                "usable": False,
                "warm": False,
                "reason": "disabled",
                "endpoint": self.base,
                "model": self.model,
                "runtime": "ExecuTorch (XNNPACK) via ExecuServe",
                "key_set": bool(self.key),
            }
        if not self.loopback_only:
            return {
                "configured": True,
                "reachable": False,
                "usable": False,
                "warm": False,
                "reason": "not_loopback",
                "endpoint": self.base,
                "model": self.model,
                "runtime": "ExecuTorch (XNNPACK) via ExecuServe",
                "key_set": bool(self.key),
            }

        probe = self.probe()
        model_present = (not probe["models"]) or any(
            self.model in name or name in self.model for name in probe["models"]
        )
        return {
            "configured": True,
            "reachable": probe["reachable"],
            "usable": bool(probe["reachable"] and model_present),
            "warm": self.is_warm(),
            "reason": probe["reason"] if not probe["reachable"] else None,
            "endpoint": self.base,
            "model": self.model,
            "models_offered": probe["models"],
            "runtime": "ExecuTorch (XNNPACK) via ExecuServe",
            "key_set": bool(self.key),
            "last_error": self._last_error,
        }


_default_client = None


def client():
    """Process-wide client, created on first use."""
    global _default_client
    if _default_client is None:
        _default_client = ExecuServeClient()
    return _default_client


def reset_client():
    """Drop the cached client; used by tests after changing config."""
    global _default_client
    _default_client = None
