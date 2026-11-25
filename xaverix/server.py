"""Xaverix HTTP server: standard library only.

No third-party web framework. Termux installs python, numpy and Pillow as
system packages; anything else would have to be built on the device.
``http.server`` is sufficient for a single-user, on-device application.

The offline guard is installed before anything else can open a socket, which
is why it is imported and applied at the very top of this file rather than
inside main().

Routes
------
GET  /                      the single-page app
GET  /api/config            runtime, model and Llama status
GET  /api/samples           built-in demo images
GET  /api/offline-proof     live proof that outbound access is blocked
GET  /api/guard-demo        the grounding guard applied to invalid outputs
POST /api/analyze           the pipeline
"""
from __future__ import annotations

# --- offline policy first, before any import that could open a socket -----
from app import offline_guard

offline_guard.install()
# --------------------------------------------------------------------------

import argparse
import json
import logging
import mimetypes
import os
import socketserver
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from app import classifier, config, guard, i18n, knowledge, llm, pipeline

LOG = logging.getLogger("xaverix")

MAX_BODY = config.MAX_UPLOAD_BYTES + 1_000_000


def _json_default(value):
    """Coerce NumPy scalars, which json.dumps refuses to serialise.

    The pipeline is full of NumPy maths, and a single np.float64 or np.bool_
    escaping into the response would otherwise fail the whole request with an
    opaque TypeError. Converting here is cheaper than auditing every return.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


def _json_bytes(payload, pretty=False):
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=False,
        default=_json_default,
    )
    return text.encode("utf-8")


class XaverixHandler(BaseHTTPRequestHandler):
    server_version = "Xaverix/%s" % config.APP_VERSION
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ helpers
    def log_message(self, fmt, *args):
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send(self, status, body, content_type="application/json; charset=utf-8",
              extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page must work with the radio off, so nothing may be fetched
        # from a CDN. A strict CSP makes that a guarantee rather than a habit.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, payload, status=200, pretty=True):
        self._send(status, _json_bytes(payload, pretty))

    def _error(self, status, message, **extra):
        payload = {"error": message, "status": status}
        payload.update(extra)
        self._send_json(payload, status=status)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise pipeline.BadImage("invalid Content-Length header")
        if length <= 0:
            raise pipeline.BadImage("empty request body")
        if length > MAX_BODY:
            raise pipeline.BadImage("request body too large")
        return self.rfile.read(length)

    # --------------------------------------------------------------- GET
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        try:
            if route == "/":
                return self._serve_file(config.WEB_DIR / "index.html", "text/html; charset=utf-8")
            if route in ("/api/config", "/api/status"):
                return self._api_config()
            if route == "/api/samples":
                return self._api_samples()
            if route == "/api/offline-proof":
                return self._api_offline_proof()
            if route == "/api/guard-demo":
                return self._api_guard_demo()
            if route == "/api/i18n":
                language = (query.get("lang") or ["en"])[0]
                return self._send_json(
                    {
                        "language": i18n.normalise(language),
                        "strings": i18n.ui_strings(language),
                        "languages": i18n.languages(),
                    }
                )
            if route.startswith("/samples/"):
                name = os.path.basename(route)
                return self._serve_file(config.SAMPLES_DIR / name)
            if route.startswith("/web/") or route in (
                "/app.js",
                "/style.css",
            ):
                name = os.path.basename(route)
                return self._serve_file(config.WEB_DIR / name)
        except BrokenPipeError:
            return
        except Exception:
            LOG.error("GET %s failed\n%s", route, traceback.format_exc())
            return self._error(500, "internal error handling %s" % route)

        return self._error(404, "no such route: %s" % route)

    def _serve_file(self, path, content_type=None):
        path = path.resolve()
        allowed = (config.WEB_DIR.resolve(), config.SAMPLES_DIR.resolve())
        if not any(str(path).startswith(str(root)) for root in allowed):
            return self._error(403, "path not permitted")
        if not path.is_file():
            return self._error(404, "not found: %s" % path.name)
        if content_type is None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with open(path, "rb") as handle:
            body = handle.read()
        self._send(200, body, content_type, {"Cache-Control": "no-store"})

    # ---------------------------------------------------------- API: GET
    def _api_config(self):
        client = llm.client()
        payload = {
            "runtime": config.runtime_info(),
            "llm": client.info(),
            "classifier": classifier.status(),
            "knowledge_base": {
                "version": knowledge.load()["kb_version"],
                "entries": sorted(knowledge.load()["entries"].keys()),
            },
            "crops": knowledge.regions()["crops"],
            "regions": knowledge.regions()["regions"],
            "offline_guard": {
                "active": offline_guard.is_installed(),
                "policy": offline_guard.POLICY_ALLOW,
                "blocked_attempts": len(offline_guard.BLOCKED_ATTEMPTS),
            },
            "disclosures": [
                "Binary prototype: it detects whether damage is present, not which disease or pest.",
                "Trained on procedurally generated leaves. No real-world accuracy is claimed.",
                "Severity is a calibrated colour heuristic, not a segmentation model.",
                "The vision model runs as pure NumPy. Llama is the ExecuTorch component.",
                "Energy figures are estimates, not measurements.",
            ],
        }
        self._send_json(payload)

    def _api_samples(self):
        manifest_path = config.SAMPLES_DIR / "manifest.json"
        samples = []
        if manifest_path.is_file():
            samples = config.load_json(manifest_path).get("samples", [])
        for sample in samples:
            sample["url"] = "/samples/%s" % sample["file"]
        self._send_json({"samples": samples})

    def _api_offline_proof(self):
        self._send_json(offline_guard.proof(local_port=self.server.server_address[1]))

    def _api_guard_demo(self):
        self._send_json(guard.demo())

    # -------------------------------------------------------------- POST
    def do_POST(self):
        route = urlparse(self.path).path.rstrip("/") or "/"
        if route != "/api/analyze":
            return self._error(404, "no such route: %s" % route)

        try:
            body = self._read_body()
        except pipeline.BadImage as exc:
            return self._error(400, str(exc))
        except Exception:
            LOG.error("reading body failed\n%s", traceback.format_exc())
            return self._error(400, "could not read the request body")

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()

        try:
            if content_type == "application/json":
                payload = json.loads(body.decode("utf-8"))
                image_bytes = _decode_data_url(payload.get("image") or "")
                options = payload
            else:
                image_bytes = body
                options = {}
                for key in ("crop", "region", "lang", "language", "use_llm"):
                    value = self.headers.get("X-Xaverix-%s" % key)
                    if value:
                        options[key] = value
        except (ValueError, UnicodeDecodeError) as exc:
            return self._error(400, "invalid request payload: %s" % exc)

        try:
            image = pipeline.decode_image(image_bytes)
        except pipeline.BadImage as exc:
            return self._error(400, str(exc))

        use_llm = options.get("use_llm", True)
        if isinstance(use_llm, str):
            use_llm = use_llm.strip().lower() not in ("0", "false", "no", "off")

        try:
            result = pipeline.analyse(
                image,
                crop=str(options.get("crop") or "tomato"),
                region=str(options.get("region") or knowledge.regions()["default_region"]),
                language=str(options.get("language") or options.get("lang") or "en"),
                use_llm=bool(use_llm),
            )
        except Exception:
            LOG.error("analyze failed\n%s", traceback.format_exc())
            return self._error(500, "analysis failed; see the server log")

        self._send_json(result)


def _decode_data_url(value):
    """Accept either a bare base64 string or a full data: URL."""
    import base64

    if not value:
        return b""
    if value.startswith("data:"):
        _header, _, encoded = value.partition(",")
        value = encoded
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # One request at a time would make a slow Llama call block the whole UI,
    # including the config poll that tells the user Llama is still loading.
    request_queue_size = 16


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Xaverix server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; stays on loopback by default",
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the Llama warm-up")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    print("%s %s - %s" % (config.APP_NAME, config.APP_VERSION, config.APP_TAGLINE))
    print("offline guard: active, permitted targets %s"
          % ", ".join(offline_guard.POLICY_ALLOW))

    # Fail loudly at startup if the knowledge base is inconsistent, rather
    # than per request.
    try:
        kb = knowledge.load()
        print("knowledge base: v%s, %d entries" % (kb["kb_version"], len(kb["entries"])))
    except Exception as exc:
        print("knowledge base failed to load: %s" % exc, file=sys.stderr)
        return 2

    status = classifier.status()
    if status["available"]:
        meta = status.get("training") or {}
        print(
            "classifier: %s, val accuracy %.3f on synthetic data (%s)"
            % (
                status["feature_version"],
                meta.get("val_accuracy_synthetic", float("nan")),
                "SYNTHETIC - not a real-world accuracy",
            )
        )
    else:
        print("classifier: UNAVAILABLE - %s" % status["error"])

    client = llm.client()
    info = client.info()
    print(
        "llama: %s at %s (model %s)"
        % (
            "usable" if info.get("usable") else "not reachable (%s)" % info.get("reason"),
            info.get("endpoint"),
            info.get("model"),
        )
    )
    if client.enabled and not args.no_warmup:
        client.warm_up_async()
        print("llama: warm-up started in the background")

    httpd = Server((args.host, args.port), XaverixHandler)
    url = "http://%s:%d" % ("localhost" if args.host == "127.0.0.1" else args.host, args.port)
    print("\nopen %s in the phone browser" % url)
    print("press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
