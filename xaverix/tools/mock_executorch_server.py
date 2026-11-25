"""Development stand-in for the ExecuServe OpenAI-compatible endpoint.

ExecuServe runs only on the Android device, so the language-model integration
cannot otherwise be exercised on a development machine or in CI. This serves
the same protocol on loopback, allowing the full path -- client, prompt,
grounding guard, provenance and fallback -- to be tested off-device.

The response mode is selectable, including a fabricated-pathogen answer
representative of small-model failure, so the guard can be verified end to end
rather than only in unit tests.

    python tools/mock_executorch_server.py --port 8080 --mode faithful
    python tools/mock_executorch_server.py --port 8080 --mode unfaithful
    python tools/mock_executorch_server.py --port 8080 --mode slow --delay 25

Then, in another shell:

    XAVERIX_LLM=1 XAVERIX_LLM_KEY=test python server.py --port 8000

This is a development tool. It is NOT part of the on-device deliverable, and
nothing in app/ imports it.
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "llama-3.2-1b-instruct"

# A faithful rewrite: every number and product comes from the knowledge base.
FAITHFUL = """ Your tomato leaf shows brown dead patches with yellowing around them. About 38 percent of the leaf blade is damaged.
WHAT TO DO NOW:
Act within 24 hours. Remove the affected lower leaves and check the neighbouring plants, because the damage is spreading. Spray mancozeb 75 WP at 2 g per litre of water, every 10 days.
ORGANIC OPTION:
Spray neem oil at 5 ml per litre of water every 7 days, in the evening, and wet both sides of the leaf.
PREVENT NEXT SEASON:
Switch from overhead watering to drip, mulch the soil, and rotate to a non-host crop for two to three seasons."""

# A fabricated pathogen, an invented dose and an unsupported guarantee.
# The guard must reject this.
UNFAITHFUL = """ Early blight of tomato is caused by the bacterium Colletobacter, which spreads through the air.
WHAT TO DO NOW:
Spray azoxystrobin 25 SC at 1.5 ml per litre immediately.
ORGANIC OPTION:
Neem oil at 12 ml per litre every 3 days.
PREVENT NEXT SEASON:
This will completely eliminate the disease, guaranteed."""

# Formatted correctly but drifts entirely off the supplied facts.
OFF_TOPIC = """ The weather has been variable this season across many districts.
WHAT TO DO NOW:
Consider speaking to neighbours about market prices and transport arrangements.
ORGANIC OPTION:
Composting kitchen waste improves soil structure over several years.
PREVENT NEXT SEASON:
Attend a training programme if one is offered nearby."""

MODES = {
    "faithful": FAITHFUL,
    "unfaithful": UNFAITHFUL,
    "offtopic": OFF_TOPIC,
}


class MockHandler(BaseHTTPRequestHandler):
    server_version = "MockExecuServe/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("  [mock] %s" % (fmt % args))

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            # Expected when the client gives up first, which the timeout test
            # does on purpose. Swallow it rather than printing a traceback.
            pass

    def handle_one_request(self):
        try:
            BaseHTTPRequestHandler.handle_one_request(self)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    def _auth_ok(self):
        expected = self.server.api_key
        if not expected:
            return True
        header = self.headers.get("Authorization") or ""
        return header == "Bearer %s" % expected

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            if not self._auth_ok():
                return self._send(401, {"error": {"message": "invalid api key"}})
            return self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": MODEL_ID, "object": "model", "owned_by": "executorch"}
                    ],
                },
            )
        self._send(404, {"error": {"message": "not found: %s" % self.path}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "not found: %s" % self.path}})
        if not self._auth_ok():
            return self._send(401, {"error": {"message": "invalid api key"}})

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except ValueError:
            return self._send(400, {"error": {"message": "invalid json"}})

        messages = request.get("messages") or []
        max_tokens = request.get("max_tokens") or 0

        # A 4-token request is the warm-up ping; answer it instantly.
        if max_tokens and max_tokens <= 8:
            return self._send(200, self._completion("OK"))

        if self.server.delay:
            time.sleep(self.server.delay)

        print(
            "  [mock] prompt: %d messages, %d chars, prefill=%r"
            % (
                len(messages),
                sum(len(m.get("content") or "") for m in messages),
                (messages[-1].get("content") if messages else "")[:24],
            )
        )
        self._send(200, self._completion(MODES[self.server.mode]))

    def _completion(self, content):
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 730, "completion_tokens": 120, "total_tokens": 850},
        }


class MockServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port=8080, mode="faithful", delay=0.0, api_key="test"):
    httpd = MockServer(("127.0.0.1", port), MockHandler)
    httpd.mode = mode
    httpd.delay = delay
    httpd.api_key = api_key
    return httpd


def main():
    parser = argparse.ArgumentParser(description="Mock the ExecuServe endpoint")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", choices=sorted(MODES), default="faithful")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds per generation")
    parser.add_argument("--key", default="test")
    args = parser.parse_args()

    httpd = serve(args.port, args.mode, args.delay, args.key)
    print(
        "mock ExecuServe on http://127.0.0.1:%d/v1 (model %s, mode %s, key %r)"
        % (args.port, MODEL_ID, args.mode, args.key)
    )
    print("this is a DEVELOPMENT STAND-IN for the real on-device endpoint")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
