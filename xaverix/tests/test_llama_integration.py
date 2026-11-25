"""End-to-end language-model integration against a loopback stand-in.

ExecuServe runs only on the Android device, so this suite is how the full path
-- client, constrained prompt, grounding guard, retry, provenance and fallback
-- is exercised on a development machine or in CI.

The stand-in speaks the same OpenAI-compatible protocol and can return a
fabricated-pathogen answer representative of small-model failure.
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import advisory, knowledge, llm  # noqa: E402
from tools import mock_executorch_server as mock  # noqa: E402

PORT = 8231
URL = "http://127.0.0.1:%d/v1" % PORT
KEY = "test-key"


class _MockEndpoint:
    """Run the mock in a background thread for the duration of a test."""

    def __init__(self, mode="faithful", delay=0.0, key=KEY):
        self.httpd = mock.serve(port=PORT, mode=mode, delay=delay, api_key=key)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        time.sleep(0.2)
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        time.sleep(0.1)
        return False


def _client(key=KEY):
    return llm.ExecuServeClient(url=URL, key=key, enabled=True)


def _build(client):
    return advisory.build(
        knowledge.entry_for("infected"),
        "tomato",
        "tamil_nadu",
        38.0,
        "moderate",
        74.0,
        client=client,
    )


class LlamaIntegrationTest(unittest.TestCase):
    def test_probe_reports_the_hosted_model(self):
        with _MockEndpoint():
            info = _client().info()
            self.assertTrue(info["usable"], info)
            self.assertIn(mock.MODEL_ID, info["models_offered"])
            self.assertEqual(info["runtime"], "ExecuTorch (XNNPACK) via ExecuServe")

    def test_faithful_output_is_accepted_and_attributed(self):
        with _MockEndpoint(mode="faithful"):
            result = _build(_client())

        self.assertEqual(result["source"], "llama_grounded")
        self.assertEqual(result["guard"]["decision"], "accept")
        self.assertGreater(result["guard"]["support"], 0.70)
        self.assertIsNotNone(result["llama_note"])
        self.assertEqual(result["model"]["name"], mock.MODEL_ID)
        self.assertIn("ExecuTorch", result["model"]["runtime"])
        self.assertIn("ExecuTorch", result["source_label"])
        # The four sections must come from the generated text.
        self.assertEqual(
            [s["key"] for s in result["sections"]], ["what", "now", "organic", "prevent"]
        )

    def test_unfaithful_output_is_rejected_and_falls_back(self):
        """A fabricated pathogen must not reach the displayed advisory."""
        with _MockEndpoint(mode="unfaithful"):
            result = _build(_client())

        self.assertEqual(result["source"], "kb_fallback_guard_reject")
        self.assertEqual(result["guard"]["decision"], "reject")
        self.assertIn("ORGANISM_NOT_IN_KB", result["guard"]["codes"])

        # The displayed advisory must be the knowledge-base text and must
        # not contain any of the unsupported content.
        shown = " ".join(section["body"] for section in result["sections"])
        self.assertNotIn("Colletobacter", shown)
        self.assertNotIn("azoxystrobin", shown)
        self.assertNotIn("guaranteed", shown)
        self.assertTrue(shown.strip())

    def test_off_topic_output_is_rejected_for_low_support(self):
        with _MockEndpoint(mode="offtopic"):
            result = _build(_client())

        self.assertEqual(result["source"], "kb_fallback_guard_reject")
        self.assertIn("LOW_SUPPORT", result["guard"]["codes"])

    def test_a_wrong_api_key_degrades_cleanly(self):
        with _MockEndpoint(key="the-right-key"):
            client = _client(key="the-wrong-key")
            info = client.info()
            self.assertFalse(info["usable"])
            self.assertEqual(info["reason"], "bad_key")

            result = _build(client)
            self.assertEqual(result["source"], "kb_fallback_llm_unavailable")
            self.assertTrue(result["sections"])

    def test_generation_timeout_degrades_cleanly(self):
        with _MockEndpoint(mode="faithful", delay=2.0):
            client = _client()
            client.probe()
            client._warm.set()  # bypass the warm-up wait for this test
            with self.assertRaises(llm.LLMUnavailable) as caught:
                client.chat([{"role": "user", "content": "hello"}], timeout=0.5)
            self.assertIn(caught.exception.reason, ("timeout", "not_running"))

    def test_warm_up_marks_the_client_warm(self):
        with _MockEndpoint():
            client = _client()
            self.assertFalse(client.is_warm())
            client.warm_up_async()
            self.assertTrue(client.wait_warm(10.0))
            self.assertTrue(client.is_warm())

    def test_the_prompt_sent_is_the_caged_one(self):
        """What reaches the model must carry the constraints and the prefill."""
        captured = {}
        original = llm.ExecuServeClient.chat

        def spy(self, messages, **kwargs):
            captured["messages"] = messages
            return original(self, messages, **kwargs)

        llm.ExecuServeClient.chat = spy
        try:
            with _MockEndpoint(mode="faithful"):
                _build(_client())
        finally:
            llm.ExecuServeClient.chat = original

        messages = captured["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertIn("Use ONLY the facts", messages[0]["content"])
        self.assertIn("FACTS:", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "WHAT IS HAPPENING:")

    def test_offline_guard_still_permits_the_loopback_endpoint(self):
        """The offline policy must not block the loopback endpoint."""
        from app import offline_guard

        offline_guard.install()
        with _MockEndpoint():
            self.assertTrue(_client().info()["usable"])


if __name__ == "__main__":
    unittest.main()
