"""The Llama client, the advisory fallback chain, i18n, and the pipeline.

No test here requires ExecuServe to be running. The point is that every
failure degrades to the knowledge-base advisory with an honest label, so the
suite must pass on a laptop with nothing listening on port 8080.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    advisory,
    classifier,
    config,
    i18n,
    knowledge,
    llm,
    pipeline,
    prompt,
)
from tools import synth  # noqa: E402


class LLMClientTest(unittest.TestCase):
    def test_refuses_a_non_loopback_endpoint(self):
        for url in (
            "http://api.openai.com/v1",
            "https://example.com/v1",
            "http://8.8.8.8:8080/v1",
        ):
            client = llm.ExecuServeClient(url=url, key="x", enabled=True)
            self.assertFalse(client.loopback_only, url)
            with self.assertRaises(llm.LLMUnavailable) as caught:
                client.chat([{"role": "user", "content": "hi"}])
            self.assertEqual(caught.exception.reason, "not_loopback")

    def test_accepts_loopback_endpoints(self):
        for url in (
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://127.3.2.1:9000/v1",
        ):
            self.assertTrue(
                llm.ExecuServeClient(url=url, enabled=True).loopback_only, url
            )

    def test_disabled_client_raises_disabled(self):
        client = llm.ExecuServeClient(enabled=False)
        with self.assertRaises(llm.LLMUnavailable) as caught:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(caught.exception.reason, "disabled")

    def test_info_is_honest_when_nothing_is_listening(self):
        client = llm.ExecuServeClient(url="http://127.0.0.1:9", enabled=True)
        info = client.info()
        self.assertFalse(info["usable"])
        self.assertIn("reason", info)
        self.assertEqual(info["runtime"], "ExecuTorch (XNNPACK) via ExecuServe")

    def test_host_parsing(self):
        self.assertEqual(llm._host_of("http://127.0.0.1:8080/v1"), "127.0.0.1")
        self.assertEqual(llm._host_of("http://localhost/v1"), "localhost")
        self.assertEqual(llm._host_of("http://[::1]:8080/v1"), "::1")


class PromptTest(unittest.TestCase):
    def test_builds_three_messages_with_a_prefill(self):
        entry = knowledge.entry_for("infected")
        messages = prompt.build_messages(
            entry, "tomato", "tamil_nadu", 38.0, "moderate", 74.0
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], prompt.PREFILL)

    def test_facts_contain_only_knowledge_base_content(self):
        entry = knowledge.entry_for("infected")
        messages = prompt.build_messages(
            entry, "tomato", "tamil_nadu", 38.0, "moderate", 74.0
        )
        user = messages[1]["content"]
        self.assertIn("FACTS:", user)
        self.assertIn("Alternaria solani", user)
        self.assertIn("Tamil Nadu", user)

    def test_stays_within_a_sensible_prompt_budget(self):
        # Prefill tokens are not free on a phone CPU.
        entry = knowledge.entry_for("infected")
        messages = prompt.build_messages(
            entry, "tomato", "tamil_nadu", 38.0, "moderate", 74.0
        )
        total = sum(len(m["content"]) for m in messages)
        self.assertLess(total, 4000, "prompt is %d chars" % total)

    def test_retry_messages_name_the_violations(self):
        entry = knowledge.entry_for("infected")
        messages = prompt.build_messages(
            entry, "tomato", "tamil_nadu", 38.0, "moderate", 74.0
        )
        retried = prompt.retry_messages(messages, ["NUMBER_NOT_IN_KB"])
        self.assertIn("DO NOT", retried[0]["content"])
        self.assertEqual(retried[1], messages[1])


class AdvisoryFallbackTest(unittest.TestCase):
    def setUp(self):
        self.entry = knowledge.entry_for("infected")

    def test_kb_advisory_has_all_four_sections(self):
        sections = advisory.render_kb_advisory(
            self.entry, "tomato", "tamil_nadu", 38.0, "moderate", "en"
        )
        self.assertEqual([s["key"] for s in sections], ["what", "now", "organic", "prevent"])
        for section in sections:
            self.assertTrue(section["body"].strip(), section["key"])

    def test_falls_back_cleanly_when_llama_is_disabled(self):
        result = advisory.build(
            self.entry,
            "tomato",
            "tamil_nadu",
            38.0,
            "moderate",
            74.0,
            client=llm.ExecuServeClient(enabled=False),
        )
        self.assertEqual(result["source"], "kb_only_llm_disabled")
        self.assertTrue(result["sections"])
        self.assertIsNone(result["llama_note"])
        self.assertTrue(result["source_label"])

    def test_falls_back_cleanly_when_llama_is_unreachable(self):
        result = advisory.build(
            self.entry,
            "tomato",
            "tamil_nadu",
            38.0,
            "moderate",
            74.0,
            client=llm.ExecuServeClient(url="http://127.0.0.1:9", enabled=True),
        )
        self.assertEqual(result["source"], "kb_fallback_llm_unavailable")
        self.assertTrue(result["sections"])

    def test_llama_is_not_called_for_an_abstention(self):
        """Generation is skipped when there is no verdict to personalise."""
        trace = []
        result = advisory.build(
            knowledge.entry_for("uncertain"),
            "tomato",
            "tamil_nadu",
            0.0,
            "trace",
            None,
            client=llm.ExecuServeClient(url="http://127.0.0.1:8080/v1", enabled=True),
            trace=trace,
        )
        self.assertEqual(result["source"], "kb_only_llm_disabled")
        self.assertIn("abstains", result["source_detail"])

    def test_every_source_code_has_a_label(self):
        for source in i18n.SOURCE_STRING:
            for language in config.LANGUAGES:
                self.assertTrue(i18n.source_label(source, language), (source, language))

    def test_parses_llama_sections(self):
        text = (
            "WHAT IS HAPPENING:\nBrown patches.\nWHAT TO DO NOW:\nRemove leaves.\n"
            "ORGANIC OPTION:\nNeem oil.\nPREVENT NEXT SEASON:\nRotate crops."
        )
        sections = advisory._parse_llama_sections(text, "en")
        self.assertEqual(len(sections), 4)
        self.assertEqual(sections[0]["body"], "Brown patches.")
        self.assertEqual(sections[3]["body"], "Rotate crops.")


class I18nTest(unittest.TestCase):
    def test_every_string_exists_in_every_language(self):
        gaps = i18n.missing_keys()
        self.assertEqual(gaps, {}, "untranslated strings: %r" % gaps)

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(i18n.normalise("fr"), "en")
        self.assertEqual(i18n.normalise(None), "en")
        self.assertEqual(i18n.normalise("TA"), "ta")

    def test_verdict_labels_differ_between_languages(self):
        english = i18n.verdict_label("infected", "en")
        tamil = i18n.verdict_label("infected", "ta")
        self.assertTrue(english and tamil)
        self.assertNotEqual(english, tamil)

    def test_ui_strings_are_complete(self):
        for language in config.LANGUAGES:
            strings = i18n.ui_strings(language)
            self.assertIn("analyse", strings)
            self.assertTrue(all(strings.values()))


class KnowledgeTest(unittest.TestCase):
    def test_loads_and_validates(self):
        kb = knowledge.load()
        self.assertEqual(set(kb["entries"]), {"healthy", "infected", "uncertain"})

    def test_derived_sets_are_populated(self):
        infected = knowledge.entry_for("infected")
        self.assertTrue(infected["_numbers"])
        self.assertTrue(infected["_inputs"])
        self.assertIn("alternaria solani", infected["_pathogens"])
        self.assertIn("fungus", infected["_kingdoms"])

    def test_compound_pathogen_names_split(self):
        infected = knowledge.entry_for("infected")
        self.assertIn("bemisia tabaci", infected["_pathogens"])
        self.assertIn("aphis gossypii", infected["_pathogens"])

    def test_healthy_entry_licenses_no_pathogen(self):
        healthy = knowledge.entry_for("healthy")
        self.assertEqual(healthy["_pathogens"], set())
        self.assertEqual(healthy["_kingdoms"], set())

    def test_severity_bands(self):
        self.assertEqual(knowledge.severity_band(3), "trace")
        self.assertEqual(knowledge.severity_band(18), "mild")
        self.assertEqual(knowledge.severity_band(35), "moderate")
        self.assertEqual(knowledge.severity_band(80), "severe")

    def test_urgency_exists_for_every_band(self):
        for key in ("healthy", "infected", "uncertain"):
            entry = knowledge.entry_for(key)
            for value in (3, 18, 35, 80):
                self.assertTrue(knowledge.urgency_for(entry, value), (key, value))

    def test_other_state_names_excludes_the_selected_region(self):
        others = knowledge.other_state_names("tamil_nadu")
        self.assertNotIn("tamil nadu", others)
        self.assertIn("punjab", others)


class PipelineTest(unittest.TestCase):
    def test_rejects_undecodable_data(self):
        with self.assertRaises(pipeline.BadImage):
            pipeline.decode_image(b"")
        with self.assertRaises(pipeline.BadImage):
            pipeline.decode_image(b"this is not an image")

    def test_resizes_to_the_model_input(self):
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (640, 480), (30, 120, 40)).save(buffer, format="PNG")
        decoded = pipeline.decode_image(buffer.getvalue())
        self.assertEqual(decoded.shape, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

    @unittest.skipUnless(classifier.is_available(), "model not trained yet")
    def test_end_to_end_without_llama(self):
        image, _ = synth.render_leaf(True, np.random.default_rng(77), severity_hint=0.45)
        result = pipeline.analyse(image, use_llm=False)

        self.assertIn(result["verdict"], ("healthy", "infected", "uncertain"))
        self.assertTrue(result["verdict_label"])
        self.assertTrue(result["advisory"]["sections"])
        self.assertTrue(result["advisory"]["source_label"])
        self.assertIn("total", result["timings_ms"])

        stages = [entry["stage"] for entry in result["trace"]]
        for stage in ("features", "classifier", "severity", "knowledge", "llm_advisory"):
            self.assertIn(stage, stages)

    @unittest.skipUnless(classifier.is_available(), "model not trained yet")
    def test_result_is_json_serialisable(self):
        """NumPy scalars must not leak into the response."""
        import json

        image, _ = synth.render_leaf(False, np.random.default_rng(78))
        result = pipeline.analyse(image, use_llm=False)
        json.dumps(result)  # must not raise

    @unittest.skipUnless(classifier.is_available(), "model not trained yet")
    def test_severity_is_suppressed_when_abstaining(self):
        image = synth.render_negative(np.random.default_rng(2), kind="lined_paper")
        result = pipeline.analyse(image, use_llm=False)
        self.assertTrue(result["abstained"])
        self.assertTrue(result["severity"]["suppressed"])
        self.assertFalse(result["severity"]["available"])

    @unittest.skipUnless(classifier.is_available(), "model not trained yet")
    def test_language_choice_is_honoured(self):
        image, _ = synth.render_leaf(True, np.random.default_rng(79), severity_hint=0.4)
        tamil = pipeline.analyse(image, language="ta", use_llm=False)
        english = pipeline.analyse(image, language="en", use_llm=False)
        self.assertEqual(tamil["context"]["language"], "ta")
        self.assertNotEqual(tamil["verdict_label"], english["verdict_label"])


if __name__ == "__main__":
    unittest.main()
