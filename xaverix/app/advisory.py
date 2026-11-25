"""Advisory orchestration: knowledge base, Llama, guard, provenance.

The contract here is narrow and deliberate. Whatever happens, this module
returns an advisory with the same shape and an honest ``source`` label:

    llama_grounded              Llama wrote it and the guard accepted it
    kb_fallback_guard_reject    Llama wrote it, the guard rejected it
    kb_fallback_llm_unavailable ExecuServe is not running
    kb_fallback_llm_timeout     it ran but did not finish in budget
    kb_fallback_llm_cold        the model is still loading
    kb_only_llm_disabled        Llama is switched off

The fallback is never silent: the label and the guard's reasons travel in the
response and are rendered in the interface, so a rejection is visible rather
than substituted without notice.
"""
from __future__ import annotations

import time

from app import config, guard, i18n, knowledge, llm, prompt


def _render_section_list(items, bullet="- "):
    return "\n".join(bullet + item for item in items)


def render_kb_advisory(entry, crop, region_key, severity, band, language="en"):
    """Deterministic advisory composed directly from the knowledge base.

    Always available, cannot introduce unsupported content, and emits the same
    four sections as the generated path so the interface renders either
    identically.
    """
    sections = []

    sections.append(
        {
            "key": "what",
            "heading": i18n.t("section_what", language),
            "body": entry["what_is_happening_en"],
        }
    )

    urgency = knowledge.urgency_for(entry, severity)
    now_lines = [urgency] if urgency else []
    for item in (entry.get("chemical_treatment") or [])[:1]:
        line = "%s: %s, %s" % (item["input"], item["dose"], item["interval"])
        if item.get("phi_days") is not None:
            line += ". Do not harvest for %d days after spraying." % item["phi_days"]
        now_lines.append(line)
    if not now_lines:
        now_lines = list((entry.get("prevention") or [])[:1])
    sections.append(
        {
            "key": "now",
            "heading": i18n.t("section_now", language),
            "body": "\n".join(now_lines),
        }
    )

    organic_lines = []
    for item in (entry.get("organic_treatment") or [])[:2]:
        line = "%s: %s, %s" % (item["input"], item["dose"], item["interval"])
        if item.get("note"):
            line += " (%s)" % item["note"]
        organic_lines.append(line)
    sections.append(
        {
            "key": "organic",
            "heading": i18n.t("section_organic", language),
            "body": "\n".join(organic_lines) if organic_lines else "-",
        }
    )

    sections.append(
        {
            "key": "prevent",
            "heading": i18n.t("section_prevent", language),
            "body": _render_section_list((entry.get("prevention") or [])[:4]),
        }
    )

    return sections


def _parse_llama_sections(text, language="en"):
    """Split the four fixed headings out of a generated answer."""
    headings = [
        ("what", "WHAT IS HAPPENING:"),
        ("now", "WHAT TO DO NOW:"),
        ("organic", "ORGANIC OPTION:"),
        ("prevent", "PREVENT NEXT SEASON:"),
    ]
    upper = text.upper()
    positions = []
    for key, marker in headings:
        index = upper.find(marker)
        positions.append((key, marker, index))

    sections = []
    for order, (key, marker, index) in enumerate(positions):
        if index < 0:
            continue
        start = index + len(marker)
        end = len(text)
        for _k, _m, later in positions[order + 1 :]:
            if later > index:
                end = later
                break
        body = text[start:end].strip()
        label_key = {
            "what": "section_what",
            "now": "section_now",
            "organic": "section_organic",
            "prevent": "section_prevent",
        }[key]
        sections.append(
            {"key": key, "heading": i18n.t(label_key, language), "body": body}
        )
    return sections


def _model_block(client, retries, elapsed_ms):
    return {
        "name": client.model,
        "quantisation": "INT4",
        "runtime": "ExecuTorch (XNNPACK) on Arm CPU",
        "host": "ExecuServe at %s" % client.base,
        "retries": retries,
        "latency_ms": round(elapsed_ms),
    }


def build(entry, crop, region_key, severity, band, confidence_pct, language="en",
          client=None, trace=None):
    """Produce the advisory for one analysis.

    ``trace`` is an optional list that receives a stage record, so the
    interface can report what the generation stage did.
    """
    language = i18n.normalise(language)
    client = client or llm.client()

    kb_sections = render_kb_advisory(entry, crop, region_key, severity, band, language)
    base = {
        "sections": kb_sections,
        "llama_note": None,
        "guard": None,
        "model": None,
        "kb": {"entry_id": entry["id"], "kb_version": knowledge.load()["kb_version"]},
        "lang_source": "human_authored",
        "language": language,
    }

    def finish(source, detail=None):
        base["source"] = source
        base["source_label"] = i18n.source_label(source, language)
        base["source_detail"] = detail
        if trace is not None:
            trace.append(
                {
                    "stage": "llm_advisory",
                    "source": source,
                    "detail": detail or "",
                    "guard": base["guard"],
                }
            )
        return base

    if not client.enabled:
        return finish("kb_only_llm_disabled", "XAVERIX_LLM is not enabled")

    # An abstention has no diagnosis to restate, so generation is skipped
    # rather than spending device CPU on an image that was not assessed.
    if entry["verdict"] == "uncertain":
        return finish(
            "kb_only_llm_disabled",
            "no verdict to restate; generation is skipped when the "
            "classifier abstains",
        )

    info = client.info()
    if not info.get("usable"):
        reason = info.get("reason") or "not_running"
        source = (
            "kb_fallback_llm_unavailable"
            if reason != "not_loopback"
            else "kb_fallback_llm_unavailable"
        )
        return finish(source, "endpoint %s: %s" % (client.base, reason))

    if not client.is_warm():
        client.warm_up_async()
        if not client.wait_warm(config.LLM_COLD_WAIT):
            return finish(
                "kb_fallback_llm_cold",
                "model still loading after %.0f s; it stays loading in the "
                "background, so the next analysis should use Llama"
                % config.LLM_COLD_WAIT,
            )

    context = {
        "crop": crop,
        "region": region_key,
        "region_label": knowledge.region_label(region_key, "en"),
        "severity": severity,
        "band": band,
        "confidence_pct": confidence_pct,
    }

    messages = prompt.build_messages(
        entry, crop, region_key, severity, band, confidence_pct, language
    )

    started = time.perf_counter()
    retries = 0
    verdict = None
    text = None

    try:
        text, elapsed_ms = client.chat(messages)
    except llm.LLMUnavailable as exc:
        source = (
            "kb_fallback_llm_timeout"
            if exc.reason == "timeout"
            else "kb_fallback_llm_unavailable"
        )
        return finish(source, "%s: %s" % (exc.reason, exc.detail))

    # The assistant turn was prefilled with the first heading, so the model
    # continues from it rather than repeating it.
    if not text.upper().startswith("WHAT IS HAPPENING"):
        text = prompt.PREFILL + " " + text

    verdict = guard.check(text, entry, context)
    base["guard"] = verdict

    if not verdict["ok"]:
        spent_ms = (time.perf_counter() - started) * 1000.0
        budget_left = config.LLM_BUDGET_MS - spent_ms
        if budget_left > 15_000:
            retries = 1
            try:
                retry_text, _ = client.chat(
                    prompt.retry_messages(messages, verdict["codes"]),
                    max_tokens=160,
                    temperature=0.0,
                )
                if not retry_text.upper().startswith("WHAT IS HAPPENING"):
                    retry_text = prompt.PREFILL + " " + retry_text
                retry_verdict = guard.check(retry_text, entry, context)
                base["guard"] = retry_verdict
                if retry_verdict["ok"]:
                    text, verdict = retry_text, retry_verdict
            except llm.LLMUnavailable as exc:
                base["guard"]["warnings"] = list(
                    base["guard"].get("warnings") or []
                ) + ["retry failed: %s" % exc.reason]

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    base["model"] = _model_block(client, retries, elapsed_ms)

    if base["guard"]["ok"]:
        base["sections"] = _parse_llama_sections(text, language) or kb_sections
        base["llama_note"] = text
        base["lang_source"] = "llama_generated_en"
        if language != "en":
            # Generation is English-only; the structured advisory below it
            # is rendered in the selected language and labelled accordingly.
            base["sections"] = kb_sections
            base["llama_note_label"] = i18n.t("personalised_note", language)
        return finish("llama_grounded")

    return finish(
        "kb_fallback_guard_reject",
        "guard rejected the generated text: %s" % ", ".join(base["guard"]["codes"]),
    )
