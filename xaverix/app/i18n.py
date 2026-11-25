"""Language rendering for English, Hindi and Tamil.

Generation is confined to the English layer, for three reasons:

1. Hindi and Tamil output from a 1B INT4 model is weak, and an incorrect dose
   in a translated string is a safety defect rather than a wording defect.
2. Tamil costs roughly three to five times the tokens under this tokenizer, on
   a call that already takes tens of seconds on-device.
3. The grounding guard verifies output by token overlap against an English
   knowledge base and cannot verify a translation, so machine translation
   would bypass the check that makes generated text safe to display.

User-facing text is therefore rendered from human-authored strings, with the
generated paragraph shown as an additional, clearly labelled English note.
"""
from __future__ import annotations

from app import config

_cache = {}


def _table():
    if "i18n" not in _cache:
        _cache["i18n"] = config.load_json(config.I18N_PATH)
    return _cache["i18n"]


def normalise(language):
    language = (language or config.DEFAULT_LANGUAGE).strip().lower()
    return language if language in config.LANGUAGES else config.DEFAULT_LANGUAGE


def t(key, language="en", default=None):
    """Look up one string. Falls back to English, then to the key itself."""
    entry = _table()["strings"].get(key)
    if not entry:
        return default if default is not None else key
    language = normalise(language)
    return entry.get(language) or entry.get("en") or (default if default is not None else key)


def ui_strings(language="en"):
    """Every string at once, for the single-page app to render client side."""
    language = normalise(language)
    table = _table()["strings"]
    return {key: (entry.get(language) or entry.get("en")) for key, entry in table.items()}


def languages():
    return _table()["languages"]


def missing_keys():
    """Which strings lack a translation. Used by the test suite."""
    gaps = {}
    for key, entry in _table()["strings"].items():
        absent = [lang for lang in config.LANGUAGES if not entry.get(lang)]
        if absent:
            gaps[key] = absent
    return gaps


def verdict_label(verdict, language="en"):
    return t("verdict_%s" % verdict, language, default=verdict)


def band_label(band, language="en"):
    return t("band_%s" % band, language, default=band)


SOURCE_STRING = {
    "llama_grounded": "source_llama",
    "kb_fallback_guard_reject": "source_guard_reject",
    "kb_fallback_llm_unavailable": "source_unavailable",
    "kb_fallback_llm_timeout": "source_unavailable",
    "kb_fallback_llm_cold": "source_cold",
    "kb_only_llm_disabled": "source_disabled",
}


def source_label(source, language="en"):
    return t(SOURCE_STRING.get(source, "source_unavailable"), language)
