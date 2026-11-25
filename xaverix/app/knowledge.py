"""Knowledge-base loading, validation and derivation of the guard's allowed sets.

The knowledge base is the single source of factual truth in Xaverix. Llama is
never trusted to supply a fact; it is only allowed to rephrase what is in here.

The sets the grounding guard checks against (allowed numbers, allowed input
names, allowed pathogens, allowed kingdoms) are *derived* from each entry's own
strings at load time rather than hand-authored. Hand-authored allow-lists drift
out of step with the prose they are supposed to describe; derived ones cannot.
"""
from __future__ import annotations

import re

from app import config

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# Latin binomial shape, e.g. "Alternaria solani".
BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]{3,}\s+[a-z]{4,}\b")

KINGDOM_WORDS = {
    "fungus": ("fungus", "fungal", "fungi"),
    "oomycete": ("oomycete", "water mould", "water mold"),
    "bacterium": ("bacterium", "bacterial", "bacteria"),
    "virus": ("virus", "viral"),
    "insect": ("insect", "pest", "larva", "larvae", "caterpillar", "aphid", "whitefly"),
    "mite": ("mite", "acarine"),
    "nematode": ("nematode", "eelworm"),
    "abiotic": ("deficiency", "nutrient", "abiotic"),
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by",
    "can", "do", "does", "for", "from", "has", "have", "if", "in", "into", "is",
    "it", "its", "no", "not", "of", "on", "once", "only", "or", "per", "so",
    "than", "that", "the", "their", "then", "there", "these", "they", "this",
    "to", "up", "was", "were", "when", "which", "while", "will", "with",
    "within", "you", "your",
}

_cache = {}


def _walk_strings(node):
    """Yield every string anywhere inside a nested dict/list structure."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_strings(value)


def tokenise(text):
    """Lowercase content words, plus their bigrams."""
    words = re.findall(r"[a-z]+", text.lower())
    content = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    bigrams = {" ".join(pair) for pair in zip(content, content[1:])}
    return set(content), bigrams


def _treatment_inputs(entry):
    names = set()
    for key in ("organic_treatment", "chemical_treatment"):
        for item in entry.get(key) or []:
            if item.get("input"):
                names.add(item["input"].lower())
    for cause in entry.get("probable_causes") or []:
        for key in ("organic", "chemical"):
            item = cause.get(key) or {}
            if item.get("input"):
                names.add(item["input"].lower())
    return names


def _pathogens(entry):
    names = set()
    if entry.get("pathogen"):
        names.add(entry["pathogen"].lower())
    for cause in entry.get("probable_causes") or []:
        if cause.get("pathogen"):
            names.add(cause["pathogen"].lower())
    # "Bemisia tabaci and Aphis gossypii" also licenses each binomial alone,
    # otherwise the guard would reject an advisory that names only one of them.
    for name in list(names):
        for part in re.split(r"\s+and\s+|,\s*", name):
            part = part.strip()
            if len(part.split()) == 2:
                names.add(part)
    return names


def _kingdoms(entry):
    kinds = set()
    if entry.get("pathogen_type"):
        kinds.add(entry["pathogen_type"].lower())
    for cause in entry.get("probable_causes") or []:
        if cause.get("pathogen_type"):
            kinds.add(cause["pathogen_type"].lower())
    return kinds


def _derive(entry):
    text = " ".join(_walk_strings(entry))
    tokens, bigrams = tokenise(text)
    entry["_numbers"] = set(NUMBER_RE.findall(text))
    entry["_inputs"] = _treatment_inputs(entry)
    entry["_pathogens"] = _pathogens(entry)
    entry["_kingdoms"] = _kingdoms(entry)
    entry["_ref_tokens"] = tokens
    entry["_ref_bigrams"] = bigrams
    # Pair each pathogen with its own kingdom, so "the bacterium Alternaria
    # solani" can be caught even though both words appear somewhere in the KB.
    pairs = {}
    if entry.get("pathogen") and entry.get("pathogen_type"):
        pairs[entry["pathogen"].lower()] = entry["pathogen_type"].lower()
    for cause in entry.get("probable_causes") or []:
        if cause.get("pathogen") and cause.get("pathogen_type"):
            pairs[cause["pathogen"].lower()] = cause["pathogen_type"].lower()
    for name in list(pairs):
        for part in re.split(r"\s+and\s+|,\s*", name):
            part = part.strip()
            if len(part.split()) == 2:
                pairs.setdefault(part, pairs[name])
    entry["_pathogen_kingdom"] = pairs
    return entry


class ValidationError(Exception):
    """Raised at startup when the knowledge base is internally inconsistent."""


def _validate(kb):
    entries = kb.get("entries") or {}
    for required in ("healthy", "infected", "uncertain"):
        if required not in entries:
            raise ValidationError("knowledge base is missing the %r entry" % required)

    for key, entry in entries.items():
        if entry.get("id") != key:
            raise ValidationError("entry %r has mismatched id %r" % (key, entry.get("id")))
        for field in ("verdict", "name_en", "headline_en", "what_is_happening_en"):
            if not entry.get(field):
                raise ValidationError("entry %r is missing %r" % (key, field))
        if not entry.get("prevention"):
            raise ValidationError("entry %r has no prevention guidance" % key)
        urgency = entry.get("urgency") or {}
        for band in ("trace", "mild", "moderate", "severe"):
            if not urgency.get(band):
                raise ValidationError("entry %r has no %r urgency line" % (key, band))

    infected = entries["infected"]
    if not infected.get("probable_causes"):
        raise ValidationError("the infected entry must list probable causes")
    for cause in infected["probable_causes"]:
        for field in ("name", "distinguishing_symptom", "organic", "chemical"):
            if not cause.get(field):
                raise ValidationError(
                    "probable cause %r is missing %r" % (cause.get("name"), field)
                )
    if not infected.get("chemical_treatment") or not infected.get("organic_treatment"):
        raise ValidationError("the infected entry needs organic and chemical options")
    return kb


def load(force=False):
    """Load, validate and derive. Cached, because this is read on every request."""
    if not force and "kb" in _cache:
        return _cache["kb"]
    kb = config.load_json(config.KB_PATH)
    _validate(kb)
    for entry in kb["entries"].values():
        _derive(entry)
    _cache["kb"] = kb
    return kb


def entry_for(verdict):
    """Map a classifier verdict onto its knowledge-base entry."""
    entries = load()["entries"]
    return entries.get(verdict, entries["uncertain"])


def chem_lexicon(force=False):
    """Flat set of agro-input names the guard watches for."""
    if not force and "chem" in _cache:
        return _cache["chem"]
    raw = config.load_json(config.CHEM_LEXICON_PATH)
    names = set()
    for key, value in raw.items():
        if key == "note" or not isinstance(value, list):
            continue
        names.update(name.lower() for name in value)
    _cache["chem"] = names
    return names


def regions(force=False):
    if not force and "regions" in _cache:
        return _cache["regions"]
    _cache["regions"] = config.load_json(config.REGIONS_PATH)
    return _cache["regions"]


def region_keys():
    return [r["key"] for r in regions()["regions"]]


def crop_keys():
    return [c["key"] for c in regions()["crops"]]


def region_label(key, language="en"):
    for region in regions()["regions"]:
        if region["key"] == key:
            return region.get("label_%s" % language) or region["label_en"]
    return key


def crop_label(key, language="en"):
    for crop in regions()["crops"]:
        if crop["key"] == key:
            return crop.get("label_%s" % language) or crop["label_en"]
    return key


def region_note(entry, region_key):
    notes = entry.get("region_notes") or {}
    return notes.get(region_key) or notes.get("default") or ""


def severity_band(severity):
    """Shared by severity.py, the advisory and the UI so the bands never diverge."""
    if severity < 10:
        return "trace"
    if severity < 25:
        return "mild"
    if severity < 50:
        return "moderate"
    return "severe"


def urgency_for(entry, severity):
    return (entry.get("urgency") or {}).get(severity_band(severity), "")


def other_state_names(region_key):
    """Indian states that do NOT belong to the selected region.

    The guard uses this to catch an advisory that drifts to the wrong place.
    """
    data = regions()
    own = set()
    for region in data["regions"]:
        if region["key"] == region_key:
            own = {s.lower() for s in region.get("states") or []}
            break
    return {s.lower() for s in data["all_indian_states"]} - own
