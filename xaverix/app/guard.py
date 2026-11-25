"""Grounding guard: verify generated advisory text against the knowledge base.

Rationale
---------
A 1-billion-parameter instruction-tuned model is not a reliable source of
agronomic fact. During evaluation, asked what early blight of tomato is, the
model attributed it to a non-existent bacterial genus; the actual cause is the
fungus Alternaria solani. Acting on such an answer would mean applying the
wrong class of product. No generated text is therefore displayed without
passing the checks in this module.

Structure
---------
* **Hard checks** provide the safety guarantee: any single failure rejects the
  output. They target actionable content -- numeric quantities, agro-input
  names, organisms and pathogen classes -- because those are what a user would
  act and spend money on.
* **Soft scores** detect wholesale topic drift only. Their thresholds are
  intentionally permissive, since tightening them would reject ordinary
  paraphrase without improving safety.

Scope: the guard bounds actionable error, not all error. Fluent but
uninformative text that invents no quantity and names no product can pass.
"""
from __future__ import annotations

import re
import unicodedata

from app import config, knowledge

SECTIONS = (
    "WHAT IS HAPPENING:",
    "WHAT TO DO NOW:",
    "ORGANIC OPTION:",
    "PREVENT NEXT SEASON:",
)

# A kingdom word followed by a capitalised name, as in "the bacterium X".
# Whatever is named there must be a pathogen the knowledge base lists.
NAMED_CAUSE_RE = re.compile(
    r"\b(bacterium|bacteria|bacterial|fungus|fungi|fungal|oomycete|virus|viral|"
    r"pathogen|mite|nematode|insect)\s+(?:called\s+|named\s+|known as\s+)?"
    r"([A-Z][a-z]{3,}(?:\s+[a-z]{4,})?)",
    re.IGNORECASE,
)

# Genus-shaped single words: an invented organism usually still looks like one.
GENUS_SUFFIX_RE = re.compile(
    r"\b[A-Z][a-z]{2,}(?:bacter|bacterium|monas|coccus|myces|mycete|spora|"
    r"sporium|phthora|phora|cladium|toria|ella|opsis|fusarium)\b"
)

# Invented-but-plausible pesticide names: catch by morphology, not just by list.
CHEM_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}(?:conazole|strobin|mectin|cloprid|thrin|carb|phos|mycin|zeb|"
    r"oxychloride|sulfan|thiuron|diamide|prid|xacarb|fenthion|azole)\b",
    re.IGNORECASE,
)

BANNED_PATTERNS = (
    ("absolute_claim", r"\b100\s*%"),
    ("guarantee", r"\bguarantee|\bguaranteed\b"),
    ("will_cure", r"\bwill (?:cure|definitely|surely)\b|\bcompletely eliminat"),
    ("consumption_advice", r"\bsafe to (?:eat|drink|consume)\b"),
    ("ai_self_reference", r"\bas an AI\b|\bI am an AI\b|\blanguage model\b"),
    ("link", r"https?://|\bwww\."),
    ("phone_number", r"\b[6-9]\d{9}\b"),
    ("price_or_scheme", r"₹|\brs\.?\s*\d|\brupees\b|\bsubsidy\b|\bgovernment scheme\b"),
    ("hedging", r"\bI (?:think|believe|guess)\b"),
    ("cloud_reference", r"\bcloud\b|\binternet\b|\bonline database\b"),
    ("fabricated_referral", r"\bconsult (?:our|the) (?:website|app)\b"),
)

SPELLED_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15",
    "twenty": "20", "thirty": "30", "fifty": "50", "hundred": "100",
}

# Generic agricultural vocabulary, so ordinary paraphrase is not scored as
# unsupported content by the soft overlap measure.
AGRI_WHITELIST = {
    "leaf", "leaves", "plant", "plants", "crop", "crops", "field", "soil",
    "water", "watering", "irrigation", "spray", "sprays", "spraying", "apply",
    "applying", "mix", "mixing", "litre", "litres", "water", "morning",
    "evening", "day", "days", "week", "weeks", "season", "seasons", "remove",
    "removing", "burn", "destroy", "check", "checking", "look", "watch",
    "healthy", "damage", "damaged", "affected", "spread", "spreading", "stop",
    "prevent", "protect", "treat", "treatment", "disease", "pest", "pests",
    "spot", "spots", "patch", "patches", "yellow", "brown", "green", "grey",
    "white", "black", "underside", "surface", "edge", "edges", "margin",
    "lower", "upper", "older", "young", "new", "farmer", "officer",
    "extension", "advice", "today", "tomorrow", "now", "soon", "early",
    "late", "again", "first", "next", "between", "around", "near", "after",
    "before", "cover", "wet", "dry", "humid", "rain", "weather", "air",
    "space", "spacing", "row", "rows", "tool", "tools", "hand", "hands",
    "wash", "clean", "need", "needed", "use", "using", "good", "better",
    "small", "large", "more", "less", "most", "many", "some", "other",
    "harvest", "yield", "fruit", "flower", "stem", "root", "vein", "veins",
    "insect", "insects", "scout", "scouting", "trap", "traps", "manure",
    "compost", "mulch", "mulching", "rotate", "rotation", "drip", "basin",
    "overhead", "splash", "trust", "ask", "visit", "help", "sure", "see",
    "sign", "signs", "show", "shows", "shown", "take", "keep", "make",
    "give", "find", "start", "started", "starting", "continue", "repeat",
}

_TAMIL_ZERO = 0x0BE6
_DEVANAGARI_ZERO = 0x0966


def normalise_digits(text):
    """Map Devanagari and Tamil digits to ASCII so the number check survives
    a localised output."""
    out = []
    for ch in text:
        code = ord(ch)
        if _DEVANAGARI_ZERO <= code <= _DEVANAGARI_ZERO + 9:
            out.append(str(code - _DEVANAGARI_ZERO))
        elif _TAMIL_ZERO <= code <= _TAMIL_ZERO + 9:
            out.append(str(code - _TAMIL_ZERO))
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def _strip_list_markers(text):
    """Remove '1.' / '2)' list numbering, which is formatting, not a claim."""
    return re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", "", text)


def _numbers_in(text):
    text = _strip_list_markers(normalise_digits(text))
    for word, digit in SPELLED_NUMBERS.items():
        text = re.sub(r"\b%s\b" % word, digit, text, flags=re.IGNORECASE)
    # Treat "3-4" as two separate numbers rather than one odd token.
    text = text.replace("-", " ").replace("/", " ")
    return set(knowledge.NUMBER_RE.findall(text))


def _normalise_number(value):
    """'2.0' and '2' are the same dose; compare numerically where possible."""
    try:
        return "%g" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _content_words(text):
    tokens, bigrams = knowledge.tokenise(text)
    return tokens, bigrams


def _kingdom_words_present(text):
    lowered = text.lower()
    found = set()
    for kingdom, words in knowledge.KINGDOM_WORDS.items():
        for word in words:
            if re.search(r"\b%s\b" % re.escape(word), lowered):
                found.add(kingdom)
                break
    return found


def _check_format(text, violations):
    present = [s for s in SECTIONS if s in text.upper()]
    if len(present) != len(SECTIONS):
        missing = [s for s in SECTIONS if s not in text.upper()]
        violations.append(
            {"code": "FORMAT", "detail": "missing sections: %s" % ", ".join(missing)}
        )
        return
    positions = [text.upper().index(s) for s in SECTIONS]
    if positions != sorted(positions):
        violations.append({"code": "FORMAT", "detail": "sections out of order"})


def _check_length(text, word_count, violations):
    if word_count < config.GUARD_MIN_WORDS:
        violations.append(
            {"code": "LENGTH", "detail": "only %d content words" % word_count}
        )
    elif word_count > config.GUARD_MAX_WORDS:
        violations.append(
            {"code": "LENGTH", "detail": "%d content words" % word_count}
        )
    if len(text) > config.GUARD_MAX_CHARS:
        violations.append({"code": "LENGTH", "detail": "%d characters" % len(text)})


def _check_numbers(text, entry, ctx, violations):
    allowed = {_normalise_number(n) for n in entry["_numbers"]}
    for key in ("severity", "confidence_pct", "phi_days"):
        if ctx.get(key) is not None:
            allowed.add(_normalise_number(round(float(ctx[key]))))
            allowed.add(_normalise_number(ctx[key]))
    for raw in _numbers_in(text):
        if _normalise_number(raw) not in allowed:
            violations.append(
                {
                    "code": "NUMBER_NOT_IN_KB",
                    "detail": "%s is not a value from the knowledge base" % raw,
                }
            )


def _check_chemicals(text, entry, violations):
    lowered = text.lower()
    allowed = entry["_inputs"]
    named = set()

    for name in knowledge.chem_lexicon():
        if re.search(r"\b%s\b" % re.escape(name), lowered):
            named.add(name)
    for match in CHEM_SUFFIX_RE.findall(lowered):
        named.add(match.lower())

    for name in sorted(named):
        if any(name in allowed_name for allowed_name in allowed):
            continue
        if any(name in part for allowed_name in allowed for part in [allowed_name]):
            continue
        violations.append(
            {
                "code": "CHEMICAL_NOT_IN_KB",
                "detail": "%r is not an input listed for this diagnosis" % name,
            }
        )


def _looks_like_english(word):
    """True for ordinary prose words, so a capitalised sentence start such as
    'This will' or 'Your leaf' is not mistaken for a Latin binomial."""
    lowered = word.lower()
    return (
        lowered in knowledge._STOPWORDS
        or lowered in AGRI_WHITELIST
        or lowered in SPELLED_NUMBERS
        or lowered in {"spray", "your", "this", "these", "those", "remove",
                       "early", "late", "powdery", "downy", "bacterial",
                       "nutrient", "sucking", "chewing", "severity", "xaverix"}
    )


def _check_organisms(text, entry, violations):
    allowed = entry["_pathogens"]
    flagged = set()

    def flag(name, why):
        key = name.lower()
        if key in allowed or key in flagged:
            return
        flagged.add(key)
        violations.append({"code": "ORGANISM_NOT_IN_KB", "detail": why % name})

    # 1. A kingdom word followed by a capitalised name, which is how a
    #    fabricated causal agent is usually phrased.
    for kingdom, name in NAMED_CAUSE_RE.findall(text):
        name = name.strip()
        if _looks_like_english(name.split()[0]):
            continue
        flag(name, "%r is named as the cause but is not in the knowledge base")

    # 2. Genus-shaped single words, caught by morphology.
    for match in GENUS_SUFFIX_RE.findall(text):
        if _looks_like_english(match):
            continue
        flag(match, "%r is not an organism named in the knowledge base")

    # 3. Latin binomials.
    for match in knowledge.BINOMIAL_RE.findall(text):
        first, second = match.split()[0], match.split()[1]
        if _looks_like_english(first) or _looks_like_english(second):
            continue
        flag(match, "%r is not an organism named in the knowledge base")


def _check_kingdoms(text, entry, violations):
    allowed = entry["_kingdoms"]
    present = _kingdom_words_present(text)
    for kingdom in sorted(present - allowed):
        violations.append(
            {
                "code": "PATHOGEN_TYPE_MISMATCH",
                "detail": "output calls the cause a %s, which this diagnosis does not support"
                % kingdom,
            }
        )

    # Sharper check: a named pathogen paired with the wrong kingdom word nearby.
    lowered = text.lower()
    for pathogen, kingdom in (entry.get("_pathogen_kingdom") or {}).items():
        for match in re.finditer(re.escape(pathogen), lowered):
            window = lowered[max(0, match.start() - 80) : match.start()]
            for other, words in knowledge.KINGDOM_WORDS.items():
                if other == kingdom:
                    continue
                if any(re.search(r"\b%s\b" % re.escape(w), window) for w in words):
                    violations.append(
                        {
                            "code": "PATHOGEN_TYPE_MISMATCH",
                            "detail": "%s is a %s, but the output calls it a %s"
                            % (pathogen, kingdom, other),
                        }
                    )
                    break


def _check_banned(text, violations):
    for code, pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append({"code": "BANNED_PATTERN", "detail": code})


def _check_scope(text, entry, ctx, violations):
    lowered = text.lower()

    # Naming a different crop than the farmer selected.
    for crop in knowledge.crop_keys():
        if crop in ("other",) or crop == ctx.get("crop"):
            continue
        if re.search(r"\b%s\b" % re.escape(crop), lowered):
            violations.append(
                {"code": "WRONG_CROP", "detail": "names %r, not the selected crop" % crop}
            )

    # Naming a state outside the selected region.
    for state in knowledge.other_state_names(ctx.get("region") or "default"):
        if re.search(r"\b%s\b" % re.escape(state), lowered):
            violations.append(
                {"code": "WRONG_REGION", "detail": "names %r" % state}
            )
            break

    # Must actually give an action, not just sympathy.
    if entry["_inputs"]:
        if not any(name.split()[0] in lowered for name in entry["_inputs"]):
            violations.append(
                {
                    "code": "MISSING_ACTION",
                    "detail": "names none of the knowledge-base treatment inputs",
                }
            )


def check(output, entry, ctx=None):
    """Verify one generated output against one knowledge-base entry.

    Returns the decision, the soft scores and every violation, so a rejection
    can be reported rather than silently substituted.
    """
    ctx = ctx or {}
    text = (output or "").strip()
    violations = []

    if not text:
        return {
            "ok": False,
            "decision": "reject",
            "support": 0.0,
            "bigram_overlap": 0.0,
            "words": 0,
            "violations": [{"code": "EMPTY", "detail": "no output"}],
            "warnings": [],
            "codes": ["EMPTY"],
        }

    tokens, bigrams = _content_words(text)
    word_count = len(re.findall(r"[A-Za-z]+", text))

    _check_format(text, violations)
    _check_length(text, word_count, violations)
    _check_numbers(text, entry, ctx, violations)
    _check_chemicals(text, entry, violations)
    _check_organisms(text, entry, violations)
    _check_kingdoms(text, entry, violations)
    _check_banned(text, violations)
    _check_scope(text, entry, ctx, violations)

    # ---- soft scores -------------------------------------------------
    context_tokens = set()
    for value in (ctx.get("crop"), ctx.get("region_label"), ctx.get("band")):
        if value:
            context_tokens |= knowledge.tokenise(str(value))[0]
    reference = entry["_ref_tokens"] | context_tokens | AGRI_WHITELIST

    support = len(tokens & reference) / len(tokens) if tokens else 0.0
    overlap = (
        len(bigrams & entry["_ref_bigrams"]) / len(bigrams) if bigrams else 0.0
    )

    if support < config.GUARD_SUPPORT_MIN:
        violations.append(
            {"code": "LOW_SUPPORT", "detail": "token support %.2f" % support}
        )
    if overlap < config.GUARD_BIGRAM_MIN:
        violations.append(
            {"code": "LOW_OVERLAP", "detail": "bigram overlap %.2f" % overlap}
        )

    warnings = []
    if config.GUARD_BIGRAM_MIN <= overlap < config.GUARD_BIGRAM_WARN:
        warnings.append("bigram overlap %.2f is low but acceptable" % overlap)

    ok = not violations
    return {
        "ok": ok,
        "decision": "accept" if ok else "reject",
        "support": round(support, 3),
        "bigram_overlap": round(overlap, 3),
        "words": word_count,
        "violations": violations,
        "warnings": warnings,
        "codes": sorted({v["code"] for v in violations}),
    }


# ------------------------------------------------------------------ demo
#: Fixed invalid outputs used by /api/guard-demo, so the guard can be
#: exercised deterministically. Each case violates a different rule.
DEMO_CASES = (
    {
        "name": "fabricated pathogen and invented dose",
        "text": (
            "WHAT IS HAPPENING:\n"
            "Early blight of tomato is caused by the bacterium Colletobacter, "
            "which spreads through the air.\n"
            "WHAT TO DO NOW:\n"
            "Spray azoxystrobin 25 SC at 1.5 ml per litre immediately.\n"
            "ORGANIC OPTION:\n"
            "Neem oil at 12 ml per litre every 3 days.\n"
            "PREVENT NEXT SEASON:\n"
            "This will completely eliminate the disease, guaranteed."
        ),
    },
    {
        "name": "invented dose for a real product",
        "text": (
            "WHAT IS HAPPENING:\n"
            "The leaf shows brown spots with rings, a sign of disease.\n"
            "WHAT TO DO NOW:\n"
            "Spray mancozeb 75 WP at 9 g per litre of water every 2 days.\n"
            "ORGANIC OPTION:\n"
            "Use neem oil at 5 ml per litre of water every 7 days.\n"
            "PREVENT NEXT SEASON:\n"
            "Remove and burn the affected lower leaves and rotate the crop."
        ),
    },
    {
        "name": "fabricated referral and price",
        "text": (
            "WHAT IS HAPPENING:\n"
            "Your leaf is damaged by disease and needs urgent care.\n"
            "WHAT TO DO NOW:\n"
            "Buy the treatment for Rs 450 and consult our website for details.\n"
            "ORGANIC OPTION:\n"
            "Spray neem oil at 5 ml per litre of water every 7 days.\n"
            "PREVENT NEXT SEASON:\n"
            "Remove the affected leaves and use drip irrigation."
        ),
    },
)


def demo(ctx=None):
    """Run the guard over the fixed invalid outputs; used by /api/guard-demo."""
    entry = knowledge.entry_for("infected")
    ctx = ctx or {"crop": "tomato", "region": "tamil_nadu", "severity": 38}
    results = []
    for case in DEMO_CASES:
        verdict = check(case["text"], entry, ctx)
        results.append(
            {
                "case": case["name"],
                "text": case["text"],
                "decision": verdict["decision"],
                "codes": verdict["codes"],
                "violations": verdict["violations"],
                "support": verdict["support"],
            }
        )
    return {
        "note": (
            "Each case below is an invalid output the guard must reject, drawn "
            "from classes of error observed during model evaluation."
        ),
        "all_rejected": all(r["decision"] == "reject" for r in results),
        "cases": results,
    }
