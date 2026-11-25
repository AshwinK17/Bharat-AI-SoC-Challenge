"""Prompt construction for the on-device advisory.

The model is Llama 3.2 1B Instruct at INT4. Left to its own parametric
knowledge it fabricates pathogens and doses, so it is never asked what a
disease is. It receives the facts and is asked only to restate them for one
user, in a fixed structure the grounding guard can verify.

Two deliberate choices:

* The system message is restrictive: use only FACTS, introduce nothing new,
  four fixed headings, a word cap.
* The assistant turn is **prefilled** with the first heading. A model this
  size needs the output shape demonstrated, and a prefill costs far fewer
  tokens than a few-shot exemplar on an already slow on-device call.
"""
from __future__ import annotations

from app import knowledge

SYSTEM_PROMPT = """You are an agricultural extension assistant. You rewrite supplied facts for one farmer.

RULES, these override everything else:
1. Use ONLY the facts given in FACTS. Add no disease, no organism, no chemical, no dose and no number that is not written in FACTS.
2. Never state a cause or a treatment that is not written in FACTS.
3. No guarantees, no prices, no links, no phone numbers, no advice about eating the crop.
4. Plain simple English for a smallholder farmer. Short sentences.
5. Output EXACTLY these four headings, each on its own line, nothing before or after:
WHAT IS HAPPENING:
WHAT TO DO NOW:
ORGANIC OPTION:
PREVENT NEXT SEASON:
6. Maximum 110 words in total. One or two short lines under each heading.
7. If FACTS do not cover something, leave it out. Do not fill gaps from memory."""

PREFILL = "WHAT IS HAPPENING:"


def _fact_lines(entry, crop, region_key, severity, band):
    """Flatten the matched knowledge-base entry into a numbered fact list.

    Only facts relevant to this verdict and severity are included, which keeps
    the prompt near 400 tokens and keeps the guard's reference set tight.
    """
    lines = []

    lines.append("Observation: %s" % entry["what_is_happening_en"])

    for symptom in (entry.get("symptoms") or [])[:3]:
        lines.append("Symptom seen in this class: %s" % symptom)

    if entry["verdict"] == "infected":
        lines.append("Measured leaf damage: %d percent (%s)" % (round(severity), band))

    urgency = knowledge.urgency_for(entry, severity)
    if urgency:
        lines.append("Urgency for this severity: %s" % urgency)

    # Probable causes: names and distinguishing symptoms only. The farmer is
    # told to scout, not given a diagnosis the model cannot support.
    for cause in (entry.get("probable_causes") or [])[:3]:
        label = cause["name"]
        if cause.get("pathogen"):
            label += " (%s, a %s)" % (cause["pathogen"], cause["pathogen_type"])
        lines.append(
            "Possible cause to check: %s - %s" % (label, cause["distinguishing_symptom"])
        )

    for item in (entry.get("organic_treatment") or [])[:1]:
        text = "Organic option: %s at %s, %s" % (
            item["input"],
            item["dose"],
            item["interval"],
        )
        if item.get("note"):
            text += " (%s)" % item["note"]
        lines.append(text)

    for item in (entry.get("chemical_treatment") or [])[:1]:
        text = "Chemical option: %s at %s, %s" % (
            item["input"],
            item["dose"],
            item["interval"],
        )
        if item.get("phi_days") is not None:
            text += ", do not harvest for %d days after spraying" % item["phi_days"]
        lines.append(text)

    for step in (entry.get("prevention") or [])[:3]:
        lines.append("Prevention: %s" % step)

    note = knowledge.region_note(entry, region_key)
    if note:
        lines.append("Note for this region: %s" % note)

    return lines


def build_messages(entry, crop, region_key, severity, band, confidence_pct, language="en"):
    """Build the chat messages for one advisory.

    Returns the message list. The caller prefills the assistant turn.
    """
    crop_label = knowledge.crop_label(crop, "en")
    region_label = knowledge.region_label(region_key, "en")

    context = "CONTEXT: crop=%s | region=%s" % (crop_label, region_label)
    if entry["verdict"] == "infected":
        context += " | leaf damage=%d percent (%s)" % (round(severity), band)
    if confidence_pct is not None:
        context += " | detection confidence=%d percent" % round(confidence_pct)

    facts = "\n".join(
        "%d. %s" % (i, line)
        for i, line in enumerate(_fact_lines(entry, crop, region_key, severity, band), 1)
    )

    user = "%s\n\nFACTS:\n%s\n\nWrite the advisory now, following the four headings." % (
        context,
        facts,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        # Prefill: locks the output shape without spending tokens on an example.
        {"role": "assistant", "content": PREFILL},
    ]


def retry_messages(messages, violation_codes):
    """Rebuild the messages after a guard rejection, naming what went wrong.

    Only the system message changes, so the prompt stays cached-friendly and
    the facts are identical.
    """
    prohibitions = {
        "NUMBER_NOT_IN_KB": "do not write any number that is not in FACTS",
        "CHEMICAL_NOT_IN_KB": "do not name any product that is not in FACTS",
        "ORGANISM_NOT_IN_KB": "do not name any organism that is not in FACTS",
        "PATHOGEN_TYPE_MISMATCH": "do not say what kind of organism causes this unless FACTS says so",
        "BANNED_PATTERN": "no guarantees, no prices, no links, no website or app references",
        "FORMAT": "use exactly the four headings, each on its own line",
        "LENGTH": "stay under 110 words",
        "WRONG_CROP": "write about the crop in CONTEXT only",
        "WRONG_REGION": "do not name any other state or region",
        "MISSING_ACTION": "name at least one treatment from FACTS",
        "LOW_SUPPORT": "stay close to the wording of FACTS",
        "LOW_OVERLAP": "stay close to the wording of FACTS",
    }
    extra = sorted({prohibitions[c] for c in violation_codes if c in prohibitions})
    if not extra:
        extra = ["stay strictly within FACTS"]

    hardened = SYSTEM_PROMPT + "\n\nYour previous answer was rejected. DO NOT: " + "; ".join(
        extra
    ) + "."

    rebuilt = list(messages)
    rebuilt[0] = {"role": "system", "content": hardened}
    return rebuilt
