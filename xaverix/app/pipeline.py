"""The analysis pipeline, with a per-stage trace attached to every response.

The trace is part of the response contract rather than incidental debug
output: it records which stage produced which value, how long each stage
took, and the gate values behind an abstention.
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from app import (
    advisory,
    classifier,
    config,
    features,
    i18n,
    knowledge,
    severity as severity_module,
)


class BadImage(Exception):
    """The upload could not be decoded as an image."""


def decode_image(data):
    """Decode and square-resize to the model's input size."""
    if not data:
        raise BadImage("no image data received")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise BadImage(
            "image is %.1f MB; the limit is %.0f MB"
            % (len(data) / 1e6, config.MAX_UPLOAD_BYTES / 1e6)
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = image.convert("RGB")
    except Exception as exc:  # Pillow raises a wide variety here
        raise BadImage("could not decode the image: %s" % exc)

    if image.size != (config.IMAGE_SIZE, config.IMAGE_SIZE):
        image = image.resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
    return np.asarray(image, dtype=np.uint8)


class _Timer:
    """Collects per-stage wall-clock times into the trace."""

    def __init__(self, trace):
        self.trace = trace
        self.started = time.perf_counter()
        self.last = self.started

    def mark(self, stage, detail=""):
        now = time.perf_counter()
        self.trace.append(
            {
                "stage": stage,
                "ms": round((now - self.last) * 1000.0, 1),
                "detail": detail,
            }
        )
        self.last = now

    def total_ms(self):
        return round((time.perf_counter() - self.started) * 1000.0, 1)


def analyse(image_u8, crop="tomato", region="tamil_nadu", language="en", use_llm=True):
    """Run the full pipeline on one decoded image."""
    language = i18n.normalise(language)
    if crop not in knowledge.crop_keys():
        crop = knowledge.crop_keys()[0]
    if region not in knowledge.region_keys():
        region = knowledge.regions()["default_region"]

    trace = []
    timer = _Timer(trace)

    # ---- 1. descriptor ------------------------------------------------
    descriptor, intermediates = features.extract(image_u8)
    timer.mark(
        "features",
        "%d-D descriptor, %s" % (descriptor.size, features.FEATURE_VERSION),
    )

    # ---- 2. classification + abstention ------------------------------
    try:
        prediction = classifier.predict(descriptor, intermediates)
        model_error = None
    except classifier.ModelUnavailable as exc:
        model_error = str(exc)
        prediction = {
            "verdict": "uncertain",
            "abstained": True,
            "reason": "model_unavailable",
            "detail": model_error,
            "confidence": None,
            "confidence_capped": False,
            "probability_infected": None,
            "probabilities": {},
            "ood_distance": None,
            "nearest_class": None,
            "gates": {},
            "caveat": config.SYNTHETIC_CAVEAT,
        }
    timer.mark(
        "classifier",
        "%s%s"
        % (
            prediction["verdict"],
            " (%s)" % prediction["reason"] if prediction["abstained"] else "",
        ),
    )

    # ---- 3. severity --------------------------------------------------
    calibration = classifier.severity_calibration()
    severity_result = severity_module.estimate(intermediates, calibration)

    # An abstention means we do not trust the leaf region, so a severity number
    # would be unfounded. Suppress it rather than show a figure we disown.
    if prediction["abstained"]:
        severity_result = dict(severity_result)
        severity_result["suppressed"] = True
        severity_result["available"] = False
        severity_value = 0.0
        band = "trace"
    else:
        severity_result["suppressed"] = False
        severity_value = severity_result["severity"]
        band = severity_result["band"]
    timer.mark(
        "severity",
        "suppressed (no verdict)"
        if prediction["abstained"]
        else "%.1f%% (%s)" % (severity_value, band),
    )

    # ---- 4. knowledge retrieval --------------------------------------
    entry = knowledge.entry_for(prediction["verdict"])
    timer.mark("knowledge", "entry %r" % entry["id"])

    # ---- 5/6. Llama + grounding guard --------------------------------
    confidence_pct = (
        100.0 * prediction["confidence"] if prediction["confidence"] else None
    )
    client = None
    if not use_llm:
        from app import llm

        client = llm.ExecuServeClient(enabled=False)

    advice = advisory.build(
        entry,
        crop,
        region,
        severity_value,
        band,
        confidence_pct,
        language=language,
        client=client,
        trace=trace,
    )

    # ---- 7. localisation ---------------------------------------------
    probable_causes = []
    for cause in entry.get("probable_causes") or []:
        probable_causes.append(
            {
                "name": cause["name"],
                "pathogen": cause.get("pathogen"),
                "pathogen_type": cause.get("pathogen_type"),
                "distinguishing_symptom": cause["distinguishing_symptom"],
                "organic": cause.get("organic"),
                "chemical": cause.get("chemical"),
            }
        )

    result = {
        "app": {"name": config.APP_NAME, "version": config.APP_VERSION},
        "verdict": prediction["verdict"],
        "verdict_label": i18n.verdict_label(prediction["verdict"], language),
        "abstained": prediction["abstained"],
        "abstain_reason": prediction["reason"],
        "abstain_detail": prediction["detail"],
        "confidence": prediction["confidence"],
        "confidence_capped": prediction["confidence_capped"],
        "probabilities": prediction["probabilities"],
        "caveat": prediction["caveat"],
        "gates": prediction["gates"],
        "severity": severity_result,
        "severity_band_label": i18n.band_label(band, language),
        "advisory": advice,
        "probable_causes": probable_causes,
        "probable_causes_note": i18n.t("probable_causes_note", language),
        "scouting": entry.get("scouting") or [],
        "region_note": knowledge.region_note(entry, region),
        "context": {
            "crop": crop,
            "crop_label": knowledge.crop_label(crop, language),
            "region": region,
            "region_label": knowledge.region_label(region, language),
            "language": language,
        },
        "model_error": model_error,
        "trace": trace,
        "timings_ms": {"total": timer.total_ms()},
    }
    return result
