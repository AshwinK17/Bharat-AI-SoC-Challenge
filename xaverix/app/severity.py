"""Severity estimation: the fraction of leaf area showing damage.

A calibrated colour heuristic, not a segmentation model. It reuses the HSV
planes and masks already computed by ``features.extract``, so the additional
cost is negligible.

Chlorotic (yellowing) tissue is weighted 0.5, having partially rather than
completely lost function. The linear calibration ``a * raw + b`` is fitted at
training time against the renderer exact lesion mask, which yields a measured
mean absolute error rather than an assumed one.

Known failure modes:
  * shadows and dark soil read as necrosis
  * red and purple cultivars read as damaged
  * dry leaf margins read as necrosis
  * specular highlights read as powdery mildew
  * several overlapping leaves break the leaf-area denominator
"""
from __future__ import annotations

import numpy as np

from app import knowledge

DEFAULT_CALIBRATION = (1.0, 0.0)


def estimate(intermediates, calibration=None):
    """Return a severity dict from the masks produced by ``features.extract``."""
    a, b = calibration or DEFAULT_CALIBRATION
    masks = intermediates["masks"]
    leaf = intermediates["leaf"]

    leaf_area = float(leaf.sum())
    if leaf_area < 1.0:
        return {
            "severity": 0.0,
            "band": "trace",
            "interval": 0.0,
            "raw": 0.0,
            "leaf_area_px": 0,
            "components": {},
            "method": "lesion-area colour heuristic (no leaf region found)",
            "available": False,
        }

    from app.features import majority3

    damage = majority3(masks["necrotic"] | masks["water_soaked"] | masks["pale"])
    chlorotic = majority3(masks["chlorotic"])

    damage_px = float((damage & leaf).sum())
    chlorotic_px = float((chlorotic & leaf).sum())

    raw = 100.0 * (damage_px + 0.5 * chlorotic_px) / leaf_area
    severity = float(np.clip(a * raw + b, 0.0, 100.0))

    return {
        "severity": round(severity, 1),
        "band": knowledge.severity_band(severity),
        # Widening interval: the heuristic is least reliable on heavy damage.
        "interval": round(5.0 + 0.2 * severity, 1),
        "raw": round(raw, 1),
        "leaf_area_px": int(leaf_area),
        "components": {
            "necrotic_pct": round(100.0 * float((masks["necrotic"] & leaf).sum()) / leaf_area, 1),
            "water_soaked_pct": round(100.0 * float((masks["water_soaked"] & leaf).sum()) / leaf_area, 1),
            "chlorotic_pct": round(100.0 * chlorotic_px / leaf_area, 1),
            "pale_pct": round(100.0 * float((masks["pale"] & leaf).sum()) / leaf_area, 1),
        },
        "method": "lesion-area colour heuristic, linearly calibrated on synthetic ground truth",
        "available": True,
    }


def raw_fraction(intermediates):
    """Uncalibrated damaged-area fraction; used by the calibration fit."""
    masks = intermediates["masks"]
    leaf = intermediates["leaf"]
    leaf_area = float(leaf.sum())
    if leaf_area < 1.0:
        return 0.0

    from app.features import majority3

    damage = majority3(masks["necrotic"] | masks["water_soaked"] | masks["pale"])
    chlorotic = majority3(masks["chlorotic"])
    return 100.0 * (
        float((damage & leaf).sum()) + 0.5 * float((chlorotic & leaf).sum())
    ) / leaf_area
