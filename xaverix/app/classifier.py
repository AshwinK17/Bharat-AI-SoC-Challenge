"""Leaf-health classifier with a three-gate abstention cascade.

The classifier is trained on synthetic imagery, so arbitrary real-world input
is out of distribution by construction. Confidently classifying such input is
the dominant failure risk: an incorrect verdict presented with high confidence
could lead to an unnecessary or wrong pesticide application. The cascade below
exists so the classifier can decline instead.

Three gates run cheapest first, and each records the value that triggered it:

Gate 0  leaf presence   is a green leaf region present and large enough?
Gate 1  Mahalanobis     is the descriptor near the training distribution?
Gate 2  confidence      is the calibrated probability decisive enough?

Any gate firing yields ``uncertain`` with no health verdict and no severity
figure. Displayed confidence is capped below 100%, since a model validated
only on synthetic data should not present absolute certainty.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from app import config, features as F

_lock = threading.Lock()
_model = None


class ModelUnavailable(Exception):
    """The model file is missing or was built from different features."""


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def load(force=False):
    """Load and validate the model file. Cached."""
    global _model
    with _lock:
        if _model is not None and not force:
            return _model

        if not config.MODEL_PATH.exists():
            raise ModelUnavailable(
                "model file %s is missing; run: python tools/train.py"
                % config.MODEL_PATH
            )

        data = np.load(config.MODEL_PATH, allow_pickle=False)

        version = str(data["feat_version"])
        if version != F.FEATURE_VERSION:
            # Refuse rather than predict from features the model was not
            # fitted on; silent train/inference skew is hard to detect later.
            raise ModelUnavailable(
                "model was trained on features %r but this build computes %r; "
                "retrain with: python tools/train.py" % (version, F.FEATURE_VERSION)
            )

        try:
            meta = json.loads(str(data["train_meta"]))
        except (ValueError, KeyError):
            meta = {}

        _model = {
            "classes": [str(c) for c in data["classes"]],
            "mu": data["mu"].astype(np.float64),
            "sd": data["sd"].astype(np.float64),
            "w": data["w"].astype(np.float64),
            "b": float(data["b"]),
            "temperature": float(data["temperature"]),
            "centroids": data["centroids"].astype(np.float64),
            "precision": data["precision"].astype(np.float64),
            "tau_maha": float(data["tau_maha"]),
            "p_min": float(data["p_min"]),
            "leaf_frac_lo": float(data["leaf_frac_lo"]),
            "leaf_frac_hi": float(data["leaf_frac_hi"]),
            "green_frac_min": float(data["green_frac_min"]),
            "severity_calibration": (
                float(data["severity_a"]),
                float(data["severity_b"]),
            ),
            "feature_version": version,
            "train_meta": meta,
        }
        return _model


def is_available():
    try:
        load()
        return True
    except ModelUnavailable:
        return False


def status():
    """Model status for /api/config, including the absent-model case."""
    try:
        model = load()
    except ModelUnavailable as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "classes": model["classes"],
        "feature_version": model["feature_version"],
        "thresholds": {
            "tau_maha": round(model["tau_maha"], 2),
            "p_min": round(model["p_min"], 3),
            "leaf_frac_lo": round(model["leaf_frac_lo"], 3),
            "leaf_frac_hi": round(model["leaf_frac_hi"], 3),
            "green_frac_min": round(model["green_frac_min"], 3),
        },
        "training": model["train_meta"],
    }


def _mahalanobis(z, centroids, precision):
    deltas = z[None, :] - centroids
    distances = np.einsum("kd,de,ke->k", deltas, precision, deltas)
    return float(np.min(distances)), int(np.argmin(distances))


def predict(descriptor, intermediates):
    """Classify one descriptor, or abstain.

    Every gate value is returned even when abstaining, so the pipeline trace
    can report precisely which criterion withheld the verdict.
    """
    model = load()

    leaf_frac = float(intermediates.get("leaf_frac", 0.0))
    green_frac = float(intermediates.get("green_frac", 0.0))

    z = (np.asarray(descriptor, dtype=np.float64) - model["mu"]) / model["sd"]
    logit = float(z @ model["w"] + model["b"])
    # Cast to float: NumPy scalars propagate into comparisons as np.bool_,
    # which json.dumps cannot serialise.
    probability = float(_sigmoid(logit / model["temperature"]))
    distance, nearest = _mahalanobis(z, model["centroids"], model["precision"])

    confidence = float(max(probability, 1.0 - probability))
    label_index = 1 if probability > 0.5 else 0

    gates = {
        "leaf_fraction": round(leaf_frac, 4),
        "leaf_fraction_min": round(model["leaf_frac_lo"], 4),
        "leaf_fraction_max": round(model["leaf_frac_hi"], 4),
        "green_fraction": round(green_frac, 4),
        "green_fraction_min": round(model["green_frac_min"], 4),
        "ood_distance": round(distance, 2),
        "ood_threshold": round(model["tau_maha"], 2),
        "confidence_raw": round(confidence, 4),
        "confidence_min": round(model["p_min"], 4),
    }

    # ---- Gate 0: is there a leaf at all? -----------------------------
    if leaf_frac < model["leaf_frac_lo"]:
        return _abstain(
            "no_leaf_detected",
            "No single leaf fills enough of the frame (%.0f%% of the image, "
            "at least %.0f%% expected). Hold the camera 20 to 30 cm from one "
            "leaf." % (100 * leaf_frac, 100 * model["leaf_frac_lo"]),
            gates,
            probability,
            model,
        )
    if leaf_frac > model["leaf_frac_hi"]:
        return _abstain(
            "frame_filled",
            "The whole frame is one surface, so no leaf outline can be found.",
            gates,
            probability,
            model,
        )
    if green_frac < model["green_frac_min"]:
        return _abstain(
            "not_leaf_like",
            "Too little green leaf tissue in the frame (%.0f%%, at least %.0f%% "
            "expected). Photograph one leaf against a plain background."
            % (100 * green_frac, 100 * model["green_frac_min"]),
            gates,
            probability,
            model,
        )

    # ---- Gate 1: is this anywhere near what we trained on? -----------
    if distance > model["tau_maha"]:
        return _abstain(
            "out_of_distribution",
            "This image is outside the range this prototype was trained on "
            "(distance %.0f, limit %.0f). Xaverix abstains rather than guess."
            % (distance, model["tau_maha"]),
            gates,
            probability,
            model,
        )

    # ---- Gate 2: is the model decisive? ------------------------------
    if confidence < model["p_min"]:
        return _abstain(
            "low_confidence",
            "The model is not confident enough to call this leaf either way "
            "(%.0f%%, at least %.0f%% required)."
            % (100 * confidence, 100 * model["p_min"]),
            gates,
            probability,
            model,
        )

    label = model["classes"][label_index]
    return {
        "verdict": label,
        "abstained": False,
        "reason": None,
        "detail": None,
        "confidence": round(min(confidence, config.CONFIDENCE_DISPLAY_CAP), 4),
        "confidence_capped": confidence > config.CONFIDENCE_DISPLAY_CAP,
        "probability_infected": round(probability, 4),
        "probabilities": {
            model["classes"][0]: round(1.0 - probability, 4),
            model["classes"][1]: round(probability, 4),
        },
        "ood_distance": round(distance, 2),
        "nearest_class": model["classes"][nearest],
        "gates": gates,
        "caveat": config.SYNTHETIC_CAVEAT,
    }


def _abstain(reason, detail, gates, probability, model):
    return {
        "verdict": "uncertain",
        "abstained": True,
        "reason": reason,
        "detail": detail,
        # No confidence is reported when abstaining: there is no verdict for
        # the figure to describe.
        "confidence": None,
        "confidence_capped": False,
        "probability_infected": round(probability, 4),
        "probabilities": {
            model["classes"][0]: round(1.0 - probability, 4),
            model["classes"][1]: round(probability, 4),
        },
        "ood_distance": gates["ood_distance"],
        "nearest_class": None,
        "gates": gates,
        "caveat": config.SYNTHETIC_CAVEAT,
    }


def severity_calibration():
    try:
        return load()["severity_calibration"]
    except ModelUnavailable:
        return (1.0, 0.0)
