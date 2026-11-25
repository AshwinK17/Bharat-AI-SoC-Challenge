"""Train the leaf-health classifier, calibrate it, and fit the abstention gates.

Runs on the development machine. Gradients are implemented directly in NumPy:
scikit-learn and torch cannot be installed on the target device, and the
project avoids depending on anything the device cannot run.

Stages
------
1. Render leaves (healthy / infected) and a negative set of things that are not
   a single leaf, extracting the 150-D descriptor for each.
2. Fit a standardiser and train a logistic head with Adam.
3. Temperature-scale the logit on a held-out split, so confidence means
   something.
4. Fit the Mahalanobis statistics and choose the three abstention thresholds
   against the negative set.
5. Least-squares calibrate the severity heuristic against the renderer's exact
   lesion fraction, which gives a real mean absolute error to publish.
6. Write models/leaf_health.npz, plus demo images into samples/.

Usage
-----
    python tools/train.py                 # default sizes
    python tools/train.py --quick         # small run, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import config, features as F, severity as SEV  # noqa: E402
from tools import synth  # noqa: E402

CLASSES = ("healthy", "infected")


# --------------------------------------------------------------- generation
def generate(n_per_class, rng, augment=True):
    X, y, leaf_frac, green_frac = [], [], [], []
    for label, _name in enumerate(CLASSES):
        for _ in range(n_per_class):
            hint = rng.uniform(0.05, 0.8) if label == 1 else None
            image, _info = synth.render_leaf(bool(label), rng, severity_hint=hint)
            if augment:
                image = synth.augment(image, rng)
            descriptor, intermediates = F.extract(image)
            X.append(descriptor)
            y.append(label)
            leaf_frac.append(intermediates["leaf_frac"])
            green_frac.append(intermediates["green_frac"])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(leaf_frac, dtype=np.float64),
        np.asarray(green_frac, dtype=np.float64),
    )


def generate_severity_pairs(count, rng):
    """Severity calibration data, deliberately NOT augmented.

    The severity heuristic runs on the photograph the farmer actually took, so
    calibrating it against rotated, blurred and colour-shifted copies would fit
    the wrong relationship. Augmentation exists to harden the *classifier*.
    """
    raw, truth = [], []
    for index in range(count):
        # A third of the set is healthy. Fitting only on infected leaves leaves
        # the intercept unconstrained at zero damage: it landed at 17.6, so a
        # perfectly healthy leaf was reported as 17.6% damaged.
        infected = (index % 3) != 0
        hint = rng.uniform(0.02, 0.85) if infected else None
        image, info = synth.render_leaf(infected, rng, severity_hint=hint)
        _descriptor, intermediates = F.extract(image)
        raw.append(SEV.raw_fraction(intermediates))
        truth.append(info["lesion_fraction"])
    return np.asarray(raw, dtype=np.float64), np.asarray(truth, dtype=np.float64)


def generate_negatives(count, rng):
    X, leaf_frac, green_frac, kinds = [], [], [], []
    for _ in range(count):
        kind = str(rng.choice(synth.NEGATIVE_KINDS))
        image = synth.render_negative(rng, kind=kind)
        if rng.random() < 0.5:
            image = synth.augment(image, rng)
        descriptor, intermediates = F.extract(image)
        X.append(descriptor)
        leaf_frac.append(intermediates["leaf_frac"])
        green_frac.append(intermediates["green_frac"])
        kinds.append(kind)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(leaf_frac, dtype=np.float64),
        np.asarray(green_frac, dtype=np.float64),
        kinds,
    )


# ------------------------------------------------------------------ model
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def train_logistic(X, y, epochs=600, lr=3e-3, l2=3e-4, batch=128, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    mw, vw = np.zeros(d), np.zeros(d)
    mb = vb = 0.0
    step = 0
    target = y.astype(np.float64)

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            xb, yb = X[idx].astype(np.float64), target[idx]
            m = len(idx)

            p = sigmoid(xb @ w + b)
            dz = (p - yb) / m
            gw = xb.T @ dz + l2 * w
            gb = dz.sum()

            step += 1
            mw = 0.9 * mw + 0.1 * gw
            vw = 0.999 * vw + 0.001 * (gw * gw)
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * (gb * gb)
            bias1 = 1 - 0.9 ** step
            bias2 = 1 - 0.999 ** step
            w -= lr * (mw / bias1) / (np.sqrt(vw / bias2) + 1e-8)
            b -= lr * (mb / bias1) / (np.sqrt(vb / bias2) + 1e-8)

        if verbose and (epoch + 1) % 150 == 0:
            p = sigmoid(X @ w + b)
            acc = ((p > 0.5).astype(int) == y).mean()
            loss = -np.mean(
                target * np.log(p + 1e-12) + (1 - target) * np.log(1 - p + 1e-12)
            )
            print("    epoch %4d  loss %.4f  train acc %.3f" % (epoch + 1, loss, acc))

    return w, b


def fit_temperature(logits, y):
    """Grid-search the temperature that minimises negative log likelihood."""
    best = (1e18, 1.0)
    target = y.astype(np.float64)
    for T in np.exp(np.linspace(np.log(0.3), np.log(12.0), 120)):
        p = sigmoid(logits / T)
        nll = -np.mean(
            target * np.log(p + 1e-12) + (1 - target) * np.log(1 - p + 1e-12)
        )
        if nll < best[0]:
            best = (nll, float(T))
    return best[1]


def expected_calibration_error(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    confidence = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(int) == y).astype(float)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def fit_mahalanobis(Xs, y, shrink=0.12):
    """Tied-covariance Mahalanobis in standardised feature space."""
    centroids = np.stack([Xs[y == c].mean(0) for c in range(len(CLASSES))])
    deviations = Xs - centroids[y]
    cov = (deviations.T @ deviations) / max(len(Xs), 1)
    trace_mean = np.trace(cov) / cov.shape[0]
    cov = (1 - shrink) * cov + shrink * trace_mean * np.eye(cov.shape[0])

    condition = np.linalg.cond(cov)
    if not np.isfinite(condition) or condition > 1e8:
        precision = np.diag(1.0 / np.maximum(np.diag(cov), 1e-6))
        kind = "diagonal"
    else:
        precision = np.linalg.inv(cov)
        kind = "full"
    return centroids.astype(np.float32), precision.astype(np.float32), kind, float(condition)


def mahalanobis(Xs, centroids, precision):
    """Minimum squared distance to any class centroid."""
    scores = []
    for centroid in centroids:
        delta = Xs - centroid
        scores.append(np.einsum("nd,de,ne->n", delta, precision, delta))
    return np.min(np.stack(scores, axis=1), axis=1)


# ------------------------------------------------------------------- main
def gate0_pass(leaf_frac, green_frac, lo, hi, green_min):
    """Gate 0: is there a single leaf-like region filling a sensible area?"""
    leaf_frac = np.asarray(leaf_frac)
    green_frac = np.asarray(green_frac)
    return (leaf_frac >= lo) & (leaf_frac <= hi) & (green_frac >= green_min)


def gate_reject_mask(distance, probability, leaf_frac, green_frac, thresholds):
    """True where the full three-gate cascade refuses to answer."""
    passes_gate0 = gate0_pass(
        leaf_frac,
        green_frac,
        thresholds["leaf_frac_lo"],
        thresholds["leaf_frac_hi"],
        thresholds["green_frac_min"],
    )
    confidence = np.maximum(probability, 1 - probability)
    return (
        (~passes_gate0)
        | (np.asarray(distance) > thresholds["tau_maha"])
        | (confidence < thresholds["p_min"])
    )


def gate_reject_rate(distance, probability, leaf_frac, green_frac, thresholds):
    """Overall rejection rate, plus which gate did the work."""
    passes_gate0 = gate0_pass(
        leaf_frac,
        green_frac,
        thresholds["leaf_frac_lo"],
        thresholds["leaf_frac_hi"],
        thresholds["green_frac_min"],
    )
    confidence = np.maximum(probability, 1 - probability)
    by_gate0 = ~passes_gate0
    by_maha = passes_gate0 & (np.asarray(distance) > thresholds["tau_maha"])
    by_conf = (
        passes_gate0
        & (np.asarray(distance) <= thresholds["tau_maha"])
        & (confidence < thresholds["p_min"])
    )
    total = by_gate0 | by_maha | by_conf
    breakdown = "(gate0 %.0f%%, maha %.0f%%, conf %.0f%%)" % (
        100 * by_gate0.mean(),
        100 * by_maha.mean(),
        100 * by_conf.mean(),
    )
    return float(total.mean()), breakdown


def per_kind_rejection(distance, probability, leaf_frac, green_frac, kinds, thresholds):
    """Rejection rate per negative kind, so weak spots are visible not averaged away."""
    rejected = gate_reject_mask(distance, probability, leaf_frac, green_frac, thresholds)
    out = {}
    kinds = np.asarray(kinds)
    for kind in sorted(set(kinds.tolist())):
        mask = kinds == kind
        if mask.any():
            out[kind] = float(rejected[mask].mean())
    return out


def calibrate(calib_logits, calib_y, val_logits, val_y):
    """Fit the temperature, but never make calibration worse.

    Minimising NLL on a small split can drive the temperature to the grid edge
    and flatten every probability towards 0.5, which lowers NLL while making
    the expected calibration error worse. So the fitted value is accepted only
    if it actually improves ECE on the validation split; otherwise we keep the
    best of the candidates by ECE.
    """
    ece_before = expected_calibration_error(sigmoid(val_logits), val_y)

    fitted = fit_temperature(calib_logits, calib_y)
    ece_fitted = expected_calibration_error(sigmoid(val_logits / fitted), val_y)

    best_t, best_ece = 1.0, ece_before
    for candidate in np.exp(np.linspace(np.log(0.5), np.log(6.0), 60)):
        ece = expected_calibration_error(sigmoid(val_logits / candidate), val_y)
        if ece < best_ece:
            best_t, best_ece = float(candidate), ece

    if ece_fitted <= best_ece:
        return fitted, ece_before, ece_fitted
    return best_t, ece_before, best_ece


def main():
    parser = argparse.ArgumentParser(description="Train the Xaverix leaf classifier")
    parser.add_argument("--train", type=int, default=700, help="leaves per class")
    parser.add_argument("--val", type=int, default=200, help="validation per class")
    parser.add_argument("--calib", type=int, default=200, help="calibration per class")
    parser.add_argument("--negatives", type=int, default=700)
    parser.add_argument("--severity", type=int, default=500, help="severity pairs")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--l2", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20251125)
    parser.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    parser.add_argument("--samples", type=int, default=20, help="demo images to write")
    args = parser.parse_args()

    if args.quick:
        args.train, args.val, args.calib = 90, 60, 60
        args.negatives, args.epochs, args.samples, args.severity = 150, 200, 6, 80

    rng = np.random.default_rng(args.seed)
    started = time.time()

    print("[1/6] rendering data (this is the slow part)")
    Xtr, ytr, ltr, gtr = generate(args.train, rng)
    print("      train       %5d" % len(Xtr))
    Xva, yva, lva, gva = generate(args.val, rng)
    print("      val         %5d" % len(Xva))
    Xca, yca, _, _ = generate(args.calib, rng)
    print("      calib       %5d" % len(Xca))
    Xneg, lneg, gneg, neg_kinds = generate_negatives(args.negatives, rng)
    print("      negatives   %5d" % len(Xneg))
    sev_raw, sev_true = generate_severity_pairs(args.severity, rng)
    print("      severity    %5d (unaugmented)" % len(sev_raw))

    print("[2/6] standardising and training the logistic head")
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-6] = 1.0

    Ztr = (Xtr - mu) / sd
    Zva = (Xva - mu) / sd
    Zca = (Xca - mu) / sd
    Zneg = (Xneg - mu) / sd

    w, b = train_logistic(Ztr, ytr, epochs=args.epochs, l2=args.l2, seed=args.seed)

    train_acc = ((sigmoid(Ztr @ w + b) > 0.5).astype(int) == ytr).mean()
    val_logits = Zva @ w + b
    val_acc = ((sigmoid(val_logits) > 0.5).astype(int) == yva).mean()
    print("      train acc %.3f | val acc %.3f" % (train_acc, val_acc))

    print("[3/6] temperature calibration")
    calib_logits = Zca @ w + b
    temperature, ece_before, ece_after = calibrate(calib_logits, yca, val_logits, yva)
    p_val = sigmoid(val_logits / temperature)
    print("      T = %.3f | ECE %.3f -> %.3f" % (temperature, ece_before, ece_after))

    print("[4/6] fitting the abstention gates")
    centroids, precision, cov_kind, condition = fit_mahalanobis(Ztr, ytr)
    d_val = mahalanobis(Zva, centroids, precision)
    d_neg = mahalanobis(Zneg, centroids, precision)

    # Gate 0 thresholds are FITTED, not guessed. A hand-picked upper bound of
    # 0.95 on leaf area rejected a quarter of genuine leaves, because a leaf
    # photographed close up legitimately fills the whole frame. The green
    # fraction turns out to be the discriminative half: most negatives have
    # none at all.
    leaf_lo = float(max(0.20, np.percentile(ltr, 1)))
    leaf_hi = 1.01  # no upper bound: filling the frame is correct behaviour
    green_min = float(max(0.05, np.percentile(gtr, 3)))

    # Gate 0 already removes much of the negative set, so the Mahalanobis
    # threshold is fitted against what survives it rather than double-counting.
    g0_val = gate0_pass(lva, gva, leaf_lo, leaf_hi, green_min)
    g0_neg = gate0_pass(lneg, gneg, leaf_lo, leaf_hi, green_min)

    survivors = d_neg[g0_neg]
    id_pass = d_val[g0_val]
    if survivors.size and id_pass.size:
        tau_target = float(np.percentile(survivors, 10))
        tau_ceiling = float(np.percentile(id_pass, 99))
        tau_floor = float(np.percentile(id_pass, 92))
        tau_maha = max(min(tau_target, tau_ceiling), tau_floor)
    else:
        tau_maha = float(np.percentile(d_val, 99)) if d_val.size else 1e9

    confidence_val = np.maximum(p_val, 1 - p_val)
    # Floor at 0.60: searching from 0.50 satisfies the accuracy target
    # immediately and leaves Gate 2 switched off entirely, so a genuinely
    # borderline leaf would still get a verdict.
    p_min = 0.60
    for candidate in np.linspace(0.60, 0.90, 31):
        keep = confidence_val >= candidate
        if keep.sum() < max(10, int(0.25 * len(yva))):
            break
        accepted_acc = (((p_val > 0.5).astype(int) == yva)[keep]).mean()
        if accepted_acc >= 0.90:
            p_min = float(candidate)
            break

    thresholds = {
        "tau_maha": tau_maha,
        "p_min": p_min,
        "leaf_frac_lo": leaf_lo,
        "leaf_frac_hi": leaf_hi,
        "green_frac_min": green_min,
    }

    p_neg = sigmoid((Zneg @ w + b) / temperature)

    # The number that actually matters: all three gates together.
    id_reject, id_breakdown = gate_reject_rate(d_val, p_val, lva, gva, thresholds)
    ood_reject, ood_breakdown = gate_reject_rate(d_neg, p_neg, lneg, gneg, thresholds)

    accepted = ~gate_reject_mask(d_val, p_val, lva, gva, thresholds)
    accepted_acc = (
        float((((p_val > 0.5).astype(int) == yva)[accepted]).mean())
        if accepted.any()
        else 0.0
    )

    print("      covariance %s (cond %.2e)" % (cov_kind, condition))
    print("      tau_maha %.1f | p_min %.2f" % (tau_maha, p_min))
    print("      leaves    rejected %5.1f%%  %s" % (100 * id_reject, id_breakdown))
    print("      negatives rejected %5.1f%%  %s" % (100 * ood_reject, ood_breakdown))
    print("      accuracy on accepted leaves %.3f" % accepted_acc)

    per_kind = per_kind_rejection(d_neg, p_neg, lneg, gneg, neg_kinds, thresholds)
    worst = sorted(per_kind.items(), key=lambda kv: kv[1])[:5]
    print(
        "      weakest negative kinds: %s"
        % ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in worst)
    )

    print("[5/6] calibrating severity against the renderer ground truth")
    A = np.stack([sev_raw, np.ones(len(sev_raw))], axis=1)
    coefficients, *_ = np.linalg.lstsq(A, sev_true, rcond=None)
    sev_a, sev_b = float(coefficients[0]), float(coefficients[1])
    predicted = np.clip(sev_a * sev_raw + sev_b, 0, 100)
    sev_mae = float(np.abs(predicted - sev_true).mean())
    raw_mae = float(np.abs(sev_raw - sev_true).mean())
    correlation = (
        float(np.corrcoef(sev_raw, sev_true)[0, 1]) if len(sev_raw) > 2 else 0.0
    )
    print(
        "      a=%.3f b=%.2f | MAE %.1f points (uncalibrated %.1f) | r=%.2f"
        % (sev_a, sev_b, sev_mae, raw_mae, correlation)
    )

    print("[6/6] writing the model and demo samples")
    train_meta = {
        "feature_version": F.FEATURE_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "synthetic_data": True,
        "data_note": (
            "Procedurally generated leaves only. No PlantVillage, PlantDoc or "
            "IP102 data was used. Every accuracy below is on held-out "
            "synthetic images and is NOT a real-world accuracy."
        ),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_calib": int(len(Xca)),
        "n_negatives": int(len(Xneg)),
        "n_severity_pairs": int(len(sev_raw)),
        "train_accuracy_synthetic": round(float(train_acc), 4),
        "val_accuracy_synthetic": round(float(val_acc), 4),
        "accepted_accuracy_synthetic": round(accepted_acc, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "temperature": round(temperature, 4),
        "covariance": cov_kind,
        "covariance_condition": condition,
        "gate_reject_rate_negatives": round(ood_reject, 4),
        "gate_reject_rate_leaves": round(id_reject, 4),
        "gate_reject_breakdown_negatives": ood_breakdown,
        "gate_reject_per_negative_kind": {k: round(v, 3) for k, v in per_kind.items()},
        "severity_mae_synthetic": round(sev_mae, 2),
        "severity_mae_uncalibrated": round(raw_mae, 2),
        "severity_correlation": round(correlation, 3),
        "thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
        "classes": list(CLASSES),
        "seed": args.seed,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.MODEL_PATH,
        feat_version=np.array(F.FEATURE_VERSION),
        classes=np.array(CLASSES),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        w=w.astype(np.float32),
        b=np.float32(b),
        temperature=np.float32(temperature),
        centroids=centroids,
        precision=precision,
        tau_maha=np.float32(tau_maha),
        p_min=np.float32(p_min),
        margin_min=np.float32(0.12),
        leaf_frac_lo=np.float32(leaf_lo),
        leaf_frac_hi=np.float32(leaf_hi),
        green_frac_min=np.float32(green_min),
        severity_a=np.float32(sev_a),
        severity_b=np.float32(sev_b),
        train_meta=np.array(json.dumps(train_meta)),
    )
    size_kb = config.MODEL_PATH.stat().st_size / 1024
    print("      %s (%.0f KB)" % (config.MODEL_PATH, size_kb))

    write_samples(args.samples, rng)

    print("")
    print("done in %.0f s" % (time.time() - started))


def write_samples(count, rng):
    """Demo images shipped with the app, so the UI works with no camera."""
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for existing in config.SAMPLES_DIR.glob("*.png"):
        existing.unlink()

    manifest = []
    half = max(count // 2, 1)
    for index in range(half):
        image, info = synth.render_leaf(False, rng)
        name = "healthy_%02d.png" % (index + 1)
        Image.fromarray(image).save(config.SAMPLES_DIR / name)
        manifest.append({"file": name, "expected": "healthy", "true_damage_pct": round(info["lesion_fraction"], 1)})

    for index in range(count - half):
        hint = 0.1 + 0.7 * (index / max(count - half - 1, 1))
        image, info = synth.render_leaf(True, rng, severity_hint=hint)
        name = "infected_%02d.png" % (index + 1)
        Image.fromarray(image).save(config.SAMPLES_DIR / name)
        manifest.append(
            {
                "file": name,
                "expected": "infected",
                "lesion_kinds": info["kinds"],
                "true_damage_pct": round(info["lesion_fraction"], 1),
            }
        )

    # Two out-of-distribution demo images: these must make the app abstain.
    for name, kind in (("ood_lined_paper.png", "lined_paper"), ("ood_soil.png", "soil")):
        image = synth.render_negative(rng, kind=kind)
        Image.fromarray(image).save(config.SAMPLES_DIR / name)
        manifest.append({"file": name, "expected": "uncertain", "kind": kind})

    with open(config.SAMPLES_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"samples": manifest}, fh, indent=2)
    print("      %d sample images + manifest" % len(manifest))


if __name__ == "__main__":
    main()
