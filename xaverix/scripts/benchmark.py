"""Per-stage latency and memory benchmark.

Run this on the phone, inside Termux, to produce the real Arm numbers for the
technical report:

    TARGET_DEVICE="Redmi Note 13 (Arm64, Termux)" python scripts/benchmark.py

``TARGET_DEVICE`` is recorded in the output. Without it the result is labelled
"unspecified" rather than attributed to whatever machine happened to run it,
so development-host figures cannot be mistaken for device figures.

Memory is read from /proc/self/status. psutil is deliberately not used,
because it needs a compiler and will not build in Termux.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
    advisory,
    classifier,
    config,
    features,
    knowledge,
    llm,
    severity as severity_module,
)
from tools import synth  # noqa: E402


def peak_rss_mb():
    """Peak resident set size in MB, without psutil."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return round(usage / (1024.0 if usage > 1 << 20 else 1.0) / 1024.0, 1)
    except Exception:
        return None


def current_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        return None
    return None


def percentiles(samples):
    array = np.asarray(samples, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(array, 50)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "mean": round(float(array.mean()), 2),
        "min": round(float(array.min()), 2),
        "max": round(float(array.max()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Xaverix pipeline")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="also time one grounded Llama advisory (needs ExecuServe running)",
    )
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "benchmark.json"))
    args = parser.parse_args()

    device = config.TARGET_DEVICE or "unspecified"
    if device == "unspecified":
        print(
            "WARNING: TARGET_DEVICE is not set, so these numbers will be "
            "labelled 'unspecified'.\n"
            "         On the phone, run:\n"
            '           TARGET_DEVICE="<your phone> (Arm64 Android, Termux)" '
            "python scripts/benchmark.py\n"
        )

    print("Xaverix benchmark - %d iterations on %s" % (args.iterations, device))
    print("platform: %s %s / %s" % (platform.system(), platform.release(), platform.machine()))

    if not classifier.is_available():
        print("\nmodel unavailable: %s" % classifier.status()["error"], file=sys.stderr)
        return 2

    # Prepare a fixed set of images so every stage sees the same work.
    rng = np.random.default_rng(2025)
    images = []
    for index in range(args.iterations):
        infected = index % 2 == 1
        image, _ = synth.render_leaf(
            infected, rng, severity_hint=0.4 if infected else None
        )
        images.append(image)

    # Warm up: first call pays for lazy loads and NumPy's own warm-up.
    descriptor, intermediates = features.extract(images[0])
    classifier.predict(descriptor, intermediates)

    timings = {"decode": [], "features": [], "classifier": [], "severity": [], "knowledge": []}
    calibration = classifier.severity_calibration()

    import io

    from PIL import Image

    encoded = []
    for image in images:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=88)
        encoded.append(buffer.getvalue())

    for index, image in enumerate(images):
        start = time.perf_counter()
        from app import pipeline

        decoded = pipeline.decode_image(encoded[index])
        timings["decode"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        descriptor, intermediates = features.extract(decoded)
        timings["features"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        prediction = classifier.predict(descriptor, intermediates)
        timings["classifier"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        severity_module.estimate(intermediates, calibration)
        timings["severity"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        knowledge.entry_for(prediction["verdict"])
        timings["knowledge"].append((time.perf_counter() - start) * 1000)

    stages = {name: percentiles(values) for name, values in timings.items()}
    vision_total = sum(stages[name]["p50"] for name in timings)

    print("\nper-stage latency (ms)")
    print("  %-12s %8s %8s %8s" % ("stage", "p50", "p95", "mean"))
    for name in ("decode", "features", "classifier", "severity", "knowledge"):
        row = stages[name]
        print("  %-12s %8.2f %8.2f %8.2f" % (name, row["p50"], row["p95"], row["mean"]))
    print("  %-12s %8.2f" % ("TOTAL", vision_total))

    # --------------------------------------------------- optional LLM leg
    llm_result = {"measured": False}
    if args.llm:
        client = llm.client()
        info = client.info()
        if not info.get("usable"):
            llm_result["note"] = "ExecuServe not usable: %s" % info.get("reason")
            print("\nllama: not benchmarked (%s)" % info.get("reason"))
        else:
            print("\nllama: warming up, this can take a minute on a phone...")
            client.warm_up_async()
            client.wait_warm(config.LLM_WARMUP_TIMEOUT)
            entry = knowledge.entry_for("infected")
            start = time.perf_counter()
            advice = advisory.build(
                entry, "tomato", "tamil_nadu", 38.0, "moderate", 74.0, client=client
            )
            elapsed = (time.perf_counter() - start) * 1000
            llm_result = {
                "measured": True,
                "source": advice["source"],
                "latency_ms": round(elapsed),
                "guard_decision": (advice.get("guard") or {}).get("decision"),
                "guard_support": (advice.get("guard") or {}).get("support"),
                "model": advice.get("model"),
            }
            print(
                "  advisory: %s in %.1f s (guard: %s)"
                % (advice["source"], elapsed / 1000, llm_result["guard_decision"])
            )

    # ---------------------------------------------------------- energy
    # Reported as an estimate with the assumption recorded alongside it, so the
    # figure cannot be mistaken for a measurement.
    assumed_watts = 2.0
    energy_mj = round(assumed_watts * (vision_total / 1000.0) * 1000.0, 1)

    report = {
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "target_device": device,
        "device_is_specified": device != "unspecified",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "iterations": args.iterations,
        "feature_version": features.FEATURE_VERSION,
        "stages_ms": stages,
        "vision_total_p50_ms": round(vision_total, 2),
        "memory": {
            "current_rss_mb": current_rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            "note": "read from /proc/self/status; psutil is not used because it cannot build in Termux",
        },
        "llm": llm_result,
        "energy": {
            "estimated_mj_per_image": energy_mj,
            "method": "ESTIMATE ONLY: assumed %.1f W average CPU draw times measured "
            "wall-clock latency. Not measured with a power monitor." % assumed_watts,
            "measured": False,
        },
        "notes": [
            "The vision pipeline is pure NumPy. Llama is the ExecuTorch component, "
            "hosted by ExecuServe on this device.",
            "Latency excludes camera capture and browser rendering.",
            "Energy is an estimate and is labelled as such.",
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\nmemory: current %s MB, peak %s MB"
          % (report["memory"]["current_rss_mb"], report["memory"]["peak_rss_mb"]))
    print("energy: ~%.0f mJ per image (ESTIMATE, assumed %.1f W)" % (energy_mj, assumed_watts))
    print("written to %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
