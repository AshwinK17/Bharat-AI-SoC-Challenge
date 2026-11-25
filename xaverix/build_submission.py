"""Build a distributable archive, then verify it from a clean extraction.

Model weights and sample images are treated as **required** contents: if any
are missing the build fails rather than producing an archive that silently
falls back to default behaviour when extracted elsewhere.

After packaging, the archive is extracted into a temporary directory where the
test suite and one end-to-end analysis are run, so the artefact that ships is
the artefact that was verified.

    python build_submission.py
    python build_submission.py --skip-verify     (faster, not recommended)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402

# Everything that must be present, or the build is not a submission.
REQUIRED = (
    "server.py",
    "README.md",
    "app/features.py",
    "app/classifier.py",
    "app/guard.py",
    "app/llm.py",
    "app/offline_guard.py",
    "data/knowledge_base.json",
    "data/i18n.json",
    "models/leaf_health.npz",
    "samples/manifest.json",
    "web/index.html",
    "docs/TECHNICAL_REPORT.md",
    "docs/PHONE_SETUP.md",
    "docs/LIMITATIONS.md",
)

INCLUDE_DIRS = ("app", "data", "models", "tools", "scripts", "samples", "web", "tests", "docs")
INCLUDE_FILES = ("server.py", "build_submission.py", "README.md", "requirements.txt")

# Never ship caches, logs or editor droppings.
EXCLUDE_DIRS = {"__pycache__", ".git", ".idea", ".vscode", ".pytest_cache", "node_modules"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".swp"}
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", "server.log", "train_log.txt"}


def _should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    return False


def collect():
    files = []
    for name in INCLUDE_FILES:
        candidate = ROOT / name
        if candidate.is_file():
            files.append(candidate)
    for directory in INCLUDE_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not _should_skip(path.relative_to(ROOT)):
                files.append(path)

    # Ship the benchmark result if one exists, so the report's numbers are
    # traceable to a file in the submission.
    benchmark = config.RESULTS_DIR / "benchmark.json"
    if benchmark.is_file():
        files.append(benchmark)
    return files


def check_required(files):
    present = {str(path.relative_to(ROOT)).replace(os.sep, "/") for path in files}
    missing = [name for name in REQUIRED if name not in present]
    return missing


def build(output_path: Path, files):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT)
            archive.write(path, arcname=str(Path("xaverix") / relative).replace(os.sep, "/"))
    return output_path


def verify(zip_path: Path):
    """Unzip into a temporary directory and run the suite there."""
    workdir = Path(tempfile.mkdtemp(prefix="xaverix-verify-"))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(workdir)
        project = workdir / "xaverix"

        print("  unzipped to %s" % project)

        # 1. The model must actually be in the archive.
        model = project / "models" / "leaf_health.npz"
        if not model.is_file():
            return False, "models/leaf_health.npz is not in the archive"
        print("  model present: %.0f KB" % (model.stat().st_size / 1024))

        samples = list((project / "samples").glob("*.png"))
        if len(samples) < 4:
            return False, "only %d sample images in the archive" % len(samples)
        print("  sample images present: %d" % len(samples))

        env = dict(os.environ)
        env["PYTHONPATH"] = str(project)
        env["XAVERIX_LLM"] = "0"  # verification must not need ExecuServe
        env["PYTHONWARNINGS"] = "ignore"

        # 2. The test suite must pass from the unzipped copy.
        print("  running the test suite from the clean unzip...")
        result = subprocess.run(
            [sys.executable, "-W", "ignore", "-m", "unittest", "discover", "-s", "tests"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        tail = (result.stderr or result.stdout).strip().splitlines()
        summary = tail[-1] if tail else "(no output)"
        ran = next((line for line in tail if line.startswith("Ran ")), "")
        if result.returncode != 0:
            return False, "tests failed in the clean unzip: %s" % summary
        print("  tests: %s %s" % (ran, summary))

        # 3. One end-to-end analysis must work with nothing but the archive.
        print("  running one end-to-end analysis...")
        script = (
            "import json, sys;"
            "sys.path.insert(0, '.');"
            "from app import pipeline;"
            "from tools import synth;"
            "import numpy as np;"
            "img, _ = synth.render_leaf(True, np.random.default_rng(5), severity_hint=0.45);"
            "r = pipeline.analyse(img, use_llm=False);"
            "json.dumps(r);"
            "print('verdict=%s source=%s total=%sms' % "
            "(r['verdict'], r['advisory']['source'], r['timings_ms']['total']))"
        )
        result = subprocess.run(
            [sys.executable, "-W", "ignore", "-c", script],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, "end-to-end analysis failed: %s" % result.stderr.strip()[-400:]
        print("  analysis: %s" % result.stdout.strip())

        return True, "verified from a clean unzip"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build the Xaverix submission zip")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    print("Xaverix %s - building submission" % config.APP_VERSION)

    files = collect()
    missing = check_required(files)
    if missing:
        print("\nBUILD FAILED. These required files are missing:", file=sys.stderr)
        for name in missing:
            print("  - %s" % name, file=sys.stderr)
        if "models/leaf_health.npz" in missing:
            print("\nRun: python tools/train.py", file=sys.stderr)
        return 2

    output = Path(args.out) if args.out else ROOT.parent / (
        "xaverix-submission-%s.zip" % time.strftime("%Y%m%d")
    )
    build(output, files)
    size_mb = output.stat().st_size / 1e6
    print("\n  %d files -> %s (%.2f MB)" % (len(files), output, size_mb))

    with zipfile.ZipFile(output) as archive:
        by_dir = {}
        for info in archive.infolist():
            top = info.filename.split("/")[1] if "/" in info.filename else "(root)"
            by_dir[top] = by_dir.get(top, 0) + 1
        for name in sorted(by_dir):
            print("    %-14s %3d" % (name, by_dir[name]))

    if args.skip_verify:
        print("\nverification skipped (--skip-verify)")
        return 0

    print("\nverifying...")
    ok, message = verify(output)
    if not ok:
        print("\nVERIFICATION FAILED: %s" % message, file=sys.stderr)
        return 3

    print("\n%s" % message)
    print("submission ready: %s" % output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
