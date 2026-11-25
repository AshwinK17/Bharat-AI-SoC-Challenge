"""Central configuration for Xaverix.

Everything tunable lives here so the rest of the code never reads os.environ
directly. Paths are resolved relative to the project root, so the app works
from any working directory (important: in Termux you may launch it from $HOME).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "Xaverix"
APP_TAGLINE = "Offline crop leaf health & advisory"
APP_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
WEB_DIR = ROOT / "web"
RESULTS_DIR = ROOT / "results"

KB_PATH = DATA_DIR / "knowledge_base.json"
I18N_PATH = DATA_DIR / "i18n.json"
CHEM_LEXICON_PATH = DATA_DIR / "chem_lexicon.json"
REGIONS_PATH = DATA_DIR / "regions.json"
MODEL_PATH = MODELS_DIR / "leaf_health.npz"

# ---------------------------------------------------------------- vision
IMAGE_SIZE = 224          # model input edge, px
SEVERITY_SIZE = 112       # severity is computed at half res for stability
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# Displayed confidence is capped: a synthetic-data model must never show 100%.
CONFIDENCE_DISPLAY_CAP = 0.95

SYNTHETIC_CAVEAT = (
    "Trained on procedurally generated imagery; not validated on real "
    "field photographs."
)

# ---------------------------------------------------------------- Llama / ExecuServe
def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


LLM_ENABLED = _env_flag("XAVERIX_LLM", "1")
LLM_URL = os.environ.get("XAVERIX_LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_KEY = os.environ.get("XAVERIX_LLM_KEY", "")
LLM_MODEL = os.environ.get("XAVERIX_LLM_MODEL", "llama-3.2-1b-instruct")

LLM_PROBE_TIMEOUT = 3.0        # GET /v1/models
LLM_PROBE_TTL = 30.0           # cache probe result this long
LLM_WARMUP_TIMEOUT = 180.0     # cold model load on a phone is slow
LLM_GEN_TIMEOUT = 75.0
LLM_BUDGET_MS = 60_000         # caller-enforced wall clock for one advisory
LLM_COLD_WAIT = 20.0           # how long request #1 waits for warm-up
LLM_MAX_TOKENS = 200
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 0.9

# ---------------------------------------------------------------- offline policy
ALLOW_LAN = _env_flag("XAVERIX_ALLOW_LAN", "0")

# ---------------------------------------------------------------- guard thresholds
GUARD_SUPPORT_MIN = 0.70
GUARD_BIGRAM_MIN = 0.15
GUARD_BIGRAM_WARN = 0.35
GUARD_MIN_WORDS = 25
GUARD_MAX_WORDS = 200
GUARD_MAX_CHARS = 1400

LANGUAGES = ("en", "hi", "ta")
DEFAULT_LANGUAGE = "en"

TARGET_DEVICE = os.environ.get("TARGET_DEVICE", "").strip()


def runtime_info() -> dict:
    """Runtime configuration, exposed at /api/config for inspection."""
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "image_size": IMAGE_SIZE,
        "languages": list(LANGUAGES),
        "target_device": TARGET_DEVICE or "unspecified",
        "allow_lan": ALLOW_LAN,
        "llm": {
            "enabled": LLM_ENABLED,
            "endpoint": LLM_URL,
            "model": LLM_MODEL,
            "key_set": bool(LLM_KEY),
        },
    }


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
