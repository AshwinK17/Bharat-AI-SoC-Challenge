# Xaverix

**Offline crop leaf health assessment and treatment advisory, running entirely on an Arm Android phone.**

Xaverix analyses a photograph of a crop leaf and returns whether it shows
disease or pest damage, how much of the leaf area is affected, and a
severity-graded treatment advisory in English, Hindi or Tamil. The advisory
text is generated on the device by Llama 3.2 1B Instruct through ExecuTorch,
and is verified against a local knowledge base before it is displayed.

No network connection is used at any point, and the process is prevented from
making one.

Built for Bharat AI-SoC, problem statement PS1 (Agriculture & Food Security).

---

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Running on an Android phone](#running-on-an-android-phone)
- [How it works](#how-it-works)
- [Two design decisions worth explaining](#two-design-decisions-worth-explaining)
- [Results](#results)
- [Verifying the claims](#verifying-the-claims)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Scope and limitations](#scope-and-limitations)
- [Documentation](#documentation)
- [License](#license)

## What it does

| Capability | Detail |
|---|---|
| Leaf health verdict | `healthy`, `infected`, or `uncertain` when the input cannot be assessed reliably |
| Severity | Affected leaf area as a percentage, with an explicit error interval |
| Treatment advisory | Four sections: what is happening, what to do now, an organic option, prevention |
| Probable causes | A severity-graded shortlist to inspect, each with its distinguishing symptom, organic and chemical option, dose, interval and pre-harvest interval |
| Region-specific guidance | Tamil Nadu, Punjab, Maharashtra, plus a general default |
| Languages | English, Hindi, Tamil |
| Offline | Enforced in-process; outbound connections and DNS resolution are blocked |

## Quick start

Requires Python 3.9 or newer, NumPy and Pillow. There are no other runtime
dependencies.

```bash
pip install -r requirements.txt

python tools/train.py          # ~3 min: generates data, trains, writes models/ and samples/
python server.py --port 8000   # open http://localhost:8000
```

`tools/train.py` writes `models/leaf_health.npz` (82 KB) and a set of sample
images. Both are included in release archives, so no training is needed on the
target device.

To exercise the language-model path without an Android device, run the
development stand-in for the ExecuServe endpoint in a second terminal:

```bash
python tools/mock_executorch_server.py --port 8080 --mode faithful
XAVERIX_LLM=1 XAVERIX_LLM_KEY=test python server.py --port 8000
```

## Running on an Android phone

Two applications run side by side on the device and communicate over loopback.
No laptop, USB cable or Android build toolchain is required.

1. **ExecuServe** (from Google Play) hosts Llama 3.2 1B Instruct — the INT4 /
   XNNPACK CPU export — through ExecuTorch, and exposes an OpenAI-compatible
   endpoint on `http://127.0.0.1:8080/v1`.
2. **Termux** (from F-Droid) runs this application:

```bash
pkg install python python-numpy python-pillow unzip
unzip xaverix.zip && cd xaverix

XAVERIX_LLM=1 \
XAVERIX_LLM_KEY=<key from ExecuServe> \
TARGET_DEVICE="<phone model> (Arm64 Android, Termux)" \
python server.py --port 8000
```

Then open `http://localhost:8000` in the phone's browser.

Full instructions, including which model variant to download and a
troubleshooting table, are in **[docs/PHONE_SETUP.md](docs/PHONE_SETUP.md)**.

## How it works

```
                    ┌──────────── Android phone (Arm64) ────────────┐
                    │                                               │
  leaf photo ──────▶│  browser ──▶ server.py (Termux, stdlib HTTP)  │
                    │                    │                          │
                    │   ┌────────────────┴─────────────────┐        │
                    │   │ 1  features.py    150-D descriptor│       │
                    │   │ 2  classifier.py  3-gate cascade  │       │
                    │   │ 3  severity.py    lesion area     │       │
                    │   │ 4  knowledge.py   fact retrieval  │       │
                    │   │ 5  llm.py ───────────────────────┐│       │
                    │   │ 6  guard.py       verify output  ││       │
                    │   │ 7  i18n.py        en / hi / ta   ││       │
                    │   └──────────────────────────────────┼┘       │
                    │                                      │        │
                    │   offline_guard: loopback only ──────┤        │
                    │                                      ▼        │
                    │                     ExecuServe (Android app)  │
                    │                     Llama 3.2 1B Instruct     │
                    │                     INT4 / XNNPACK            │
                    │                     ExecuTorch on Arm CPU     │
                    └───────────────────────────────────────────────┘
```

Every `/api/analyze` response carries a per-stage trace with timings and all
gate values, so any figure in the interface can be traced to the stage that
produced it.

**Runtime dependencies are the Python standard library, NumPy and Pillow.**
Termux installs those as system packages; anything else would need to be
compiled on the device. The HTTP server is `http.server`, the model client is
`urllib.request`, the classifier is trained with gradients implemented in
NumPy, and memory statistics are read from `/proc/self/status`.

ExecuTorch is not a dependency of this process. It runs inside ExecuServe,
which is why the application itself needs nothing beyond NumPy.

> **Note on scope:** *Llama* runs through ExecuTorch on the Arm CPU. The
> *vision* classifier does not — it is NumPy, and its weights ship as an
> `.npz`. Termux cannot install `torch` or `executorch`, so the vision path was
> designed to require neither.

## Two design decisions worth explaining

### The classifier can decline to answer

The classifier is trained on procedurally generated leaf imagery, so arbitrary
real-world input is out of distribution by construction. Rather than return a
confident verdict regardless, three gates can withhold one:

| Gate | Criterion |
|---|---|
| 0 — leaf presence | a green leaf region covering at least 20% of the frame |
| 1 — novelty | Mahalanobis distance to the training distribution within a fitted threshold |
| 2 — confidence | calibrated probability at or above 0.60 |

Any gate firing returns `uncertain`, with no verdict and no severity figure,
plus guidance on retaking the photograph. All three thresholds are fitted
against a generated negative set rather than chosen by hand.

Measured: **82% of non-leaf scenes are declined**, at a cost of **19% of
genuine leaf images** also being declined. Displayed confidence is capped below
100%, since a model validated only on synthetic data should not present
absolute certainty.

### Generated text is verified before it is shown

A 1-billion-parameter model is not a reliable source of agronomic fact. During
evaluation it attributed early blight of tomato to a non-existent bacterial
genus; the actual cause is the fungus *Alternaria solani*. Acting on that would
mean buying the wrong class of product.

The model is therefore never asked what a disease is. It receives facts from
the local knowledge base and is asked only to restate them for one user, in a
fixed four-section structure. `app/guard.py` then checks the result:

- every numeric quantity must appear in the knowledge-base entry
- every agro-input name must be one the entry lists, including invented but
  plausible names caught by morphology (`*conazole`, `*strobin`, `*mectin`, …)
- no organism or pathogen class outside the entry
- no guarantees, prices, links, or consumption advice
- correct structure and length

On failure the request is retried once at temperature 0 with the violated rules
restated, then falls back to a deterministic advisory composed directly from the
knowledge base. **The fallback is never silent:** the response carries the
source label and the violation codes, and the interface displays both.

## Results

Measured on held-out **synthetic** data. These are not real-world accuracies —
see [Scope and limitations](#scope-and-limitations).

| Metric | Value |
|---|---|
| Validation accuracy | 0.928 |
| Accuracy on accepted leaves | 0.946 |
| Expected calibration error, after temperature scaling | 0.028 |
| Non-leaf scenes declined | 82.4% |
| Genuine leaves declined | 19.0% |
| Severity mean absolute error | 12.0 points |
| Severity correlation with ground truth | r = 0.44 |
| Vision pipeline latency, x86 p50 | 11.4 ms |
| Model file size | 82 KB |

Regenerate with `python tools/train.py` and `python scripts/benchmark.py`. Arm
device figures are recorded in
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) §10.

## Verifying the claims

| Claim | Check |
|---|---|
| It operates offline | `curl -s localhost:8000/api/offline-proof` → `"blocked": true`, with DNS resolution also blocked and loopback still permitted |
| Generated text is verified | `curl -s localhost:8000/api/guard-demo` → three invalid outputs, each rejected, with violation codes |
| The model is running on-device | `curl -s localhost:8000/api/config \| grep usable` → `"usable": true` |
| The figures are reproducible | `python tools/train.py`, `python scripts/benchmark.py`, `python -m unittest discover -s tests` |
| Releases contain the model | `python build_submission.py` fails the build if weights or samples are absent |

## HTTP API

| Route | Purpose |
|---|---|
| `GET /` | single-page application |
| `GET /api/config` | runtime, model and endpoint status, plus disclosures |
| `GET /api/samples` | bundled sample images |
| `GET /api/offline-proof` | live outbound connection and DNS probes |
| `GET /api/guard-demo` | the grounding guard applied to invalid outputs |
| `GET /api/i18n?lang=ta` | interface strings for one language |
| `POST /api/analyze` | `{image, crop, region, language}` → verdict, severity, advisory, trace |

`POST /api/analyze` accepts either JSON with a base64 or data-URL `image`
field, or raw image bytes with `X-Xaverix-*` headers.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `XAVERIX_LLM` | `1` | enable the generated advisory |
| `XAVERIX_LLM_URL` | `http://127.0.0.1:8080/v1` | model endpoint. Loopback only; any other host is refused |
| `XAVERIX_LLM_KEY` | — | endpoint API key |
| `XAVERIX_LLM_MODEL` | `llama-3.2-1b-instruct` | model identifier as the endpoint reports it |
| `XAVERIX_ALLOW_LAN` | `0` | permit private-range addresses |
| `TARGET_DEVICE` | — | recorded in `results/benchmark.json`; set this when benchmarking on a device |

## Project layout

```
server.py                 stdlib HTTP server; installs the offline guard first
app/
  config.py               paths, thresholds, environment configuration
  features.py             150-D descriptor, shared by training and inference
  classifier.py           logistic head and three-gate abstention cascade
  severity.py             calibrated lesion-area severity
  knowledge.py            knowledge-base loading, validation, derived allow-sets
  prompt.py               constrained prompt construction
  llm.py                  ExecuServe client; urllib only, loopback only
  guard.py                grounding verification
  advisory.py             orchestration, provenance, deterministic fallback
  offline_guard.py        loopback-only socket and DNS policy
  pipeline.py             stage sequencing and the response trace
  i18n.py                 English, Hindi, Tamil rendering
data/
  knowledge_base.json     the single source of agronomic fact
  i18n.json               interface and advisory strings
  chem_lexicon.json       agro-input names the guard watches for
  regions.json            regions, crops, state names
models/leaf_health.npz    weights, calibration, thresholds, training metadata
tools/
  synth.py                procedural leaf renderer and negative-set generator
  train.py                training, calibration, threshold fitting
  mock_executorch_server.py   development stand-in for the model endpoint
scripts/benchmark.py      per-stage latency and memory
web/                      single-page interface; no framework, no CDN
tests/                    83 standard-library unittest tests
docs/                     technical report, limitations, setup, demo script
build_submission.py       builds a release archive and verifies it
```

## Tests

```bash
python -m unittest discover -s tests
```

83 tests, standard library only. Coverage includes: the offline guard blocking
connections and DNS while permitting loopback; the grounding guard rejecting a
fabricated pathogen, an invented dose, a non-existent fungicide and
unsupported claims while accepting faithful paraphrase; the classifier
declining on noise, blank frames, printed text and a leaf on ruled paper, and
never reporting 100% confidence; the leaf-segmentation regression; severity
monotonicity and suppression when abstaining; the full model path end to end
against a loopback stand-in; translation completeness in all three languages;
and JSON-serialisability of the response.

## Scope and limitations

This is a prototype, and its boundaries are documented rather than implied.
The main points:

- **Binary detection.** It determines whether damage is present, not which
  disease or pest caused it. Probable causes are offered as a shortlist to
  inspect, labelled as indicative.
- **Synthetic training data.** No PlantVillage, PlantDoc, Cassava or IP102 data
  is used, and **no real-world accuracy is claimed.**
- **Severity is a heuristic.** A calibrated colour measure with a ±interval,
  not a segmentation model. Powdery mildew and interveinal chlorosis are
  under-read; the measured bias per lesion type is tabulated in the docs.
- **Uniform green foliage** (for example a photograph of grass) is not
  declined and is reported as `healthy`. The error direction is benign, but it
  is not an answer to the question asked.
- **Energy figures are estimates**, with the assumption recorded alongside
  them, not measurements.
- **The offline guard binds this process only.** Aeroplane mode remains the
  end-to-end verification.

Every known weakness, with measured numbers, is in
**[docs/LIMITATIONS.md](docs/LIMITATIONS.md)**. Read it before interpreting the
accuracy figures above.

## Documentation

| Document | Contents |
|---|---|
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | architecture, method, measured results, requirement compliance |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | every known weakness, with numbers |
| [docs/PHONE_SETUP.md](docs/PHONE_SETUP.md) | ExecuServe and Termux setup, with troubleshooting |
| [docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md) | demonstration shot list |

## License

MIT. See [LICENSE](LICENSE).

Agronomic content in `data/knowledge_base.json` is compiled from ICAR and
Tamil Nadu Agricultural University extension guidance, with product dosages
following Central Insecticides Board and Registration Committee label claims.
It is provided for reference and does not replace advice from a qualified
agriculture extension officer.
