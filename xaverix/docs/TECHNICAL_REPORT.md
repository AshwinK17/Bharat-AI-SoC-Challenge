# Xaverix — Technical Report

**Bharat AI-SoC Hackathon · Problem Statement PS1 — Agriculture & Food Security**
*Offline crop disease/pest detection and advisory for farmers*

| | |
|---|---|
| Project | Xaverix |
| Version | 1.0.0 |
| Team / Author | _(fill in before submission)_ |
| Target hardware | Arm64 Android phone (Termux) |
| Required technologies | Llama 3.2 1B Instruct · ExecuTorch |
| Date | 25 November 2026 |

> Every figure in this report was produced by the code in this repository and
> can be regenerated with `python tools/train.py`,
> `python scripts/benchmark.py` and `python -m unittest discover -s tests`.
> Figures measured on synthetic data are labelled as such throughout. Where a
> figure is an estimate rather than a measurement, it is marked.

---

## 1. Summary

Xaverix analyses a photograph of a crop leaf entirely on an Arm Android phone,
with no network access, and returns:

1. a **leaf-health verdict** — healthy, damaged, or *uncertain*;
2. a **severity estimate** with an explicit error interval;
3. a **severity-graded advisory** written on-device by Llama 3.2 1B through
   ExecuTorch, verified against a local knowledge base before it is shown;
4. a **probable-cause shortlist** to scout, labelled as indicative;
5. the same content in **English, Hindi or Tamil**.

Two design decisions shape the system:

- **The classifier may decline.** A model trained on procedurally generated
  imagery should not answer questions outside that distribution. Roughly 19% of
  genuine leaf images are declined as the cost of declining 82% of non-leaf
  images (§5).
- **The language model is not treated as a source of fact.** It restates a
  local knowledge base and nothing more, and a grounding guard verifies every
  output before display (§6).

## 2. Requirement compliance

| PS1 requirement | Status | Where |
|---|---|---|
| Analyse crop images | Met | `app/features.py`, `app/classifier.py` |
| Detect diseases | **Partially met** — detects damage presence, not species | §4, LIMITATIONS §1 |
| Detect pests | **Partially met** — pest damage detected; species not identified | §4, LIMITATIONS §1 |
| Classify severity levels | Met, as a calibrated heuristic with an error interval | §7 |
| Recommend treatment / intervention | Met | `data/knowledge_base.json`, `app/advisory.py` |
| Region-specific advisory | Met — 3 agro-climatic regions + default | `data/regions.json` |
| Multiple local languages | Met — English, Hindi, Tamil | `data/i18n.json`, §8 |
| Operate offline | Met, and enforced in-process + proven live | §9 |
| **Llama family model for advisory/reasoning** | **Met** | §6 |
| **Deployed on-device via ExecuTorch** | **Met** (Llama; see the caveat) | §6.1 |
| Arm-based platform | Met — Arm64 Android via Termux | §3 |
| Source code | Met | this repository |
| Application | Met — `server.py` + browser UI | §3 |
| Demonstration video | `docs/DEMO_VIDEO_SCRIPT.md` | — |
| Technical report | this document | — |

**The one caveat, stated plainly:** Llama runs through ExecuTorch. The *vision*
classifier does not — it is pure NumPy. Termux cannot install `torch` or
`executorch` at all, so a `.pte` vision model could not be executed in the app
process. See §6.1.

## 3. Architecture

```
                    ┌──────────── Android phone (Arm64) ────────────┐
                    │                                               │
  leaf photo ──────▶│  Chrome ──▶ server.py (Termux, stdlib HTTP)   │
                    │                    │                          │
                    │   ┌────────────────┴─────────────────┐        │
                    │   │ 1 features.py   150-D descriptor │        │
                    │   │ 2 classifier.py 3-gate cascade   │        │
                    │   │ 3 severity.py   lesion area      │        │
                    │   │ 4 knowledge.py  KB retrieval     │        │
                    │   │ 5 llm.py ───────────────────────┐│        │
                    │   │ 6 guard.py      verify output   ││        │
                    │   │ 7 i18n.py       en / hi / ta    ││        │
                    │   └─────────────────────────────────┼┘        │
                    │                                     │         │
                    │   offline_guard: loopback only ─────┤         │
                    │                                     ▼         │
                    │                    ExecuServe (Android app)   │
                    │                    Llama 3.2 1B Instruct      │
                    │                    INT4 / XNNPACK .pte        │
                    │                    ExecuTorch on Arm CPU      │
                    │                    127.0.0.1:8080/v1          │
                    └───────────────────────────────────────────────┘
```

### 3.1 The binding engineering constraint

The app is developed on x86 but must **run** on the phone. In Termux,
`python`, `python-numpy`, `python-pillow` and `unzip` install cleanly;
`pip install psutil` fails for lack of a compiler, and `torch`/`executorch`
cannot be installed at all.

**Therefore the entire runtime dependency set is: Python standard library +
NumPy + Pillow.** No Flask, no scikit-learn, no `requests`, no OpenCV, no
torch. The HTTP server is `http.server`, the Llama client is
`urllib.request`, the classifier's gradients were written by hand, and memory
is read from `/proc/self/status`.

The constraint is practical rather than stylistic: any additional dependency
would have to be compiled on the device. The ExecuTorch requirement is
satisfied *inside ExecuServe*, which is why the application itself does not
need it.

## 4. Vision: descriptor and classifier

### 4.1 Why handcrafted features rather than a CNN

| Option | Arm latency | Implementation cost | Supports novelty detection |
|---|---|---|---|
| Small CNN (im2col in NumPy) | 200-700 ms | convolution backward pass by hand | poorly — opaque embedding |
| Frozen random projections | same convolution cost | low | moderate |
| **Handcrafted descriptor + logistic head** | **~11 ms (x86), measured** | low | **well — low-dimensional, interpretable** |

The decisive criterion is the last column. The abstention cascade is what makes
a synthetic-data model usable, and a Mahalanobis distance is only a meaningful
novelty signal in a low-dimensional, physically interpretable space. The same
statistic over a learned embedding is not comparable.

### 4.2 The descriptor (150 dimensions, `feat-v3-150`)

HSV marginal histograms (16/8/8) · joint H×S histogram (8×4) · per-channel
moments and percentiles for R,G,B,H,S,V (36) · vegetation indices ExG, VARI,
(R−G)/(R+G) (12) · **damage-class area fractions** — necrotic, chlorotic,
water-soaked, pale, healthy-green (5) · uniform-LBP texture histogram (10) ·
Sobel gradient statistics (6) · lesion spatial structure from a 28×28 block
grid, including Gini concentration (10) · centre-versus-margin necrosis ratio
(2) · leaf geometry — area, aspect, fill, compactness (5).

`FEATURE_VERSION` is written into the model file and **asserted on load**. If
`features.py` changes, the model refuses to load rather than predicting from
features it was not fitted on. Train/inference skew is difficult to detect
from behaviour alone, so it is prevented by construction.

### 4.3 Leaf segmentation

An initial implementation defined the leaf region as any sufficiently saturated
pixel. This admits background: **brown soil also satisfies the necrosis colour
test**, so a photograph of a healthy leaf on bare earth measured roughly 30%
leaf damage. Against the renderer ground-truth lesion mask this appeared as a
constant **+25 to +30 point severity bias across all lesion types**, including
one whose true damage was 0.1%.

The fix (`app/features.py::leaf_mask`): locate the leaf by its **green core**,
then admit non-green pixels only where they are **enclosed** by that green
tissue — which is where real lesions sit. Hole filling runs on a coarse 14×14
grid, so the flood fill costs nothing, and the grid is eroded before use so
that boundary blocks of half-leaf/half-soil do not drag the background in.

Measured effect on a healthy leaf with no lesions at all:

| | before | after |
|---|---|---|
| necrotic fraction | 15.0% | **2.0%** |
| pale fraction | 11.4% | **3.0%** |
| healthy-green fraction | 66.3% | **94.6%** |
| leaf area vs true | 77% vs 41% | 56% vs 41% |
| mean severity correlation | r = 0.12 | **r = 0.40** |

`tests/test_vision.py::test_leaf_mask_excludes_brown_background` locks this in.

### 4.4 Training

Procedural leaves from `tools/synth.py`: polar-radius silhouettes with midrib
and lateral veins, multi-octave value noise, lighting gradients and specular
highlights, on procedural soil/sky/cloth/wood/skin/grass backgrounds. Infected
variants render eight lesion morphologies (concentric target rings, marginal
water-soaked patches, fine spots with chlorotic halos, rust pustules, powdery
bloom, serpentine miner trails, chewed margins, interveinal chlorosis), and
return the **exact lesion mask**, which gives free ground truth for severity
calibration (§7).

Augmentation runs *before* feature extraction and is aggressive on colour
(gamma 0.7-1.4, per-channel white balance ±8%, blur, downscale-and-back,
sensor noise, coarse quantisation) so the decision boundary leans on necrosis
and chlorosis rather than on the renderer's palette.

Head: standardiser → logistic regression, trained with hand-written Adam
(400 epochs, batch 128, L2 2e-3), then temperature-scaled on a held-out split.

| split | size |
|---|---|
| train | 1400 |
| validation | 500 |
| calibration | 500 |
| negatives (non-leaf) | 850 |
| severity pairs (unaugmented) | 600 |

### 4.5 Results — **on synthetic data only**

| metric | value |
|---|---|
| train accuracy (synthetic) | 0.941 |
| **validation accuracy (synthetic)** | **0.928** |
| accuracy on *accepted* leaves (synthetic) | **0.946** |
| expected calibration error, before → after scaling | 0.030 → 0.028 |
| fitted temperature | 0.902 |
| covariance for Mahalanobis | full (condition 1.5e2) |

**These are not real-world accuracies and must not be quoted as such.** No
PlantVillage, PlantDoc, Cassava or IP102 data was used. A real field
photograph is out of distribution by construction, which is the subject of the
next section.

## 5. Abstention — the safety core

### 5.1 The failure mode being addressed

For a classifier trained on synthetic imagery, the dominant risk is confident
misclassification of input it has never seen. A representative case is a leaf
photographed on ruled notebook paper: leaf-coloured pixels against a background
outside the training distribution. Without a cascade, such input receives a
verdict and a high confidence figure, which could lead to an unnecessary or
incorrect pesticide application.

### 5.2 Three gates

| gate | test | reason code |
|---|---|---|
| 0 — leaf presence | leaf area ≥ 0.20 of frame, green tissue ≥ 0.32 | `no_leaf_detected`, `not_leaf_like` |
| 1 — distance | Mahalanobis to class centroid ≤ 177.6 (tied covariance, 0.12 shrinkage) | `out_of_distribution` |
| 2 — confidence | calibrated confidence ≥ 0.60 | `low_confidence` |

All thresholds are **fitted** against a negative set of 850 generated non-leaf
scenes rather than chosen by hand. An initial hand-picked upper bound of 0.95
on leaf area declined a quarter of genuine leaves, since a leaf photographed
close up legitimately fills the frame; fitting removed that.

Every gate value is returned in the API response and rendered in the interface,
including on success, so a declined result can be inspected.

### 5.3 Measured rejection

| | rate |
|---|---|
| non-leaf scenes rejected | **82.4%** (gate 0: 69%, distance: 13%, confidence: 1%) |
| genuine leaves rejected | 19.0% (gate 0: 8%, distance: 7%, confidence: 4%) |

Per kind, the cases that matter most:

| negative kind | rejected |
|---|---|
| **lined paper** (the original failure) | **100%** |
| printed text, noise, brick, greyscale leaf, underexposed | 100% |
| overexposed, wood, flat colour | 90-95% |
| gradient, sky, skin | 83-90% |
| soil, cloth | 77-87% |
| scrambled leaf | 57% |
| multiple leaves / clutter | 35% |
| motion blur | 8% |
| uniform green foliage (grass) | ~0% |

The weaker cases are all derived from leaf imagery and therefore sit near the
training distribution. Grass is reported as `healthy` — "no disease signs
detected" — which is true of grass but does not answer the question asked. It
remains documented rather than closed, because every threshold change that
declined grass also declined a large share of genuine close-up leaf
photographs. The error direction is benign: it under-calls rather than
asserting a disease.

Declining one leaf in five is a deliberate trade: a declined result is
recoverable by retaking the photograph, which the interface explains, whereas a
confident incorrect verdict is not.

Displayed confidence is additionally **capped at 95%**, and no confidence is
reported at all when the classifier abstains.

## 6. Llama on-device, and the grounding guard

### 6.1 How ExecuTorch is satisfied

**ExecuServe** is an Android application that executes a model through the
**ExecuTorch** runtime on the phone's Arm CPU and exposes an OpenAI-compatible
HTTP endpoint on loopback. It hosts **Llama 3.2 1B Instruct, INT4, XNNPACK
export** at `http://127.0.0.1:8080/v1`.

`app/llm.py` is the client: `urllib.request` only, built with an empty
`ProxyHandler` so no system proxy can intercept, and it **refuses any
non-loopback URL** independently of the offline guard. Timeouts are 3 s for
the capability probe, 180 s for a background warm-up, 75 s for generation,
with a 60 s caller-enforced budget and one retry. Because a cold 1B model can
take 10-60 s to produce its first token, a warm-up thread starts at launch and
the first request waits at most 20 s before returning the knowledge-base
advisory labelled `kb_fallback_llm_cold`, so the first request does not appear
to stall.

**Scope caveat:** the vision classifier is *not* an ExecuTorch model. Termux
cannot install `torch` or `executorch`, so the vision path was designed to
require only NumPy, and its weights ship as a NumPy `.npz`. Llama is the
ExecuTorch component. This is stated here, in the README and in LIMITATIONS,
because the distinction is material to the problem statement.

### 6.2 Why generated output is not displayed directly

During evaluation, asked what early blight of tomato is, this model attributed
the disease to a non-existent bacterial genus. The actual cause is the fungus
***Alternaria solani***.

The model is therefore never asked what a disease is. The prompt supplies the
facts and constrains the task: *use ONLY the facts in FACTS; add no disease,
organism, chemical, dose or number that is not in FACTS*, four fixed headings,
a 110-word cap, temperature 0.2. The assistant turn is **prefilled** with the
first heading, which fixes the output structure at lower token cost than a
few-shot exemplar. The complete prompt is about 2,900 characters (~730 tokens),
trimmed because prefill is not free on a device CPU.

### 6.3 The guard

`app/guard.py` verifies every generation before display. Safety lives in the
hard checks; the soft score only catches wholesale drift, and is deliberately
lenient so that ordinary paraphrase is not rejected for no safety gain.

**Hard checks — any one rejects:**

| check | what it stops |
|---|---|
| `NUMBER_NOT_IN_KB` | invented doses and intervals. Devanagari and Tamil digits are normalised to ASCII first; list markers are stripped so "1." is not read as a claim |
| `CHEMICAL_NOT_IN_KB` | products not listed for this diagnosis — a ~150-name lexicon **plus** a morphology regex (`*conazole`, `*strobin`, `*mectin`, `*zeb`, …) that catches invented-but-plausible names |
| `ORGANISM_NOT_IN_KB` | a pathogen-class word followed by a capitalised name, genus-shaped single words, and Latin binomials absent from the entry. This is the check that catches a fabricated causal agent |
| `PATHOGEN_TYPE_MISMATCH` | calling a fungus a bacterium, including a named pathogen paired with the wrong kingdom word nearby |
| `BANNED_PATTERN` | guarantees, "will cure", 100% claims, prices, subsidies, URLs, phone numbers, consumption advice, cloud references |
| `FORMAT`, `LENGTH` | malformed or runaway output |
| `WRONG_CROP`, `WRONG_REGION`, `MISSING_ACTION` | drift off the farmer's actual situation, or advice with no action in it |

**Soft scores:** content-token support ≥ 0.70 against the entry plus a ~400
word generic agriculture whitelist; bigram overlap ≥ 0.15.

Measured on a faithful rewrite: support 0.95, bigram overlap 0.51, accepted.
On the three invalid fixtures: all rejected, with the fabricated-pathogen case
flagged specifically by `ORGANISM_NOT_IN_KB`.

**On rejection:** one retry at temperature 0.0 with the violated rules
appended as explicit prohibitions, then the deterministic knowledge-base
advisory. Identical response shape, zero hallucination risk.

`GET /api/guard-demo` applies the guard to all three fixtures on request, so
its behaviour can be inspected independently of a live generation.

**What the guard does not do:** it bounds *actionable* error — quantities,
products, organisms — not all error. Fluent but uninformative text can pass.

### 6.4 Provenance

Each advisory carries exactly one honest label, rendered as a coloured chip:

| source | meaning |
|---|---|
| `llama_grounded` | Llama wrote it; the guard accepted it |
| `kb_fallback_guard_reject` | Llama wrote it; the guard rejected it (codes shown) |
| `kb_fallback_llm_unavailable` / `_timeout` / `_cold` | endpoint not usable |
| `kb_only_llm_disabled` | Llama switched off, or the system abstained |

The fallback is never silent; a rejection is reported rather than substituted
without notice.

## 7. Severity

Calibrated fraction of leaf area that is necrotic, water-soaked, pale or
chlorotic (the last weighted 0.5 for partial loss of function), computed in
pure NumPy on the masks already produced by the descriptor.

Calibration is fitted by least squares against the renderer exact lesion mask,
so the error figure is measured rather than assumed. One third of the
calibration set is healthy leaves: fitting on damaged leaves alone left the
intercept unconstrained at zero damage, where it settled at **17.6**, reporting
a healthy leaf as 17.6% damaged.

| metric | value |
|---|---|
| calibration | severity = 0.837 × raw + 8.32 |
| mean absolute error (synthetic ground truth) | **12.0 points** |
| correlation with true lesion fraction | r = 0.444 |
| reported interval | ±(5 + 0.2 × severity) points |

Bands: trace < 10 ≤ mild < 25 ≤ moderate < 50 ≤ severe, which drive the
urgency line in the advisory.

Severity is **suppressed entirely when the classifier abstains**: a leaf
region we do not trust yields an area denominator we cannot use.

Known bias by lesion type: within ±3 points for target-ring, fine-spot, rust,
miner and chewed-edge lesions; **powdery mildew under-read by ~24 points** and
**interveinal chlorosis by ~17 points**. Widening the brightness bound to
catch powdery mildew tripled false positives on healthy leaves (3.8% → 12.6%)
while barely improving detection (5.3% → 7.1%), so the low-false-positive
threshold was kept and the gap documented instead.

## 8. Multilingual advisory

English, Hindi and Tamil, from **human-authored strings** in `data/i18n.json`.
The test suite fails if any string is missing in any language.

Generation is confined to the English layer, for three reasons:

1. Hindi and Tamil output from a 1B INT4 model is weak, and an incorrect dose
   in a translated string is a safety defect, not a wording defect.
2. Tamil costs roughly 3-5x the tokens under this tokenizer, on a call that
   already takes tens of seconds.
3. The guard verifies output by overlap against an English knowledge base and
   cannot verify a translation, so machine translation would bypass the check
   that makes generated text safe to display.

The generated paragraph appears as a clearly labelled additional English note;
the structured advisory is rendered in the selected language.

## 9. Offline operation

`app/offline_guard.py` is installed as the **first statement** in `server.py`,
before any import that could open a socket. It gates `connect`, `connect_ex`,
`sendto`, `create_connection` **and `getaddrinfo`**, leaving
`bind`/`listen`/`accept`/`recv`/`send` untouched so `http.server` works.

Blocking DNS is the stronger claim: **the process cannot even resolve a public
hostname.** Loopback (`127.0.0.0/8`, `::1`) stays open, which is how ExecuServe
is reached. LAN is blocked by default.

`GET /api/offline-proof` demonstrates it with live probes:

```json
{
  "blocked": true,
  "probes": [
    {"target": "api.openai.com:443", "api": "socket.create_connection", "result": "blocked"},
    {"target": "8.8.8.8:53",         "api": "socket.create_connection", "result": "blocked"},
    {"target": "huggingface.co",     "api": "socket.getaddrinfo",       "result": "blocked"},
    {"target": "127.0.0.1:8000",     "api": "socket.create_connection", "result": "allowed"}
  ]
}
```

The UI also forbids external resources through a strict Content-Security-Policy
(`default-src 'self'`), and the page loads no web fonts, no CDN scripts and no
remote images.

**Scope:** this guard constrains the Xaverix Python process only. It makes no
claim about other applications on the device, including ExecuServe (which is
configured so that only the local device may connect). Aeroplane mode is the
end-to-end verification, and `/api/offline-proof` records this in its
`scope_note` field.

## 10. Performance

### 10.1 Measured — x86 development host

`python scripts/benchmark.py --iterations 20`, Windows AMD64, Python 3.11.9,
NumPy 2.4.2.

| stage | p50 (ms) | p95 (ms) |
|---|---|---|
| JPEG decode + resize | 0.39 | 0.84 |
| descriptor (150-D) | 10.06 | 12.30 |
| classifier + 3 gates | 0.14 | 0.21 |
| severity | 0.79 | 0.82 |
| knowledge retrieval | 0.00 | 0.07 |
| **total (vision path)** | **11.38** | — |

Model file: 82 KB. The advisory stage is dominated by Llama generation, which
is measured separately with `--llm`.

### 10.2 Arm device — to be filled from the phone

Run on the phone and paste the result:

```bash
TARGET_DEVICE="<phone model> (Arm64 Android, Termux)" \
  python scripts/benchmark.py --iterations 25 --llm
```

| stage | p50 (ms) | p95 (ms) |
|---|---|---|
| decode | _TBD_ | _TBD_ |
| descriptor | _TBD_ | _TBD_ |
| classifier | _TBD_ | _TBD_ |
| severity | _TBD_ | _TBD_ |
| **total vision** | _TBD_ | — |
| Llama advisory (warm) | _TBD_ | — |
| peak RSS | _TBD_ | — |
| device | _TBD_ | |

`TARGET_DEVICE` is **required**; without it the benchmark labels its output
`"unspecified"` so development-host figures cannot be mistaken for device
figures. Memory is read from `/proc/self/status`, since `psutil` cannot be
built in Termux.

### 10.3 Energy (estimate)

`results/benchmark.json` reports energy as an assumed 2.0 W average CPU draw
multiplied by measured wall-clock latency, with `"measured": false` and the
assumption recorded in the file. No power monitor was used.

## 11. Verification

```
python -m unittest discover -s tests          # 83 tests
```

All 83 pass. Coverage of note:

- the offline guard blocks `connect` **and** DNS, permits loopback, and
  `/api/offline-proof` reports `blocked: true`
- the grounding guard rejects a fabricated pathogen, an invented dose for a
  real product, a non-existent but plausible fungicide, guarantees and prices,
  a disease claim on a healthy leaf, and the wrong region — and **accepts** a
  faithful rewrite
- the classifier abstains on noise, blank frames, printed text and the
  **lined-paper photograph**, and never reports 100% confidence
- the leaf mask does not read brown background as necrosis (§4.3 regression)
- severity rises monotonically with damage and is suppressed when abstaining
- `llm.py` refuses non-loopback endpoints; every failure path degrades to the
  knowledge base
- every i18n string exists in all three languages
- the pipeline result is JSON-serialisable (no NumPy scalars leak out)
- **the full model path is exercised end to end** against a loopback stand-in
  for the ExecuServe endpoint (`tools/mock_executorch_server.py`), covering: a
  faithful generation accepted and labelled `llama_grounded`; an unfaithful
  generation rejected and the knowledge-base text served instead, with the
  unsupported content verified absent from the displayed advisory; an
  off-topic generation rejected for low support; a wrong API key and a
  generation timeout both degrading cleanly; the warm-up handshake; the
  constrained prompt and prefill arriving intact; and the offline guard still
  permitting the loopback call. ExecuServe runs only on the device, so this is
  how the integration is verified off-device.

`python build_submission.py` additionally extracts its own output into a
temporary directory and runs the suite plus one end-to-end analysis there, and
**fails the build if `models/leaf_health.npz` or the sample images are
missing**, so a release archive cannot omit the weights it depends on.

## 12. Assessment

**Delivered:** an offline application running on Arm hardware; Llama 3.2 1B
generating advisories on-device through ExecuTorch; a grounding guard that
rejects fabricated pathogens, invented doses and unsupported claims while
accepting faithful paraphrase; an abstention cascade that declines
out-of-distribution input instead of classifying it; three languages; and a
disclosure trail in which every metric is reproducible from the code.

**Not delivered:** species-level disease or pest identification; any real-world
accuracy, the training data being synthetic; a segmentation-quality severity
model; measured power consumption; a Hailo accelerator path; and ExecuTorch
execution of the *vision* model.

**Next step.** Retraining the descriptor and head on PlantVillage and IP102
with the same abstention machinery would be the highest-value extension. The
descriptor, gates, guard, knowledge base, server and interface are all
dataset-independent; only `tools/synth.py` and the training invocation would
be replaced. The supportable claim today is that the *pipeline* is validated
and the *model* is not.

## 13. Repository map

| path | purpose |
|---|---|
| `server.py` | stdlib HTTP server; installs the offline guard first |
| `app/features.py` | 150-D descriptor, shared by training and inference |
| `app/classifier.py` | logistic head + three-gate abstention cascade |
| `app/severity.py` | calibrated lesion-area severity |
| `app/knowledge.py` | KB load, validation, derived guard allow-sets |
| `app/prompt.py` | the caged Llama prompt |
| `app/llm.py` | ExecuServe client, `urllib` only, loopback only |
| `app/guard.py` | grounding verification and `/api/guard-demo` fixtures |
| `app/advisory.py` | orchestration, provenance, deterministic fallback |
| `app/offline_guard.py` | loopback-only socket and DNS policy |
| `app/pipeline.py` | stage sequencing and the per-stage trace |
| `tools/synth.py` | procedural leaves, lesions and the negative set |
| `tools/train.py` | training, calibration, threshold fitting |
| `scripts/benchmark.py` | per-stage latency and memory |
| `build_submission.py` | zip, then verify from a clean unzip |
| `docs/LIMITATIONS.md` | every known weakness, with measured numbers |
| `docs/PHONE_SETUP.md` | ExecuServe and Termux setup |
