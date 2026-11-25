# Xaverix — known limitations

Every figure in this document was measured by the code in this repository.
Reproduce them with:

```
python tools/train.py            # prints and stores every metric
python -m unittest discover -s tests
python scripts/benchmark.py
```

---

## 1. It is a binary detector, not a diagnosis

Xaverix answers one question: **does this leaf show disease or pest damage?**
It does **not** identify which disease or which pest.

PS1 asks for disease *and* pest detection. This prototype covers damage
presence and severity. In place of species identification, the response returns
a **severity-graded shortlist of probable causes** from the knowledge base,
each with its distinguishing symptom, labelled *indicative — inspect for these,
not a diagnosis*. A single species name produced by a model trained on
procedurally generated imagery would not be supportable.

`app/classifier.py` keeps a multi-class head as a drop-in if named classes are
added later.

## 2. The training data is synthetic

There is no PlantVillage, PlantDoc, Cassava or IP102 data in this project. The
classifier is trained entirely on procedurally generated leaves from
`tools/synth.py`.

**No real-world accuracy is claimed.** The figures reported by the trainer are
on held-out *synthetic* images:

| metric | value |
|---|---|
| validation accuracy (synthetic) | ~0.93 |
| accuracy on accepted leaves (synthetic) | ~0.95 |
| expected calibration error after temperature scaling | ~0.03 |

A real field photograph is out of distribution by construction. This is the
reason for the abstention cascade, and why the interface carries the
synthetic-data caveat on every result.

## 3. What the abstention cascade does and does not catch

Three gates run before any verdict: leaf presence, Mahalanobis distance from
the training distribution, and calibrated confidence. Measured rejection rates
on the generated negative set (`gate_reject_per_negative_kind` in the model
file):

| negative kind | rejected | notes |
|---|---|---|
| leaf on ruled paper | **100%** | see §4 |
| printed text, noise, brick, greyscale leaf, underexposed | 100% | |
| overexposed, wood, flat colour | 90-95% | |
| gradient, sky, skin | 83-90% | |
| soil, cloth | 77-87% | |
| scrambled leaf | ~57% | |
| multiple leaves / clutter | ~35% | **weak** |
| motion blur | ~8% | **weak** |
| grass / uniform green foliage | **~0%** | **weak** — see below |

The three weak cases are all derived from leaf imagery, so they sit close to
the training distribution rather than outside it:

- **Motion blur** — a blurred leaf is still a leaf, so a verdict is
  reasonable, though the severity estimate degrades.
- **Multiple leaves** — foliage rather than one leaf. The leaf-area
  denominator becomes meaningless, so severity is unreliable here.
- **Uniform green foliage (grass)** — not declined at all. It is reported as
  `healthy`, i.e. "no disease signs detected", which is true of grass but is
  not an answer to the question asked. The error direction is benign: it
  under-calls rather than asserting a disease. It remains a documented
  limitation rather than being closed by tightening the gates, because every
  threshold change that declined grass also declined a large share of genuine
  close-up leaf photographs.

The cost of the cascade: **about 19% of genuine synthetic leaves are also
declined** (8% leaf presence, 7% novelty, 4% confidence). This is a deliberate
trade. A declined leaf is recoverable — the user retakes the photograph, and
the interface explains how. A confident incorrect verdict is not.

## 4. Confident misclassification of out-of-distribution input

The dominant failure mode for a model trained on synthetic data is to classify
unrelated input with high confidence. A representative case is a leaf
photographed on ruled notebook paper, which combines leaf-coloured pixels with
a background the model has never seen.

The negative set in `tools/synth.py` reproduces that scene explicitly
(`render_negative(kind="lined_paper")`), the thresholds are fitted against it,
and `tests/test_vision.py::test_abstains_on_the_lined_paper_photograph` fails
the build on regression. Displayed confidence is additionally capped below
100%, since a model validated only on synthetic data should not present
absolute certainty.

## 5. Severity is a colour heuristic

Severity is the calibrated fraction of leaf area that is necrotic,
water-soaked, pale or chlorotic. It is **not** a segmentation model.

Measured against the renderer exact lesion mask, so the ground truth is known
rather than assumed:

| metric | value |
|---|---|
| mean absolute error | ~12 severity points |
| correlation with true lesion fraction | r ≈ 0.44 |
| calibration | severity = 0.84 × raw + 8.3 |

The reported figure carries an interval of ±(5 + 0.2 × severity) points, and
severity is **suppressed entirely** whenever the classifier abstains: an
unreliable leaf region gives an unusable area denominator.

Per-lesion-type bias, measured:

| lesion type | bias |
|---|---|
| target-ring, fine-spot, rust, miner, chewed-edge lesions | within ±3 points |
| powdery mildew | **under-read by ~24 points** |
| interveinal chlorosis | **under-read by ~17 points** |

Powdery mildew is under-detected because the bloom brightens the leaf without
saturating it, and widening the brightness bound to catch it tripled false
positives on healthy leaves (3.8% → 12.6%) while barely improving detection
(5.3% → 7.1%). The low-false-positive threshold was kept.

Other known failure modes, all unquantified on real photographs: shadows and
dark soil read as necrosis; red and purple cultivars read as damaged; dry leaf
margins read as necrosis; specular highlights read as pale tissue; overlapping
leaves break the leaf-area denominator.

A healthy leaf reads about 8% damage (the calibration intercept) rather than
0%. That lands in the "trace" band, which the advisory treats as "no action
needed".

## 6. Model reliability, and the scope of the guard

The advisory is written on-device by Llama 3.2 1B Instruct (INT4, XNNPACK)
through ExecuTorch, hosted by ExecuServe on loopback.

The model is **not reliable as a source of fact**. During evaluation, asked
what early blight of tomato is, it attributed the disease to a non-existent
bacterial genus; the actual cause is the fungus *Alternaria solani*. The model
is therefore never asked what a disease is. It receives the facts and may only
restate them, and `app/guard.py` verifies the result before display.

**What the guard enforces** (any failure rejects the output): every number must
appear in the knowledge-base entry; every agro-input name must be one the entry
lists, including invented-but-plausible names caught by morphology; no organism
outside the entry; no pathogen class the entry does not support; no guarantees,
prices, links or consumption advice; the four fixed headings; a word cap.

**What the guard does not enforce:** it bounds *actionable* error — numbers,
products, organisms — not all error. Fluent but uninformative text that
introduces no quantity and names no product can pass. The soft overlap score
detects wholesale topic drift, not subtle inaccuracy.

`GET /api/guard-demo` applies the guard to three invalid outputs drawn from
observed error classes and reports the violation codes for each.

## 7. The vision model does not run through ExecuTorch

Stated explicitly, since the distinction is material to the problem statement:

- **Llama 3.2 1B → ExecuTorch (XNNPACK) on the phone's Arm CPU.** This is the
  brief's required technology, and it is genuinely running on-device.
- **The vision classifier → pure NumPy.** It is not exported to ExecuTorch.

The reason is practical: Termux cannot install `torch` or `executorch`, so a
`.pte` vision model could not be executed in the same process as the
application. The descriptor-plus-logistic-head design was chosen so the vision
path requires only NumPy, which Termux does provide. The classifier weights
ship as a NumPy `.npz`, whose presence in release archives is verified by
`build_submission.py` from a clean extraction.

## 8. Energy is estimated, not measured

`results/benchmark.json` reports energy as an explicit estimate: an assumed
2.0 W average CPU draw multiplied by measured wall-clock latency. No power
monitor was used, the field is labelled `"measured": false`, and the
assumption is recorded in the file alongside the figure.

## 9. The offline guarantee has a defined scope

`app/offline_guard.py` gates `connect`, `connect_ex`, `sendto`,
`create_connection` **and `getaddrinfo`**, so the Xaverix process cannot even
resolve a public hostname, let alone reach one. `GET /api/offline-proof` proves
it live, including a loopback control showing ExecuServe is still reachable.

**Scope:** this constrains the Xaverix Python process only. It makes no claim
about other applications on the device, including ExecuServe (which is
configured so that only the local device may connect). Aeroplane mode remains
the end-to-end verification of offline operation.

## 10. Not implemented

No Hailo NPU path. No voice or text-to-speech advisory. No U-Net severity
segmentation. No streaming Llama output. No Llama translation (Hindi and Tamil
come from human-authored strings; see §11). No on-device learning or model
updates. No multi-leaf or whole-plant assessment.

## 11. Translation coverage

English, Hindi and Tamil are human-authored in `data/i18n.json`, and the test
suite fails if any string is missing in any language.

Generation is confined to the English layer: Hindi and Tamil output from a 1B
INT4 model is weak, an incorrect dose in a translated string is a safety defect
rather than a wording defect, and the guard verifies output by token overlap
against an English knowledge base and so cannot verify a translation. Machine
translation would bypass the check that makes generated text safe to display.

Knowledge-base detail beyond the translated strings (full symptom lists and
source citations) is shown in English and labelled as such in the interface.
