# Demonstration video script — Xaverix

**Target length: 3:30, hard limit 4:00.** Screen-record on the phone.

## Before you press record

- [ ] Charger plugged in, battery above 50%
- [ ] **Aeroplane mode ON** — this is the offline proof, and it must be visible
      in the status bar for the whole video
- [ ] ExecuServe hosting Llama 3.2 1B (Hosting tab shows the address)
- [ ] Termux running `server.py` with `XAVERIX_LLM=1` and the API key set
- [ ] One analysis already run, so Llama is warm (otherwise the first call in
      the video takes a minute)
- [ ] `http://localhost:8000` open in Chrome
- [ ] Screen recorder set to capture audio

---

## 0:00-0:20 — Open with the claim

> "This is Xaverix. It checks a crop leaf for disease or pest damage and gives
> a farmer an advisory in their own language. Everything runs on this phone.
> **Offline, on-device, no cloud.** Notice aeroplane mode is on, and it stays
> on for this entire demonstration."

**Show:** the status bar with the aeroplane icon, then the app header.

## 0:20-0:50 — The honest framing, said early

> "One point up front. The vision model is trained on procedurally generated
> leaves, not on PlantVillage or real field photographs, so it is a
> pipeline-validation model. This demonstration uses the bundled samples
> rather than a field photograph, and the documentation states that
> throughout."

**Show:** the synthetic-data warning in the UI.

Stating this early sets the correct expectation for everything that follows.

## 0:50-1:35 — A healthy leaf, then an infected one

**Show:** tap a sample → `healthy_01.png` → **Analyse**.

> "No disease signs detected. Damage is in the trace band, so the advisory
> says no action is needed, just keep scouting."

**Show:** tap `infected_04.png` → **Analyse**.

> "Now it detects damage, estimates how much of the leaf is affected, and the
> urgency changes with the severity band."

**Point at:** the severity meter and the ± interval.

> "That severity figure is a calibrated colour heuristic, not a segmentation
> model, and it carries an error interval because of that."

## 1:35-2:15 — Llama generating on-device, and the grounding

**Point at:** the green provenance chip.

> "This advisory text was written by **Llama 3.2 1B, running on this phone
> through ExecuTorch** — hosted by ExecuServe on loopback. That is the brief's
> required technology, and it is genuinely running here, not in a cloud."

**Show:** expand the pipeline trace — the `llm_advisory` row and the latency.

> "But a 1B model is not reliable on its own."

**Show:** the ExecuServe chat screenshot with the incorrect answer.

> "Asked directly what early blight is, this same model attributed it to a
> bacterium that does not exist. Early blight is caused by the fungus
> *Alternaria solani*. Acting on that answer would mean buying the wrong class
> of product."

## 2:15-2:50 — The guard, demonstrated not asserted

> "So Llama is never asked what a disease is. It is given facts from a local
> knowledge base and allowed only to rewrite them, and every output is checked
> before a farmer sees it."

**Show:** tap **"Test the grounding guard"**.

> "Here is that sentence put through the guard. Rejected: the organism is not
> in the knowledge base. The second case invents a dose for a real fungicide:
> rejected, because every quantity must come from the knowledge base. The
> third invents a price and a website: rejected."

**Point at:** the violation codes.

> "And when the guard rejects something, the app says so and falls back to the
> knowledge base. It never swaps the text silently."

## 2:50-3:15 — Abstention: the part most prototypes skip

**Show:** tap `ood_lined_paper.png` → **Analyse**.

> "This is the failure mode the design targets. A leaf photographed on ruled
> paper is leaf-coloured, but the scene is nothing the model was trained on.
> Without a gate, it would receive a verdict and a high confidence figure."

**Show:** the amber uncertain panel with the gate table.

> "Instead it declines. It reports which check failed and by how much, and
> gives no verdict and no severity figure at all. **Declining is preferable to
> a confident wrong answer.**"

## 3:15-3:30 — Offline proof and languages

**Show:** tap **"Prove it is offline"**.

> "A live check: outbound connections to OpenAI and to Google DNS are blocked,
> and so is DNS resolution itself — the process cannot even look up a public
> hostname. Loopback stays open, which is how it reaches Llama. And aeroplane
> mode has been on the whole time."

**Show:** switch the language selector to **Tamil**, then **Hindi**.

> "Advisories in English, Hindi and Tamil, from reviewed translations rather
> than machine translation — because a wrong pesticide dose in Tamil is a
> safety problem, not a style problem."

---

## Points that must be stated

Omitting any of these would overstate the system:

- "trained on **synthetic** data" / "procedurally generated leaves"
- "**not validated** on real field photographs"
- "Llama 3.2 1B through **ExecuTorch** on this phone"
- "severity is a **heuristic**, with an error interval"
- "offline, on-device, no cloud" (in the first 20 seconds)

## Do not

- Do not demonstrate on a real leaf photograph you took.
- Do not show the Termux screen where the ExecuServe **API key** is visible.
- Do not claim any real-world accuracy figure.
- Do not describe the vision model as an ExecuTorch model. Llama is the
  ExecuTorch component; the vision path is NumPy.

## Afterwards

Upload as **unlisted**, open the link in a private window to confirm it plays,
and put the link in the README and the technical report.
