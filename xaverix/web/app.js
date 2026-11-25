/* Xaverix single-page app. No framework, no build step, no network.
   Everything it needs is served by the Python process on this device. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  imageDataUrl: null,
  imageName: null,
  strings: {},
  config: null,
  samples: [],
};

/* ------------------------------------------------------------- helpers */

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
  return response.json();
}

function text(id, value) {
  const node = $(id);
  if (node && value !== undefined && value !== null) node.textContent = value;
}

function show(id, visible) {
  const node = $(id);
  if (node) node.hidden = !visible;
}

function s(key, fallback) {
  return state.strings[key] || fallback || key;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

/* ---------------------------------------------------------- bootstrap */

async function boot() {
  try {
    state.config = await getJSON("/api/config");
    renderConfig(state.config);
  } catch (err) {
    console.error(err);
  }

  await loadLanguage($("language").value || "en");

  try {
    const data = await getJSON("/api/samples");
    state.samples = data.samples || [];
    renderSamples();
  } catch (err) {
    console.error(err);
  }
}

function renderConfig(config) {
  // crops / regions
  const crop = $("crop");
  clear(crop);
  (config.crops || []).forEach((c) => {
    const option = el("option", null, c.label_en);
    option.value = c.key;
    option.dataset.labels = JSON.stringify(c);
    crop.appendChild(option);
  });

  const region = $("region");
  clear(region);
  (config.regions || []).forEach((r) => {
    const option = el("option", null, r.label_en);
    option.value = r.key;
    option.dataset.labels = JSON.stringify(r);
    region.appendChild(option);
  });
  region.value = "tamil_nadu";

  const language = $("language");
  if (!language.options.length) {
    [["en", "English"], ["hi", "हिन्दी"], ["ta", "தமிழ்"]].forEach(([code, label]) => {
      const option = el("option", null, label);
      option.value = code;
      language.appendChild(option);
    });
  }

  // Llama chip
  const llm = config.llm || {};
  const chip = $("llm-chip");
  if (llm.usable) {
    chip.className = "chip chip-good";
    chip.textContent = `Llama 3.2 1B · ExecuTorch ${llm.warm ? "ready" : "loading…"}`;
  } else if (llm.configured) {
    chip.className = "chip chip-warn";
    chip.textContent = `Llama not reachable (${llm.reason || "unknown"})`;
  } else {
    chip.className = "chip chip-neutral";
    chip.textContent = "Llama disabled";
  }
  chip.title = `${llm.endpoint || ""} · ${llm.model || ""}`;

  // disclosures + version
  const list = $("disclosures");
  clear(list);
  (config.disclosures || []).forEach((item) => list.appendChild(el("li", null, item)));

  const classifier = config.classifier || {};
  const training = classifier.training || {};
  const parts = [];
  if (config.runtime) parts.push(`v${config.runtime.version}`);
  if (training.val_accuracy_synthetic !== undefined) {
    parts.push(
      `classifier ${(100 * training.val_accuracy_synthetic).toFixed(1)}% on held-out synthetic data`
    );
  }
  if (config.runtime && config.runtime.target_device) {
    parts.push(config.runtime.target_device);
  }
  text("footer-version", parts.join(" · "));

  if (!classifier.available) {
    const warning = $("synthetic-warning");
    warning.textContent = classifier.error || s("model_missing");
  }
}

async function loadLanguage(code) {
  try {
    const data = await getJSON(`/api/i18n?lang=${encodeURIComponent(code)}`);
    state.strings = data.strings || {};
  } catch (err) {
    console.error(err);
    return;
  }
  const map = {
    "app-title": "app_title",
    "app-tagline": "app_tagline",
    "offline-badge": "offline_badge",
    "lbl-choose": "choose_image",
    "lbl-take": "take_photo",
    "lbl-samples": "or_use_sample",
    "lbl-crop": "crop",
    "lbl-region": "region",
    "lbl-language": "language",
    "lbl-analyse": "analyse",
    "lbl-severity": "severity",
    "lbl-why": "why_uncertain",
    "lbl-causes": "probable_causes",
    "lbl-scouting": "scouting",
    "lbl-region-note": "region_note",
    "lbl-trace": "pipeline_trace",
    "ask-officer": "ask_officer",
  };
  Object.entries(map).forEach(([id, key]) => text(id, s(key)));

  const classifier = (state.config && state.config.classifier) || {};
  if (classifier.available) {
    text("synthetic-warning", s("synthetic_warning"));
  }

  // Re-label crop and region options in the chosen language.
  ["crop", "region"].forEach((which) => {
    Array.from($(which).options).forEach((option) => {
      if (!option.dataset.labels) return;
      const labels = JSON.parse(option.dataset.labels);
      option.textContent = labels[`label_${code}`] || labels.label_en;
    });
  });
}

/* ------------------------------------------------------------ samples */

function renderSamples() {
  const strip = $("sample-strip");
  clear(strip);
  state.samples.forEach((sample) => {
    const button = el("button");
    button.type = "button";
    const image = el("img");
    image.src = sample.url;
    image.alt = sample.file;
    image.loading = "lazy";
    button.appendChild(image);

    const label =
      sample.expected === "uncertain"
        ? "OOD"
        : sample.expected === "healthy"
        ? "healthy"
        : "infected";
    button.appendChild(el("span", "tag", label));

    button.addEventListener("click", async () => {
      Array.from(strip.children).forEach((child) => child.classList.remove("selected"));
      button.classList.add("selected");
      await selectSampleImage(sample);
    });
    strip.appendChild(button);
  });
}

async function selectSampleImage(sample) {
  const response = await fetch(sample.url, { cache: "no-store" });
  const blob = await response.blob();
  const reader = new FileReader();
  await new Promise((resolve) => {
    reader.onload = () => {
      state.imageDataUrl = reader.result;
      state.imageName = sample.file;
      resolve();
    };
    reader.readAsDataURL(blob);
  });
  showPreview(sample.file);
}

function showPreview(name) {
  $("preview").src = state.imageDataUrl;
  text("preview-name", name || "");
  show("preview-wrap", true);
  $("analyse").disabled = false;
}

/* ------------------------------------------------------------ analyse */

async function analyse() {
  if (!state.imageDataUrl) return;

  const button = $("analyse");
  button.disabled = true;
  show("progress", true);
  text("progress", s("analysing"));

  // Warn about the long first Llama call rather than letting it look hung.
  const llm = (state.config && state.config.llm) || {};
  let timer = null;
  if (llm.usable && !llm.warm) {
    timer = setTimeout(() => text("progress", s("llama_working")), 1500);
  }

  try {
    const result = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: state.imageDataUrl,
        crop: $("crop").value,
        region: $("region").value,
        language: $("language").value,
      }),
    });
    const data = await result.json();
    if (!result.ok) throw new Error(data.error || `HTTP ${result.status}`);
    renderResult(data);
    // Llama may have warmed up during this call; refresh the chip.
    getJSON("/api/config").then((config) => {
      state.config = config;
      renderConfig(config);
    }).catch(() => {});
  } catch (err) {
    console.error(err);
    text("progress", `${s("error_generic")} (${err.message})`);
    show("progress", true);
    button.disabled = false;
    if (timer) clearTimeout(timer);
    return;
  }

  if (timer) clearTimeout(timer);
  show("progress", false);
  button.disabled = false;
  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResult(data) {
  show("result-card", true);

  // ---- verdict
  $("verdict-dot").className = `verdict-dot ${data.verdict}`;
  text("verdict-label", data.verdict_label);

  const meta = [];
  if (data.confidence !== null && data.confidence !== undefined) {
    meta.push(`${s("confidence")} ${(100 * data.confidence).toFixed(0)}%`);
    if (data.confidence_capped) meta.push("capped for display");
  }
  meta.push(`${data.context.crop_label} · ${data.context.region_label}`);
  text("verdict-meta", meta.join(" · "));

  // ---- severity
  const severity = data.severity || {};
  if (severity.available && !severity.suppressed) {
    show("severity-block", true);
    const value = severity.severity;
    text("severity-value", `${value.toFixed(0)}% (${data.severity_band_label})`);
    const fill = $("severity-fill");
    fill.style.width = `${Math.min(100, Math.max(2, value))}%`;
    fill.className = `meter-fill ${severity.band}`;
    text(
      "severity-detail",
      `±${severity.interval} points · ${severity.method}`
    );
  } else {
    show("severity-block", false);
  }

  // ---- abstention
  if (data.abstained) {
    show("uncertain-panel", true);
    text("uncertain-detail", data.abstain_detail || "");
    renderGates(data.gates || {}, data.abstain_reason);
  } else {
    show("uncertain-panel", false);
  }

  // ---- provenance
  const advisory = data.advisory || {};
  const chip = $("source-chip");
  chip.textContent = advisory.source_label || "";
  chip.className =
    "chip " +
    (advisory.source === "llama_grounded"
      ? "chip-good"
      : advisory.source === "kb_fallback_guard_reject"
      ? "chip-warn"
      : "chip-neutral");

  const detail = [];
  if (advisory.source_detail) detail.push(advisory.source_detail);
  if (advisory.model) {
    detail.push(
      `${advisory.model.name} · ${advisory.model.quantisation} · ${advisory.model.runtime} · ${(
        advisory.model.latency_ms / 1000
      ).toFixed(1)} s${advisory.model.retries ? `, ${advisory.model.retries} retry` : ""}`
    );
  }
  text("source-detail", detail.join(" — "));
  renderGuard(advisory.guard);

  // ---- advisory sections
  const container = $("advisory");
  clear(container);
  (advisory.sections || []).forEach((section) => {
    const block = el("div", "advisory-section");
    block.appendChild(el("h3", null, section.heading));
    block.appendChild(el("p", null, section.body));
    container.appendChild(block);
  });

  if (advisory.llama_note && data.context.language !== "en") {
    const note = el("div", "llama-note");
    note.appendChild(el("span", "label", s("personalised_note")));
    note.appendChild(document.createTextNode(advisory.llama_note));
    container.appendChild(note);
  }

  // ---- probable causes
  const causes = data.probable_causes || [];
  if (causes.length) {
    show("causes-block", true);
    text("causes-note", data.probable_causes_note || "");
    const host = $("causes");
    clear(host);
    causes.forEach((cause) => {
      const card = el("div", "cause");
      const heading = el("h4", null, `${cause.name} `);
      if (cause.pathogen) {
        heading.appendChild(
          el("span", "latin", `— ${cause.pathogen} (${cause.pathogen_type})`)
        );
      } else if (cause.pathogen_type) {
        heading.appendChild(el("span", "latin", `— ${cause.pathogen_type}`));
      }
      card.appendChild(heading);
      card.appendChild(el("p", "small", cause.distinguishing_symptom));
      if (cause.organic) {
        card.appendChild(
          el(
            "p",
            "treat",
            `${s("section_organic")}: ${cause.organic.input} — ${cause.organic.dose}, ${cause.organic.interval}`
          )
        );
      }
      if (cause.chemical) {
        const phi =
          cause.chemical.phi_days !== undefined && cause.chemical.phi_days !== null
            ? ` · no harvest for ${cause.chemical.phi_days} days`
            : "";
        card.appendChild(
          el(
            "p",
            "treat",
            `Chemical: ${cause.chemical.input} — ${cause.chemical.dose}, ${cause.chemical.interval}${phi}`
          )
        );
      }
      host.appendChild(card);
    });
  } else {
    show("causes-block", false);
  }

  // ---- scouting + region note
  const scouting = data.scouting || [];
  if (scouting.length) {
    show("scouting-block", true);
    const list = $("scouting");
    clear(list);
    scouting.forEach((item) => list.appendChild(el("li", null, item)));
  } else {
    show("scouting-block", false);
  }

  if (data.region_note) {
    show("region-note-block", true);
    text("region-note", data.region_note);
  } else {
    show("region-note-block", false);
  }

  renderTrace(data);
}

function renderGates(gates, reason) {
  const table = $("gate-table");
  clear(table);
  const rows = [
    ["Leaf area in frame", gates.leaf_fraction, `min ${gates.leaf_fraction_min}`, "no_leaf_detected"],
    ["Green leaf tissue", gates.green_fraction, `min ${gates.green_fraction_min}`, "not_leaf_like"],
    ["Distance from training data", gates.ood_distance, `limit ${gates.ood_threshold}`, "out_of_distribution"],
    ["Model confidence", gates.confidence_raw, `min ${gates.confidence_min}`, "low_confidence"],
  ];
  rows.forEach(([label, value, limit, failReason]) => {
    if (value === undefined) return;
    const row = document.createElement("tr");
    if (failReason === reason) row.className = "failed";
    row.appendChild(el("td", null, label));
    row.appendChild(el("td", null, `${value} (${limit})`));
    table.appendChild(row);
  });
}

function renderGuard(guard) {
  if (!guard) {
    show("guard-details", false);
    return;
  }
  show("guard-details", true);
  const body = $("guard-body");
  clear(body);
  body.appendChild(
    el(
      "p",
      "small",
      `decision: ${guard.decision} · token support ${guard.support} · bigram overlap ${guard.bigram_overlap} · ${guard.words} words`
    )
  );
  (guard.violations || []).forEach((violation) => {
    body.appendChild(el("p", "violation", `${violation.code}: ${violation.detail}`));
  });
  (guard.warnings || []).forEach((warning) => {
    body.appendChild(el("p", "small muted", warning));
  });
}

function renderTrace(data) {
  const table = $("trace-table");
  clear(table);
  (data.trace || []).forEach((entry) => {
    const row = document.createElement("tr");
    row.appendChild(el("td", null, entry.stage));
    row.appendChild(el("td", null, entry.detail || entry.source || ""));
    row.appendChild(el("td", null, entry.ms !== undefined ? `${entry.ms} ms` : ""));
    table.appendChild(row);
  });
  const total = document.createElement("tr");
  total.appendChild(el("td", null, "total"));
  total.appendChild(el("td", null, ""));
  total.appendChild(el("td", null, `${data.timings_ms.total} ms`));
  table.appendChild(total);
}

/* -------------------------------------------------------------- proof */

async function showProof(url, title) {
  const output = $("proof-output");
  show("proof-output", true);
  output.textContent = `${title}\nloading…`;
  try {
    const data = await getJSON(url);
    output.textContent = `${title}\n\n${JSON.stringify(data, null, 2)}`;
  } catch (err) {
    output.textContent = `${title}\n\nfailed: ${err.message}`;
  }
}

/* ------------------------------------------------------------- wiring */

$("file-input").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    state.imageDataUrl = reader.result;
    state.imageName = file.name;
    Array.from($("sample-strip").children).forEach((child) =>
      child.classList.remove("selected")
    );
    showPreview(file.name);
  };
  reader.readAsDataURL(file);
});

$("sample-toggle").addEventListener("click", () => {
  const strip = $("sample-strip");
  strip.hidden = !strip.hidden;
});

$("analyse").addEventListener("click", analyse);

$("language").addEventListener("change", (event) => loadLanguage(event.target.value));

$("btn-offline").addEventListener("click", () =>
  showProof("/api/offline-proof", "GET /api/offline-proof — live outbound connection attempts")
);

$("btn-guard").addEventListener("click", () =>
  showProof("/api/guard-demo", "GET /api/guard-demo — grounding guard applied to invalid outputs")
);

boot();
