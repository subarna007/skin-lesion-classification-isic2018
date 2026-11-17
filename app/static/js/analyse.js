// Analyse page: upload, call /api/predict, render the result.

const $ = (id) => document.getElementById(id);
let file = null, result = null, camClass = null;

document.querySelector(".ring").innerHTML = ICONS.upload;

api("/api/model").then((m) => {
  $("model-status").textContent = m.loaded
    ? `Model loaded: ${m.arch} at ${m.img_size} px.` + (m.gallery ? ` Similar-case gallery: ${m.gallery_size} images.` : "")
    : "The model is not loaded on this server, so images cannot be analysed yet.";
  $("model-status").className = m.loaded ? "fine" : "alert warn";
}).catch(() => { $("model-status").textContent = "The server cannot be reached."; });

function choose(f) {
  if (!f) return;
  file = f;
  $("preview").src = URL.createObjectURL(f);
  $("preview").hidden = false;
  $("drop-empty").hidden = true;
  $("analyse-btn").disabled = false;
  $("error").hidden = true;
}

$("file").addEventListener("change", (e) => choose(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

$("analyse-btn").addEventListener("click", async () => {
  if (!file) return;
  const btn = $("analyse-btn");
  btn.disabled = true;
  btn.textContent = "Analysing…";
  $("error").hidden = true;
  try {
    const form = new FormData();
    form.append("file", file);
    result = await api("/api/predict", { method: "POST", body: form });
    render(result);
  } catch (e) {
    $("error").textContent = e.message;
    $("error").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse image";
  }
});

function render(r) {
  $("result").hidden = false;

  // Scope viewer
  $("img-base").src = r.images.original;
  const camCodes = Object.keys(r.images.heatmaps);
  camClass = camCodes[0];
  $("cam-classes").innerHTML = camCodes.map((c) => {
    const p = r.probabilities.find((x) => x.code === c);
    return `<button type="button" data-code="${c}" aria-pressed="${c === camClass}">${esc(p.name.split(" /")[0])}</button>`;
  }).join("");
  $("show-outline").checked = false;
  updateScope();

  // Verdict and diagnosis
  const refer = r.triage.level === "refer";
  $("verdict").className = "verdict " + r.triage.level;
  $("verdict").innerHTML = `${refer ? ICONS.refer : ICONS.routine}<div>
      <h3>${refer ? "Refer to a dermatologist" : "Routine follow-up"}</h3>
      <p>Because ${esc(r.triage.reason)}. This is a decision-support suggestion, not a diagnosis.</p></div>`;
  $("dx-name").textContent = r.prediction.name;
  $("dx-cat").textContent = r.prediction.category;
  $("dx-cat").className = "tag " + r.prediction.category;
  $("dx-prob").textContent = `model probability ${pct(r.prediction.prob)}`;
  $("case-ref").innerHTML = `<span class="fine">Case <span class="case-id">${esc(r.case_id)}</span></span>`;

  $("bars").innerHTML = r.probabilities.map((p, i) => `
    <div class="bar${i === 0 ? " top" : ""}">
      <span>${esc(p.name.split(" /")[0])}</span>
      <div class="track"><div class="fill" style="width:${Math.max(0.6, p.prob * 100)}%"></div></div>
      <span class="v">${pct(p.prob)}</span>
    </div>`).join("");

  // Melanoma meter
  const m = r.melanoma;
  $("meter-dot").style.left = `${m.prob * 100}%`;
  if (m.threshold != null) {
    $("meter-tick").hidden = false;
    $("meter-tick").style.left = `${m.threshold * 100}%`;
    $("meter-tick-label").textContent = `threshold ${pct(m.threshold)}`;
    $("mel-text").innerHTML = `Melanoma probability <b>${pct(m.prob)}</b>, ${m.flagged ? "<b>above</b>" : "below"} the
      screening threshold. On the test set this threshold flagged ${pct(m.test_sensitivity)} of melanomas at
      ${pct(m.test_specificity)} specificity.`;
  } else {
    $("meter-tick").hidden = true;
    $("mel-text").innerHTML = `Melanoma probability <b>${pct(m.prob)}</b>. No screening threshold is configured.`;
  }
  $("warnings").innerHTML = (r.warnings || []).map((w) => `<p class="alert warn" style="margin-top:14px">${esc(w)}</p>`).join("");

  renderAbcde(r.descriptors);
  renderSimilar(r.similar);
  renderCondition(r.info);

  $("book-link").href = `/appointments?case=${encodeURIComponent(r.case_id)}`;
  $("queue-btn").disabled = false;
  $("queue-btn").textContent = "Send to review queue";
  $("queue-msg").textContent = "";

  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function updateScope() {
  if (!result) return;
  const outline = $("show-outline").checked;
  const overlay = $("img-overlay");
  if (outline) {
    $("img-base").src = result.images.outline;
    overlay.style.opacity = 0;
    $("cam-caption").textContent = "The lesion region found by image segmentation, used for the ABCDE measurements below.";
  } else {
    $("img-base").src = result.images.original;
    overlay.src = result.images.heatmaps[camClass];
    overlay.style.opacity = $("cam-slider").value / 100;
    const name = result.probabilities.find((x) => x.code === camClass).name.split(" /")[0].toLowerCase();
    $("cam-caption").textContent = `Grad-CAM for ${name}: warmer regions raised the ${name} score the most. Compare classes to see what separates them.`;
  }
  document.querySelectorAll("#cam-classes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.code === camClass && !outline)));
}

$("cam-slider").addEventListener("input", updateScope);
$("show-outline").addEventListener("change", updateScope);
$("cam-classes").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  camClass = b.dataset.code;
  $("show-outline").checked = false;
  updateScope();
});

// ABCDE -----------------------------------------------------------------------
let descriptors = null;
const level = (v, lo, hi) => (v >= hi ? "high" : v >= lo ? "moderate" : "low");

function renderAbcde(d) {
  descriptors = d;
  const auto = d.segmentation_ok;
  const rows = [
    ["A", "Asymmetry", "Mismatch after folding the lesion along its two main axes",
      auto ? `<span class="gauge"><i style="width:${Math.min(100, d.asymmetry / 0.3 * 100)}%"></i></span>${d.asymmetry.toFixed(2)} (${level(d.asymmetry, 0.1, 0.2)})` : "Not measurable",
      auto ? d.asymmetry >= 0.1 : null, "Measured from image"],
    ["B", "Border", "Irregularity of the outline (0 for a perfect circle)",
      auto ? `<span class="gauge"><i style="width:${Math.min(100, d.border_irregularity / 0.4 * 100)}%"></i></span>${d.border_irregularity.toFixed(2)} (${level(d.border_irregularity, 0.15, 0.25)})` : "Not measurable",
      auto ? d.border_irregularity >= 0.2 : null, "Measured from image"],
    ["C", "Colour", "Number of the six dermoscopic colours covering at least 5% of the lesion",
      auto ? `${d.colours.length}: ${esc(d.colours.join(", ") || "none")}` : "Not measurable",
      auto ? d.colours.length >= 3 : null, "Measured from image"],
    ["D", "Diameter", "Largest diameter in millimetres, measured on the patient",
      `<input type="number" id="abcde-d" min="0" max="200" step="0.5" placeholder="mm" aria-label="Diameter in millimetres">`,
      "D", "Entered by clinician"],
    ["E", "Evolution", "Has the lesion changed in size, shape or colour?",
      `<select id="abcde-e" aria-label="Has the lesion changed"><option value="">Unknown</option><option value="yes">Yes</option><option value="no">No</option></select>`,
      "E", "Entered by clinician"],
  ];
  $("abcde-body").innerHTML = rows.map(([letter, name, desc, value, , src]) => `
    <tr><td class="letter">${letter}</td>
      <td class="crit"><b>${name}</b><span>${desc}</span></td>
      <td class="val">${value}</td>
      <td class="finding" data-letter="${letter}"></td>
      <td class="src">${src}</td></tr>`).join("");
  $("abcde-d").addEventListener("input", updateAbcde);
  $("abcde-e").addEventListener("change", updateAbcde);
  updateAbcde();
}

function updateAbcde() {
  const d = descriptors, auto = d.segmentation_ok;
  const dVal = parseFloat($("abcde-d").value);
  const eVal = $("abcde-e").value;
  const findings = {
    A: auto ? d.asymmetry >= 0.1 : null,
    B: auto ? d.border_irregularity >= 0.2 : null,
    C: auto ? d.colours.length >= 3 : null,
    D: isNaN(dVal) ? null : dVal > 6,
    E: eVal === "" ? null : eVal === "yes",
  };
  document.querySelectorAll("#abcde-body .finding").forEach((td) => {
    const f = findings[td.dataset.letter];
    td.innerHTML = f == null ? `<span class="tag">Unknown</span>` : f ? `<span class="tag malignant">Present</span>` : `<span class="tag benign">Absent</span>`;
  });
  const known = Object.values(findings).filter((v) => v != null);
  const present = known.filter(Boolean).length;
  const unknown = 5 - known.length;
  $("abcde-count").textContent = `${present} of ${known.length} assessed ${known.length === 1 ? "criterion" : "criteria"} present`;
  $("abcde-note").textContent = (unknown ? `${unknown} ${unknown === 1 ? "criterion is" : "criteria are"} not yet assessed. ` : "") +
    "Cut-offs used: asymmetry 0.10, border 0.20, three or more colours, diameter over 6 mm. They are illustrative and not validated.";
}

// Similar cases ---------------------------------------------------------------
function renderSimilar(items) {
  if (!items || !items.length) {
    $("similar-intro").textContent = "The similar-case gallery has not been built on this server, so no examples can be shown.";
    $("similar-grid").innerHTML = "";
    return;
  }
  $("similar-intro").textContent = "The six training images whose model features are closest to this image, with their confirmed diagnoses. Agreement between neighbours and the prediction is a rough check of how typical the image is.";
  $("similar-grid").innerHTML = items.map((s) => `
    <figure><img src="${esc(s.url)}" alt="${esc(s.name)} from the training set" loading="lazy">
      <figcaption>${esc(s.name.split(" /")[0])}<span>similarity ${s.similarity.toFixed(2)}</span></figcaption></figure>`).join("");
}

// Condition info --------------------------------------------------------------
function renderCondition(info) {
  $("cond-title").textContent = `About ${info.name.toLowerCase()}`;
  $("cond-summary").textContent = info.summary;
  $("cond-features").innerHTML = info.features.map((f) => `<li>${esc(f)}</li>`).join("");
  $("cond-management").textContent = info.management;
  const L = info.links;
  const items = [
    ["google", L.google, "Search Google", "General information and patient leaflets"],
    ["youtube", L.youtube, "Watch on YouTube", "Dermoscopy teaching videos"],
    ["wikipedia", L.wikipedia, "Read on Wikipedia", "Encyclopaedia overview"],
    ["scholar", L.scholar, "Search Google Scholar", "Research papers on dermoscopy and AI"],
  ];
  $("cond-links").innerHTML = items.map(([icon, href, title, sub]) =>
    `<a href="${esc(href)}" target="_blank" rel="noopener">${ICONS[icon]}<span>${title}<small>${sub}</small></span></a>`).join("");
}

$("queue-btn").addEventListener("click", async () => {
  if (!result) return;
  try {
    await api(`/api/cases/${encodeURIComponent(result.case_id)}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "queued" }) });
    $("queue-btn").disabled = true;
    $("queue-btn").textContent = "Sent to review queue";
    $("queue-msg").innerHTML = `Case ${esc(result.case_id)} is on the <a href="/hospital#queue">hospital review queue</a>.`;
  } catch (e) {
    $("queue-msg").textContent = e.message;
  }
});
