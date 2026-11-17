const $ = (id) => document.getElementById(id);
const pct = (p) => (p * 100).toFixed(1) + "%";
let lastResult = null, selectedFile = null, caseCounter = 0;
const queue = [];

const CAPTIONS = {
  gradcam: "Grad-CAM for the predicted class: warmer colours mark regions that raised this class's score.",
  original: "The image as the model sees it (resized).",
  outline: "Lesion region found by classical segmentation, used for the shape and colour measurements.",
};

async function loadModelInfo() {
  try {
    const info = await (await fetch("/api/model")).json();
    const st = $("model-status");
    if (info.loaded) {
      st.textContent = `Model ready · ${info.arch} · ${info.img_size}px input`;
      st.className = "status ok";
    } else {
      st.textContent = "Model not loaded: the server is running without a checkpoint.";
      st.className = "status bad";
    }
    renderModel(info);
  } catch (e) {
    $("model-status").textContent = "Server not reachable.";
    $("model-status").className = "status bad";
  }
}

function renderModel(info) {
  const ev = info.evaluation;
  if (!ev || !ev.test) {
    $("model-body").innerHTML = "<p>No evaluation results found on the server.</p>";
    return;
  }
  const t = ev.test, ci = ev.test_ci95 || {}, mel = ev.melanoma_operating_point || {};
  const withCi = (k) => ci[k] ? `<span>95% CI ${pct(ci[k][0])} – ${pct(ci[k][1])}</span>` : "";
  const melTest = mel.test || {};
  $("model-body").innerHTML = `
    <div class="metrics">
      <div class="metric"><span>Balanced accuracy</span><b>${pct(t.balanced_accuracy)}</b>${withCi("balanced_accuracy")}</div>
      <div class="metric"><span>Accuracy</span><b>${pct(t.accuracy)}</b>${withCi("accuracy")}</div>
      <div class="metric"><span>Macro AUC</span><b>${t.macro_auc.toFixed(3)}</b>${ci.macro_auc ? `<span>95% CI ${ci.macro_auc[0].toFixed(3)} – ${ci.macro_auc[1].toFixed(3)}</span>` : ""}</div>
      <div class="metric"><span>Melanoma screening</span><b>${melTest.sensitivity != null ? pct(melTest.sensitivity) : "–"}</b>
        <span>sensitivity at ${melTest.specificity != null ? pct(melTest.specificity) : "–"} specificity</span></div>
    </div>
    <p>Architecture <b>${info.arch || "–"}</b>, evaluated on <b>${ev.n_test}</b> held-out test images
      (${(ev.protocol || "").replace(/^[AB]: /, "")}). Balanced accuracy, the mean of the seven per-class sensitivities, is the official
      ISIC 2018 metric. The melanoma threshold (${mel.threshold != null ? mel.threshold.toFixed(3) : "–"}) was
      fixed on the validation set for 90% sensitivity before the test set was used.</p>`;
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("drop-text").hidden = true;
  $("analyse-btn").disabled = false;
  $("error").hidden = true;
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
}

async function analyse() {
  if (!selectedFile) return;
  const btn = $("analyse-btn");
  btn.disabled = true;
  btn.textContent = "Analysing…";
  $("error").hidden = true;
  try {
    const form = new FormData();
    form.append("file", selectedFile);
    const res = await fetch("/api/predict", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || "Analysis failed.");
    lastResult = body;
    renderResult(body);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyse image";
  }
}

function renderResult(r) {
  $("result").hidden = false;
  const refer = r.triage.level === "refer";
  $("triage").className = "triage " + r.triage.level;
  $("triage").innerHTML = `<span class="icon">${refer ? "⚠" : "✓"}</span><div>
    ${refer ? "Suggest referral to a dermatologist" : "No urgent features detected · routine review"}
    <small>Reason: ${r.triage.reason}. This is a decision-support suggestion, not a diagnosis.</small></div>`;

  const p = r.prediction;
  $("pred-name").textContent = p.name;
  $("pred-cat").textContent = p.category;
  $("pred-cat").className = "pill " + p.category;
  $("pred-prob").textContent = "model probability " + pct(p.prob);
  $("pred-desc").textContent = p.description;

  $("bars").innerHTML = r.probabilities.map((c, i) => `
    <div class="bar-row ${i === 0 ? "top" : ""}" title="${c.name}: ${pct(c.prob)}">
      <span>${c.name.split(" /")[0]}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0.5, c.prob * 100)}%"></div></div>
      <span class="bar-val">${pct(c.prob)}</span>
    </div>`).join("");

  const m = r.melanoma;
  $("mel-text").innerHTML = m.threshold == null
    ? `P(melanoma) = <b>${pct(m.prob)}</b>. No screening threshold is configured.`
    : `P(melanoma) = <b>${pct(m.prob)}</b> against a screening threshold of ${pct(m.threshold)} →
       <b>${m.flagged ? "flagged" : "not flagged"}</b>.` +
      (m.test_sensitivity != null ? ` <span class="muted small">On the test set this threshold caught
       ${pct(m.test_sensitivity)} of melanomas at ${pct(m.test_specificity)} specificity.</span>` : "");

  $("warnings").innerHTML = (r.warnings || []).map((w) => `<div class="warn">${w}</div>`).join("");

  const d = r.descriptors;
  $("desc-body").innerHTML = !d.segmentation_ok
    ? `<p class="muted">The lesion could not be outlined automatically in this image.</p>`
    : `<div class="desc-grid">
        <div class="desc"><span>Asymmetry (0 = symmetric)</span><b>${d.asymmetry.toFixed(2)}</b></div>
        <div class="desc"><span>Border irregularity (0 = circle)</span><b>${d.border_irregularity.toFixed(2)}</b></div>
        <div class="desc"><span>Colours present</span><b>${d.colours.length}</b><span>${d.colours.join(", ") || "–"}</span></div>
        <div class="desc"><span>Lesion share of image</span><b>${pct(d.lesion_area_fraction)}</b></div>
      </div>`;

  showImage("gradcam");
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showImage(key) {
  if (!lastResult) return;
  $("result-img").src = lastResult.images[key];
  $("img-caption").textContent = CAPTIONS[key];
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.img === key));
}

function addToQueue() {
  if (!lastResult) return;
  caseCounter += 1;
  queue.push({
    id: "HDH-" + String(caseCounter).padStart(4, "0"),
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    name: lastResult.prediction.name.split(" /")[0],
    mel: lastResult.melanoma.prob,
    level: lastResult.triage.level,
  });
  queue.sort((a, b) => (a.level === b.level ? b.mel - a.mel : a.level === "refer" ? -1 : 1));
  renderQueue();
  $("queue-btn").textContent = "Added ✓";
  setTimeout(() => ($("queue-btn").textContent = "Add to referral queue"), 1500);
}

function renderQueue() {
  $("queue-body").innerHTML = queue.length === 0
    ? `<tr class="empty"><td colspan="6">No cases yet. Analyse an image and add it to the queue.</td></tr>`
    : queue.map((c, i) => `<tr><td>${c.id}</td><td>${c.time}</td><td>${c.name}</td><td>${pct(c.mel)}</td>
        <td><span class="tag ${c.level}">${c.level === "refer" ? "Refer" : "Routine"}</span></td>
        <td><button class="small" data-remove="${i}">Reviewed</button></td></tr>`).join("");
}

$("file").addEventListener("change", (e) => setFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
$("analyse-btn").addEventListener("click", analyse);
$("queue-btn").addEventListener("click", addToQueue);
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showImage(t.dataset.img)));
$("queue-body").addEventListener("click", (e) => {
  const i = e.target.dataset.remove;
  if (i !== undefined) { queue.splice(Number(i), 1); renderQueue(); }
});
loadModelInfo();
