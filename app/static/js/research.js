// Research page: builds the result plots and tables from /api/results.

const LABELS = { baseline_resnet50: "ResNet-50 baseline", convnext_balanced: "ConvNeXt-Tiny" };
const COLOURS = { baseline_resnet50: "#9aa7b1", convnext_balanced: "#0e6b66" };
const CLASS_NAMES = { MEL: "Melanoma", NV: "Melanocytic nevus", BCC: "Basal cell carcinoma",
  AKIEC: "Actinic keratosis", BKL: "Benign keratosis", DF: "Dermatofibroma", VASC: "Vascular lesion" };

document.getElementById("hero-slider").addEventListener("input", (e) => {
  document.getElementById("hero-overlay").style.opacity = e.target.value / 100;
});

const svgEl = (tag, attrs, text) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text != null) el.textContent = text;
  return el;
};

function legend(id, runs) {
  document.getElementById(id).innerHTML = runs.map((r) =>
    `<span><i style="background:${COLOURS[r.name] || "#555"}"></i>${LABELS[r.name] || r.name}</span>`).join("");
}

function ciPlot(runs) {
  const metrics = [
    ["balanced_accuracy", "Balanced accuracy"],
    ["macro_f1", "Macro-F1"],
    ["melanoma_sensitivity", "Melanoma sensitivity"],
    ["macro_auc", "Macro AUC"],
  ];
  const W = 560, left = 170, right = 20, rowH = 58, top = 26;
  const H = top + metrics.length * rowH + 30;
  const lo = 0.4, hi = 1.0;
  const x = (v) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const svg = document.getElementById("ci-plot");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  for (let t = lo; t <= hi + 1e-9; t += 0.1) {
    svg.append(svgEl("line", { x1: x(t), x2: x(t), y1: top - 8, y2: H - 26, stroke: "#e3e8e6" }));
    svg.append(svgEl("text", { x: x(t), y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "#6f7a85" }, Math.round(t * 100) + "%"));
  }
  metrics.forEach(([key, label], i) => {
    const y0 = top + i * rowH;
    svg.append(svgEl("text", { x: 0, y: y0 + 22, "font-size": 14, fill: "#17202a" }, label));
    runs.forEach((r, j) => {
      const y = y0 + 12 + j * 18;
      const [a, b] = r.test_ci95[key];
      const c = COLOURS[r.name] || "#555";
      svg.append(svgEl("line", { x1: x(a), x2: x(b), y1: y, y2: y, stroke: c, "stroke-width": 3, "stroke-linecap": "round" }));
      svg.append(svgEl("circle", { cx: x(r.test[key]), cy: y, r: 6, fill: c, stroke: "#fff", "stroke-width": 2 }));
      const roomRight = x(b) + 52 <= W;
      svg.append(svgEl("text", { x: roomRight ? x(b) + 10 : x(a) - 10, y: y + 4, "font-size": 12, fill: "#4a5561",
        "text-anchor": roomRight ? "start" : "end" }, pct(r.test[key])));
    });
  });
  legend("ci-legend", runs);
}

function resultsText(base, main) {
  const m = main.melanoma_operating_point, t = main.test, b = base.test;
  document.getElementById("results-text").innerHTML = `
    <p>On ${main.n_test.toLocaleString()} held-out test images, ConvNeXt reached a balanced accuracy of
      <b>${pct(t.balanced_accuracy)}</b> (95% CI ${pct(main.test_ci95.balanced_accuracy[0])} to ${pct(main.test_ci95.balanced_accuracy[1])}),
      against ${pct(b.balanced_accuracy)} for the baseline. Melanoma sensitivity as the top prediction rose from
      ${pct(b.melanoma_sensitivity)} to ${pct(t.melanoma_sensitivity)}.</p>
    <p>The capstone reported 36.3% melanoma recall on an image-level validation split. Because that split let
      images of the same lesion appear on both sides, its figures are not directly comparable, but the direction
      of the change is clear.</p>
    <div class="callout">
      <strong>${pct(m.test.sensitivity)} of test melanomas flagged at ${pct(m.test.specificity)} specificity</strong>
      <span class="muted">using a probability threshold of ${m.threshold.toFixed(3)}, fixed on the validation set
        (${pct(m.validation.sensitivity)} sensitivity there). Melanoma-vs-rest AUC ${m.test_auc.toFixed(3)}.</span>
    </div>`;
}

function classTable(base, main) {
  const rows = Object.keys(CLASS_NAMES).map((c) => {
    const b = base.per_class[c], m = main.per_class[c];
    const d = m.sensitivity - b.sensitivity;
    const delta = Math.abs(d) < 0.005 ? "" :
      `<span class="${d > 0 ? "delta-up" : "delta-down"}">${d > 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)}</span>`;
    return `<tr><td>${CLASS_NAMES[c]}</td><td class="n">${m.support}</td>
      <td class="n">${pct(b.sensitivity)}</td><td class="n"><b>${pct(m.sensitivity)}</b> ${delta}</td>
      <td class="n">${pct(m.specificity)}</td><td class="n">${m.auc.toFixed(3)}</td></tr>`;
  }).join("");
  document.getElementById("class-table").innerHTML = `
    <thead><tr><th>Class</th><th class="n">Test images</th><th class="n">Sensitivity, baseline</th>
      <th class="n">Sensitivity, ConvNeXt</th><th class="n">Specificity, ConvNeXt</th><th class="n">AUC, ConvNeXt</th></tr></thead>
    <tbody>${rows}</tbody>`;
}

function historyPlot(runs) {
  const W = 560, H = 270, left = 44, right = 14, top = 14, bottom = 52;
  const maxEp = Math.max(...runs.map((r) => (r.history || []).length));
  const lo = 0.2, hi = 0.9;
  const x = (e) => left + ((e - 1) / Math.max(1, maxEp - 1)) * (W - left - right);
  const y = (v) => top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom);
  const svg = document.getElementById("history-plot");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  for (let t = 0.2; t <= 0.9 + 1e-9; t += 0.1) {
    svg.append(svgEl("line", { x1: left, x2: W - right, y1: y(t), y2: y(t), stroke: "#e3e8e6" }));
    svg.append(svgEl("text", { x: left - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 12, fill: "#6f7a85" }, Math.round(t * 100) + "%"));
  }
  [1, 5, 10, 15, 20, 25].filter((e) => e <= maxEp).forEach((e) =>
    svg.append(svgEl("text", { x: x(e), y: H - 30, "text-anchor": "middle", "font-size": 12, fill: "#6f7a85" }, e)));
  svg.append(svgEl("text", { x: (left + W - right) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12.5, fill: "#4a5561" }, "Epoch"));
  runs.forEach((r) => {
    const h = r.history || [];
    if (!h.length) return;
    const c = COLOURS[r.name] || "#555";
    svg.append(svgEl("polyline", { points: h.map((p) => `${x(p.epoch)},${y(p.val_balanced_acc)}`).join(" "),
      fill: "none", stroke: c, "stroke-width": 2.5, "stroke-linejoin": "round" }));
    const best = h.reduce((a, b) => (b.val_balanced_acc > a.val_balanced_acc ? b : a));
    svg.append(svgEl("circle", { cx: x(best.epoch), cy: y(best.val_balanced_acc), r: 6, fill: c, stroke: "#fff", "stroke-width": 2 }));
  });
  legend("history-legend", runs);
  const main = runs.find((r) => r.name === "convnext_balanced");
  if (main && main.history) {
    const best = main.history.reduce((a, b) => (b.val_balanced_acc > a.val_balanced_acc ? b : a));
    document.getElementById("history-text").innerHTML =
      `ConvNeXt reached its best validation balanced accuracy (${pct(best.val_balanced_acc)}) at epoch ${best.epoch}
       of ${main.history.length} and fluctuated afterwards, a sign that the small rare classes make the metric noisy
       and that the model starts to overfit. That epoch was kept; the test set played no part in the choice.
       Training took ${Math.round(main.train_minutes)} minutes on one T4 GPU (dots mark the selected epochs).`;
  }
}

api("/api/results").then((data) => {
  const runs = data.runs.filter((r) => r.test && r.test_ci95);
  const base = runs.find((r) => r.name === "baseline_resnet50");
  const main = runs.find((r) => r.name === "convnext_balanced");
  if (!base || !main) {
    document.getElementById("results-text").innerHTML = `<p class="alert warn">Result files were not found in the results folder.</p>`;
    return;
  }
  const ordered = [base, main];
  ciPlot(ordered);
  resultsText(base, main);
  classTable(base, main);
  historyPlot(ordered);
}).catch((e) => {
  document.getElementById("results-text").innerHTML = `<p class="alert error">${esc(e.message)}</p>`;
});
