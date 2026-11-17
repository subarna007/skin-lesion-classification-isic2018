// Appointments page: booking form linked to an analysed case.

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let caseId = params.get("case");
let hospital = null, chosenDate = null, chosenSlot = null;

async function init() {
  hospital = await api("/api/hospital");
  renderDates();
  if (caseId) {
    try {
      const c = await api(`/api/cases/${encodeURIComponent(caseId)}`);
      $("linked").hidden = false;
      $("linked").innerHTML = `<span class="field-label">Linked case</span>
        <div class="linked-case"><img src="${esc(c.thumb)}" alt="">
          <div><b class="case-id">${esc(c.id)}</b>: ${esc(c.name)} (${pct(c.prob)})
            <span class="tag ${c.triage}" style="margin-left:6px">${c.triage === "refer" ? "Refer" : "Routine"}</span>
            <div class="fine">P(melanoma) ${pct(c.mel_prob)}. <a href="#" id="unlink">Book without this case</a></div></div></div>`;
      $("unlink").addEventListener("click", (e) => { e.preventDefault(); caseId = null; $("linked").hidden = true; });
    } catch (e) {
      caseId = null;
    }
  }
  loadBooked();
}

function renderDates() {
  $("dates").innerHTML = hospital.dates.map((d) => {
    const full = (hospital.booked[d] || []).length >= hospital.slots.length;
    return `<button type="button" class="chip" data-date="${d}" aria-pressed="${d === chosenDate}" ${full ? "disabled" : ""}>
      ${fmtDate(d)}<small>${full ? "Fully booked" : `${hospital.slots.length - (hospital.booked[d] || []).length} slots free`}</small></button>`;
  }).join("");
}

function renderSlots() {
  const taken = new Set(hospital.booked[chosenDate] || []);
  $("slots").innerHTML = hospital.slots.map((s) =>
    `<button type="button" class="chip" data-slot="${s}" aria-pressed="${s === chosenSlot}" ${taken.has(s) ? "disabled" : ""}>${s}</button>`).join("");
}

$("dates").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-date]");
  if (!b || b.disabled) return;
  chosenDate = b.dataset.date;
  chosenSlot = null;
  renderDates();
  renderSlots();
});
$("slots").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-slot]");
  if (!b || b.disabled) return;
  chosenSlot = b.dataset.slot;
  renderSlots();
});

$("book-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("form-error");
  err.hidden = true;
  const name = $("f-name").value.trim(), age = parseInt($("f-age").value, 10), sex = $("f-sex").value;
  const problems = [];
  if (name.length < 2) problems.push("enter the patient's name");
  if (isNaN(age) || age < 0 || age > 120) problems.push("enter an age between 0 and 120");
  if (!sex) problems.push("select the patient's sex");
  if (!chosenDate) problems.push("choose a date");
  if (!chosenSlot) problems.push("choose a time");
  if (problems.length) {
    err.textContent = "To book, " + problems.join(", ") + ".";
    err.hidden = false;
    return;
  }
  $("book-btn").disabled = true;
  try {
    const a = await api("/api/appointments", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patient_name: name, age, sex, phone: $("f-phone").value.trim() || null,
        date: chosenDate, slot: chosenSlot, case_id: caseId, notes: $("f-notes").value.trim() || null }),
    });
    $("book-form").hidden = true;
    $("confirm").hidden = false;
    $("confirm").innerHTML = `<h2>Appointment booked</h2>
      <p class="muted">Show this reference at the outpatient registration desk.</p>
      <dl><dt>Reference</dt><dd class="case-id">${esc(a.id)}</dd>
        <dt>Patient</dt><dd>${esc(a.patient_name)}, ${a.age}</dd>
        <dt>When</dt><dd>${fmtDate(a.date, true)}, ${esc(a.slot)}</dd>
        <dt>Where</dt><dd>Dermatology clinic, ${esc(hospital.name)}</dd>
        ${a.case ? `<dt>Linked case</dt><dd>${esc(a.case.id)}: ${esc(a.case.name)}</dd>` : ""}</dl>
      <p class="fine">Bring any previous biopsy reports and a list of current medicines.</p>
      <div class="btn-row"><a class="btn" href="/appointments">Book another appointment</a><a class="btn" href="/hospital#queue">Open the review queue</a></div>`;
    hospital = await api("/api/hospital");
    loadBooked();
  } catch (e2) {
    err.textContent = e2.message;
    err.hidden = false;
    hospital = await api("/api/hospital");
    renderDates();
    if (chosenDate) renderSlots();
  } finally {
    $("book-btn").disabled = false;
  }
});

async function loadBooked() {
  const items = await api("/api/appointments");
  const today = new Date().toISOString().slice(0, 10);
  const upcoming = items.filter((a) => a.date >= today);
  $("booked").innerHTML = !upcoming.length
    ? `<p class="muted" style="margin-top:16px">No appointments booked yet.</p>`
    : `<table class="data" style="margin-top:12px"><thead><tr><th>When</th><th>Patient</th><th>Case</th><th></th></tr></thead><tbody>
      ${upcoming.map((a) => `<tr><td class="num">${fmtDate(a.date)}<br><span class="fine">${esc(a.slot)}</span></td>
        <td>${esc(a.patient_name)}<br><span class="fine">${esc(a.id)}</span></td>
        <td>${a.case ? `<span class="tag ${a.case.triage}">${esc(a.case.name.split(" /")[0])}</span>` : `<span class="fine">None</span>`}</td>
        <td><button class="linkbtn" data-cancel="${esc(a.id)}">Cancel</button></td></tr>`).join("")}</tbody></table>`;
}

$("booked").addEventListener("click", async (e) => {
  const id = e.target.dataset.cancel;
  if (!id) return;
  await api(`/api/appointments/${encodeURIComponent(id)}`, { method: "DELETE" });
  hospital = await api("/api/hospital");
  renderDates();
  if (chosenDate) renderSlots();
  loadBooked();
});

init();
