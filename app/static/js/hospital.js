// Hospital page: services, schedule and the review queue.

const DAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

api("/api/hospital").then((h) => {
  document.getElementById("h-name").textContent = h.name;
  document.getElementById("h-note").textContent = h.note;
  document.getElementById("services").innerHTML = h.services.map((s) =>
    `<div><h3>${esc(s.name)}</h3><p>${esc(s.text)}</p></div>`).join("");
  document.getElementById("schedule").innerHTML = `
    <thead><tr><th>Role</th><th>Days</th><th>Hours</th></tr></thead>
    <tbody>${h.schedule.map((s) => `<tr><td>${esc(s.role)}</td>
      <td><div class="week">${DAY.map((d, i) => `<span class="${s.days.includes(i) ? "on" : ""}" title="${d}">${d[0]}</span>`).join("")}</div></td>
      <td class="num nowrap">${esc(s.hours)}</td></tr>`).join("")}</tbody>`;
});

const STATUS = { new: "Not reviewed", queued: "In queue", reviewed: "Reviewed" };

async function loadQueue() {
  const [cases, appts] = await Promise.all([api("/api/cases"), api("/api/appointments")]);
  const booked = Object.fromEntries(appts.filter((a) => a.case_id).map((a) => [a.case_id, a]));
  const table = document.getElementById("queue-table");
  const head = `<thead><tr><th></th><th>Case</th><th>Analysed</th><th>Most likely</th><th class="n">P(melanoma)</th>
    <th>Suggestion</th><th>Status</th><th>Appointment</th><th></th></tr></thead>`;
  if (!cases.length) {
    table.innerHTML = head + `<tbody><tr><td class="empty" colspan="9">No cases yet.
      <a href="/analyse">Analyse an image</a> and it will appear here.</td></tr></tbody>`;
    return;
  }
  table.innerHTML = head + `<tbody>${cases.map((c) => {
    const a = booked[c.id];
    return `<tr>
      <td><img class="queue-thumb" src="${esc(c.thumb)}" alt=""></td>
      <td class="case-id">${esc(c.id)}</td>
      <td class="num nowrap">${fmtTime(c.created)}</td>
      <td>${esc(c.name.split(" /")[0])}</td>
      <td class="n">${pct(c.mel_prob)}</td>
      <td><span class="tag ${c.triage}">${c.triage === "refer" ? "Refer" : "Routine"}</span></td>
      <td><span class="tag ${c.status}">${STATUS[c.status]}</span></td>
      <td class="nowrap">${a ? `${fmtDate(a.date)}, ${esc(a.slot)}` : `<a class="linkbtn" href="/appointments?case=${encodeURIComponent(c.id)}">Book</a>`}</td>
      <td>${c.status === "reviewed" ? "" : `<button class="linkbtn" data-review="${esc(c.id)}">Mark reviewed</button>`}</td>
    </tr>`;
  }).join("")}</tbody>`;
}

document.getElementById("queue-table").addEventListener("click", async (e) => {
  const id = e.target.dataset.review;
  if (!id) return;
  await api(`/api/cases/${encodeURIComponent(id)}/status`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "reviewed" }) });
  loadQueue();
});

loadQueue();
