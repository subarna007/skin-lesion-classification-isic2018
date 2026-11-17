// Shared helpers: header, footer, API calls, formatting, icons.

const NAV = [
  ["/", "Research"],
  ["/analyse", "Analyse"],
  ["/hospital", "Hospital"],
  ["/appointments", "Appointments"],
];

const ICONS = {
  lens: `<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="13.5" fill="none" stroke="#17202a" stroke-width="3"/><circle cx="16" cy="16" r="8" fill="#0e6b66"/><circle cx="13.2" cy="13.2" r="2.4" fill="#e2efed"/></svg>`,
  upload: `<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>`,
  refer: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>`,
  routine: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m8 12 3 3 5-6"/></svg>`,
  github: `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-6 0-1.2.5-2.3 1.2-3.1-.1-.4-.5-1.6.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0C17.3 4.7 18.3 5 18.3 5c.7 1.6.2 2.8.1 3.2.8.8 1.2 1.9 1.2 3.1 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .5z"/></svg>`,
  google: `<svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"/><path fill="#FF3D00" d="m6.3 14.7 6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/><path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2A12 12 0 0 1 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"/><path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3a12 12 0 0 1-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z"/></svg>`,
  youtube: `<svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#FF0000" d="M43.2 14.2a5 5 0 0 0-3.5-3.5C36.6 10 24 10 24 10s-12.6 0-15.7.7a5 5 0 0 0-3.5 3.5C4 17.3 4 24 4 24s0 6.7.8 9.8a5 5 0 0 0 3.5 3.5C11.4 38 24 38 24 38s12.6 0 15.7-.7a5 5 0 0 0 3.5-3.5c.8-3.1.8-9.8.8-9.8s0-6.7-.8-9.8z"/><path fill="#fff" d="m20 30 10.4-6L20 18z"/></svg>`,
  wikipedia: `<svg viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="21" fill="#f3f5f4" stroke="#17202a" stroke-width="2"/><text x="24" y="31" text-anchor="middle" font-family="Georgia, serif" font-size="21" fill="#17202a">W</text></svg>`,
  scholar: `<svg viewBox="0 0 48 48" aria-hidden="true"><path fill="#4285F4" d="M24 6 4 19.5l8 5.4V34c3.3 3 7.4 4.5 12 4.5S32.7 37 36 34v-9.1l4-2.7V34h4V19.5z"/><circle cx="24" cy="29" r="7" fill="#A0C3FF"/></svg>`,
};

function renderChrome() {
  const here = location.pathname.replace(/\/$/, "") || "/";
  const header = document.createElement("header");
  header.className = "site-header";
  header.innerHTML = `<div class="inner">
      <a class="wordmark" href="/">${ICONS.lens}<span>DermAssist</span></a>
      <nav class="site-nav" aria-label="Main">${NAV.map(([href, label]) =>
        `<a href="${href}"${(href === "/" ? here === "/" : here.startsWith(href)) ? ' aria-current="page"' : ""}>${label}</a>`).join("")}</nav>
      <span class="prototype-note">Research prototype, not a medical device</span>
    </div>`;
  document.body.prepend(header);
  const footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML = `<div class="inner">
      <p>DermAssist is a research prototype built by Subarna Dahal as a follow-up to an MSc capstone project.
         It is not a medical device and must not be used to diagnose or treat patients.</p>
      <p>Data: ISIC 2018 Task 3 / HAM10000 (CC BY-NC 4.0).</p>
    </div>`;
  document.body.append(footer);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (!res.ok) {
    let msg = body && body.detail ? body.detail : `Request failed (${res.status}).`;
    if (Array.isArray(msg)) msg = msg.map((d) => `${d.loc ? d.loc.slice(-1)[0] + ": " : ""}${d.msg}`).join("; ");
    throw new Error(msg);
  }
  return body;
}

const pct = (x, digits = 1) => (x == null || isNaN(x) ? "–" : (x * 100).toFixed(digits) + "%");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const LONG_DATE = new Intl.DateTimeFormat("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
const SHORT_DATE = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short" });
const fmtDate = (iso, long = false) => (long ? LONG_DATE : SHORT_DATE).format(new Date(iso + "T00:00:00"));
const fmtTime = (epoch) => new Date(epoch * 1000).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

renderChrome();
