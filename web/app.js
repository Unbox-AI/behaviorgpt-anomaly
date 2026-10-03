const $ = (sel) => document.querySelector(sel);
const LN10 = Math.log(10);
// mirrors the tokens in styles.css
const PALETTE = {
  ink: "#2c2c2b", path: "#123252", accent: "#2f6bff", accentSoft: "#9bb9ff",
  ok: "#3f7d63", warn: "#f5a623", hold: "#b4353a", holdSoft: "#fbecec", line: "#dcd9d8",
};
const COLORS = {
  normal: ["#2f6bff", "#3f7d63", "#49a5e6", "#59a83e"],
  fraud: ["#b4353a", "#e2893c", "#b73f90", "#6a2fe5", "#ec6a5a", "#6e5e1f"],
};
const TYPE_LABEL = {
  search: "Search", view: "View", add_to_cart: "Add to cart",
  remove_from_cart: "Remove", order: "Order",
};
const STATUS_LABEL = { context: "context", funnel: "funnel", order: "order", unknown: "unknown" };

const state = {
  meta: null,
  examples: [],
  session: { label: "Session", kind: null, events: [], exampleId: null, notes: [], anomalyStart: null },
  report: null,
  items: new Map(),
  selected: null,
  pinned: [],
  scoreSeq: 0,
};

// ---------- helpers ----------

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  for (const c of children.flat()) if (c) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

function thumb(item) {
  if (item && item.image_url) {
    const img = el("img", { src: item.image_url, alt: "", loading: "lazy", referrerpolicy: "no-referrer" });
    img.addEventListener("error", () => img.replaceWith(el("div", { class: "ph" })));
    return img;
  }
  return el("div", { class: "ph" });
}

function node(ev, item, quiet) {
  if (ev.type === "search") {
    return el("div", { class: "node search", title: "Search" },
      svg("svg", { viewBox: "0 0 16 16", width: 15, height: 15, fill: "none", stroke: "currentColor", "stroke-width": 1.4, "stroke-linecap": "round" },
        svg("circle", { cx: 6.8, cy: 6.8, r: 4.6 }), svg("line", { x1: 10.2, y1: 10.2, x2: 14, y2: 14 })));
  }
  if (ev.type === "order") return el("div", { class: "node glyph quiet", title: "Order" });
  return el("div", { class: `node ${quiet ? "quiet" : ""}` }, thumb(item));
}

function price(item) {
  const p = Number(item && item.price);
  return p > 0 ? `$${p.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : null;
}

function fmt(x, digits = 2) {
  return x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(digits);
}

function uClass(u) {
  if (u === null || u === undefined) return "na";
  return u >= 2.5 ? "hi" : u >= 1 ? "mid" : "lo";
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 5000);
}

function busy(text) {
  $("#busy").hidden = !text;
  if (text) $("#busy-text").textContent = text;
}

function remember(item) {
  if (item && item.id) state.items.set(item.id, item);
}

function ticks(max) {
  const step = max <= 2 ? 0.5 : max <= 5 ? 1 : 2;
  const out = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
  return out;
}

function verdictOf(summary) {
  if (!summary || !summary.n_scored) return { cls: "", text: "Not scored" };
  if (summary.flagged) return { cls: "bad", text: "Flagged" };
  if (summary.peak_risk >= 0.6) return { cls: "warn", text: "Elevated" };
  return { cls: "ok", text: "Normal" };
}

// ---------- scoring ----------

async function rescore() {
  const seq = ++state.scoreSeq;
  const { events, exampleId } = state.session;
  if (!events.length) {
    state.report = null;
    render();
    return;
  }
  busy("Scoring…");
  try {
    const report = await api("/api/score", { events, example_id: exampleId });
    if (seq !== state.scoreSeq) return;
    report.steps.forEach((s) => remember(s.item));
    state.report = report;
  } catch (err) {
    if (seq === state.scoreSeq) toast(err.message);
  } finally {
    if (seq === state.scoreSeq) busy(null);
  }
  render();
}

function setSession(session) {
  state.session = { notes: [], anomalyStart: null, exampleId: null, kind: null, ...session };
  state.report = null;
  render();
  return rescore();
}

function edit(mutator) {
  const s = state.session;
  mutator(s.events);
  if (s.exampleId || s.kind) {
    s.label = s.label.endsWith("(edited)") ? s.label : `${s.label} (edited)`;
  }
  s.exampleId = null;
  s.anomalyStart = null;
  rescore();
}

// ---------- scenario bar ----------

function renderScenarioBar() {
  const ex = $("#examples");
  ex.replaceChildren(
    ...state.examples.map((e) =>
      el("button", {
        class: `chip ${state.session.exampleId === e.id ? "active" : ""}`,
        onclick: () => loadExample(e),
      }, el("span", { class: "dot", style: `background:${e.kind === "normal" ? "var(--ok)" : "var(--hold)"}` }), e.label)
    ),
  );
  if (!state.examples.length) ex.append(el("span", { class: "muted small" }, "Run scripts/calibrate.py to build examples."));

  const live = state.meta && state.meta.live;
  const sim = $("#simulate");
  sim.replaceChildren(
    ...Object.entries(state.meta ? state.meta.kinds : {}).map(([kind, label]) =>
      el("button", {
        class: "chip",
        disabled: !live,
        title: live ? `Simulate a new ${label.toLowerCase()} session` : "Needs an UNBOXAI_API_KEY",
        onclick: () => simulate(kind),
      }, el("span", { class: "dot", style: `background:${kind === "normal" ? "var(--ok)" : "var(--hold)"}` }), label)
    ),
  );
}

function loadExample(e) {
  return setSession({
    label: e.label, kind: e.kind, events: e.events.map((x) => ({ ...x })),
    exampleId: e.id, notes: e.notes, anomalyStart: e.anomaly_start,
  });
}

async function simulate(kind) {
  busy("Simulating a shopper…");
  try {
    const sc = await api("/api/simulate", { kind });
    await setSession({
      label: sc.label, kind: sc.kind, events: sc.events, notes: sc.notes, anomalyStart: sc.anomaly_start,
    });
  } catch (err) {
    busy(null);
    toast(err.message);
  }
}

// ---------- timeline ----------

function renderTimeline() {
  const { session, report } = state;
  $("#session-label").textContent = session.label;
  $("#session-notes").textContent = (session.notes || []).join(" ");
  const list = $("#timeline");
  if (!session.events.length) {
    list.replaceChildren(el("li", { class: "empty" }, "Pick an example, simulate a shopper, or add events below."));
    return;
  }
  const flaggedAt = report ? report.summary.flagged_at : null;
  const editable = state.meta && state.meta.live;
  list.replaceChildren(
    ...session.events.map((ev, i) => {
      const step = report && report.steps[i] && report.steps[i].event.type === ev.type ? report.steps[i] : null;
      const item = ev.item ? (step && step.item) || state.items.get(ev.item) || { id: ev.item, name: ev.item } : null;
      const title = ev.type === "search" ? `search “${ev.query}”` : ev.type === "order" ? "Checkout: order the cart" : item.name;
      const titleNode = ev.type === "search" ? el("code", {}, title) : title;
      const meta = [
        ev.type === "search" ? null : el("span", { class: `type ${ev.type}` }, TYPE_LABEL[ev.type]),
        item && price(item) ? el("span", {}, price(item)) : null,
        item && item.categories ? el("span", {}, item.categories) : null,
      ];
      let side;
      if (!step) side = el("span", { class: "u na" }, "…");
      else if (step.status === "scored") {
        side = el("span", {
          class: `u ${uClass(step.u)}`,
          title: `surprise ${fmt(step.u)}σ · rank #${Math.round(step.rank).toLocaleString()} · p = ${Math.exp(step.logp).toExponential(1)}`,
        }, `${step.u >= 0 ? "+" : ""}${fmt(step.u, 1)}σ`);
      } else side = el("span", { class: "u na", title: step.reasons.join(" ") }, STATUS_LABEL[step.status]);

      const tools = editable ? el("div", { class: "step-tools" },
        el("button", { class: "icon", title: "Move up", disabled: i === 0, onclick: () => edit((evs) => evs.splice(i - 1, 0, evs.splice(i, 1)[0])) }, "▲"),
        el("button", { class: "icon", title: "Remove", onclick: () => edit((evs) => evs.splice(i, 1)) }, "✕"),
        el("button", { class: "icon", title: "Move down", disabled: i === session.events.length - 1, onclick: () => edit((evs) => evs.splice(i + 1, 0, evs.splice(i, 1)[0])) }, "▼"),
      ) : null;

      const cls = ["step", flaggedAt === i ? "flag" : "", session.anomalyStart === i ? "fraud-start" : ""].join(" ");
      return el("li", { class: cls, id: `step-${i}` },
        el("span", { class: "step-n" }, i + 1),
        node(ev, item, step && step.status !== "scored"),
        el("div", { class: "step-body" }, el("div", { class: "step-title", title }, titleNode), el("div", { class: "step-meta" }, meta)),
        el("div", { class: "step-side" }, side, tools),
      );
    }),
  );
}

function highlightStep(i) {
  document.querySelectorAll(".step.highlight").forEach((n) => n.classList.remove("highlight"));
  const node = $(`#step-${i}`);
  if (!node) return;
  node.classList.add("highlight");
  node.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------- verdict ----------

function renderVerdict() {
  const { report, meta } = state;
  const head = $("#verdict-head");
  const summary = report && report.summary;
  if (!summary || !summary.n_scored) {
    head.replaceChildren(el("div", {},
      el("div", { class: "verdict-title" }, "No verdict yet"),
      el("div", { class: "verdict-sub" }, "Views and add-to-carts are the scored choices; add one to see the risk."),
    ));
    ["#risk-meter", "#chart", "#metrics", "#drivers"].forEach((s) => $(s).replaceChildren());
    return;
  }
  const v = verdictOf(summary);
  const fpr = meta ? Math.round(meta.calibration.target_fpr * 1000) / 10 : null;
  const steps = report.steps;
  let title, sub;
  if (summary.flagged) {
    const at = steps[summary.flagged_at];
    const held = steps.slice(summary.flagged_at).some((s) => s.event.type === "order");
    title = `Flagged at step ${summary.flagged_at + 1}`;
    sub = `Crossed the risk barrier on ${at.item ? `“${at.item.name}”` : "this step"}.` +
      (held ? " The order that follows should be held for review." : "");
  } else {
    title = v.cls === "warn" ? "Elevated, below the barrier" : "Looks like a normal shopper";
    sub = `Peak risk ${fmt(summary.peak_risk * 100, 0)}% of the barrier.`;
  }
  if (fpr !== null) sub += ` The barrier flags ${fpr}% of normal shoppers.`;
  head.replaceChildren(
    el("div", {}, el("div", { class: "verdict-title" }, title), el("div", { class: "verdict-sub" }, sub)),
    el("span", { class: `pill ${v.cls}` }, summary.flagged && steps.slice(summary.flagged_at).some((s) => s.event.type === "order") ? "Hold order" : v.text),
  );

  const max = 1.5;
  const fill = Math.min(summary.peak_risk / max, 1);
  const over = summary.peak_risk > 1;
  $("#risk-meter").replaceChildren(
    el("div", { class: "meter" },
      el("div", { class: `fill ${over ? "over" : ""}`, style: `width:${fill * 100}%;--b:${(1 / max / fill) * 100}%` }),
      el("div", { class: "barrier", style: `left:${(1 / max) * 100}%` }),
    ),
    el("div", { class: "meter-labels" },
      el("span", {}, "0%"),
      el("span", { class: "barrier-label", style: `left:${(1 / max) * 100}%` }, "barrier"),
      el("span", {}, `${max * 100}%`),
    ),
  );

  $("#chart").replaceChildren(sessionChart(report, state.session.anomalyStart));
  renderMetrics(summary);
  renderDrivers(report);
}

function sessionChart(report, anomalyStart) {
  const steps = report.steps;
  const W = 720, padL = 34, padR = 12, riskH = 150, gap = 36, barH = 70, top = 10;
  const H = top + riskH + gap + barH + 22;
  const n = steps.length;
  const xw = (W - padL - padR) / Math.max(n, 1);
  const x = (i) => padL + xw * (i + 0.5);
  const peak = Math.max(1.5, ...steps.map((s) => s.risk * 1.1));
  const yr = (r) => top + riskH - (r / peak) * riskH;
  const barTop = top + riskH + gap;
  const uMax = 6, uMin = -3;
  const yu = (u) => barTop + ((uMax - u) / (uMax - uMin)) * barH;

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Risk over the session" });
  // risk panel
  for (const r of ticks(peak)) {
    root.append(svg("line", { class: "gridline", x1: padL, x2: W - padR, y1: yr(r), y2: yr(r) }));
    root.append(svg("text", { class: "axis", x: padL - 6, y: yr(r) + 3, "text-anchor": "end" }, r.toFixed(1)));
  }
  if (anomalyStart !== null && anomalyStart !== undefined && anomalyStart < n) {
    root.append(svg("rect", { x: padL + xw * anomalyStart, y: top, width: xw * (n - anomalyStart), height: riskH, fill: PALETTE.holdSoft, opacity: 0.8 }));
    root.append(svg("text", { class: "legend", x: padL + xw * anomalyStart + 6, y: top + riskH - 6, fill: PALETTE.hold, style: `fill:${PALETTE.hold}` }, "FRAUD"));
  }
  root.append(svg("line", { x1: padL, x2: W - padR, y1: yr(1), y2: yr(1), stroke: PALETTE.ink, "stroke-width": 1, "stroke-dasharray": "4 4" }));
  root.append(svg("text", { class: "legend strong", x: W - padR, y: yr(1) - 6, "text-anchor": "end" }, "RISK BARRIER"));
  const pts = steps.map((s, i) => `${x(i)},${yr(s.risk)}`).join(" ");
  root.append(svg("polyline", { points: pts, fill: "none", stroke: PALETTE.path, "stroke-width": 1.75, "stroke-linejoin": "round", "stroke-linecap": "round" }));
  steps.forEach((s, i) => {
    const flagged = report.summary.flagged_at === i;
    root.append(svg("circle", {
      cx: x(i), cy: yr(s.risk), r: flagged ? 6 : 3,
      fill: flagged || s.risk >= 1 ? PALETTE.hold : "#fff", stroke: flagged || s.risk >= 1 ? "#fff" : PALETTE.path, "stroke-width": 1.5,
    }, svg("title", {}, `Step ${i + 1}: risk ${(s.risk * 100).toFixed(0)}% of the barrier`)));
  });
  root.append(svg("text", { class: "legend", x: padL + 4, y: top + 10 }, "RISK · CUSUM ÷ BARRIER"));

  // surprise bars
  root.append(svg("text", { class: "legend", x: padL, y: barTop - 8 }, "SURPRISE PER STEP · σ ABOVE NORMAL"));
  root.append(svg("line", { class: "gridline", x1: padL, x2: W - padR, y1: yu(0), y2: yu(0) }));
  root.append(svg("text", { class: "axis", x: padL - 6, y: yu(0) + 3, "text-anchor": "end" }, "0"));
  root.append(svg("text", { class: "axis", x: padL - 6, y: yu(uMax) + 8, "text-anchor": "end" }, `+${uMax}`));
  steps.forEach((s, i) => {
    if (s.status !== "scored") {
      root.append(svg("circle", { cx: x(i), cy: yu(0), r: 2, fill: PALETTE.line }, svg("title", {}, `Step ${i + 1}: ${s.status}`)));
      return;
    }
    const y0 = yu(0), y1 = yu(s.u);
    const color = s.u >= 2.5 ? PALETTE.hold : s.u >= 1 ? PALETTE.warn : PALETTE.accentSoft;
    const bar = svg("rect", {
      class: "bar", x: x(i) - Math.min(xw * 0.35, 14), width: Math.min(xw * 0.7, 28),
      y: Math.min(y0, y1), height: Math.max(Math.abs(y1 - y0), 1.5), fill: color,
    }, svg("title", {}, `Step ${i + 1}: ${s.item ? s.item.name : ""}\n${s.reasons.join("\n")}`));
    bar.addEventListener("click", () => highlightStep(i));
    root.append(bar);
  });
  // x labels
  const every = Math.ceil(n / 20);
  steps.forEach((_, i) => {
    if (i % every === 0 || i === n - 1) {
      root.append(svg("text", { class: "axis", x: x(i), y: H - 6, "text-anchor": "middle" }, String(i + 1)));
    }
  });
  return root;
}

function renderMetrics(s) {
  const log10p = s.log_prob / LN10;
  const pct = s.percentile;
  const cards = [
    ["P(sequence)", el("span", {}, "10", el("sup", {}, fmt(log10p, 1))), `product of ${s.n_scored} choice probabilities`],
    ["Surprisal per choice", `${fmt(s.mean_surprisal)} nats`, `≈ 1 in ${Math.round(s.perplexity).toLocaleString()} (perplexity)`],
    ["Excess surprisal", `${s.mean_excess >= 0 ? "+" : ""}${fmt(s.mean_excess)} nats`, "per choice, vs the model's own uncertainty"],
    ["Session z-score", fmt(s.z), "length-normalized: Σu / √n"],
    ["vs normal shoppers", pct === null ? "–" : `${fmt(pct * 100, 0)}th pct`, pct === null ? "not calibrated yet" : `more unusual than ${fmt(pct * 100, 0)}% of normal sessions`],
    ["Peak risk", `${fmt(s.peak_risk * 100, 0)}%`, "of the barrier; 100% raises the flag"],
  ];
  $("#metrics").replaceChildren(
    ...cards.map(([label, value, hint]) => el("div", { class: "metric" },
      el("div", { class: "label" }, label), el("div", { class: "value" }, value), el("div", { class: "hint" }, hint))),
  );
}

function renderDrivers(report) {
  const box = $("#drivers");
  if (!report.contributors.length) {
    box.replaceChildren(el("p", { class: "calm" }, "No step is more surprising than a normal shopper's: every choice fits the history before it."));
    return;
  }
  box.replaceChildren(
    ...report.contributors.map((c) => {
      const step = report.steps[c.index];
      const item = c.item || {};
      const expected = (step.expected || []).map((e) => {
        remember(e);
        return el("span", { class: "exp", title: e.name }, thumb(e), el("span", {}, e.name));
      });
      return el("div", { class: `driver ${uClass(step.u)}` },
        el("div", { class: "node" }, thumb(item)),
        el("div", {},
          el("div", { class: "driver-share" }, `Step ${c.index + 1} · ${TYPE_LABEL[c.event.type]} · ${fmt(c.share * 100, 0)}% of the excess surprise`),
          el("div", { class: "driver-title" }, item.name || c.event.item),
          el("ul", {}, c.reasons.map((r) => el("li", {}, r))),
          expected.length ? el("div", { class: "expected" }, el("span", { class: "eyebrow" }, "Model expected"), expected) : null,
        ),
      );
    }),
  );
  box.querySelectorAll(".driver").forEach((node, k) =>
    node.addEventListener("click", () => highlightStep(report.contributors[k].index)));
}

// ---------- compare ----------

function pinCurrent() {
  if (!state.report) return toast("Score a session first.");
  const kind = state.session.kind || "custom";
  const palette = kind === "normal" ? COLORS.normal : COLORS.fraud;
  const used = state.pinned.filter((p) => (p.kind === "normal") === (kind === "normal")).length;
  state.pinned.push({
    label: state.session.label, kind, report: state.report,
    color: palette[used % palette.length], key: `${state.session.label}-${Date.now()}`,
  });
  if (state.pinned.length > 8) state.pinned.shift();
  renderCompare();
}

function renderCompare() {
  const box = $("#compare-chart");
  const table = $("#compare-table");
  if (!state.pinned.length) {
    box.replaceChildren(el("p", { class: "calm" }, "Pin sessions to compare them."));
    table.replaceChildren();
    return;
  }
  const W = 720, H = 260, padL = 34, padR = 12, top = 12, bottom = 24;
  const CAP = 3;
  const maxN = Math.max(...state.pinned.map((p) => p.report.steps.length));
  const peak = Math.min(CAP, Math.max(1.5, ...state.pinned.flatMap((p) => p.report.steps.map((s) => s.risk * 1.05))));
  const x = (i) => padL + (i / Math.max(maxN - 1, 1)) * (W - padL - padR);
  const y = (r) => top + (H - top - bottom) * (1 - Math.min(r, peak) / peak);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const r of ticks(peak)) {
    root.append(svg("line", { class: "gridline", x1: padL, x2: W - padR, y1: y(r), y2: y(r) }));
    root.append(svg("text", { class: "axis", x: padL - 6, y: y(r) + 3, "text-anchor": "end" }, r.toFixed(1)));
  }
  if (peak === CAP) {
    root.append(svg("text", { class: "legend", x: padL + 4, y: top + 10 }, `RISK · CAPPED AT ${CAP}× THE BARRIER`));
  }
  root.append(svg("line", { x1: padL, x2: W - padR, y1: y(1), y2: y(1), stroke: PALETTE.ink, "stroke-width": 1, "stroke-dasharray": "4 4" }));
  root.append(svg("text", { class: "legend strong", x: W - padR, y: y(1) - 6, "text-anchor": "end" }, "RISK BARRIER"));
  for (const p of state.pinned) {
    const steps = p.report.steps;
    root.append(svg("polyline", {
      points: steps.map((s, i) => `${x(i)},${y(s.risk)}`).join(" "),
      fill: "none", stroke: p.color, "stroke-width": 1.75, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.9,
    }, svg("title", {}, p.label)));
    const f = p.report.summary.flagged_at;
    if (f !== null) root.append(svg("circle", { cx: x(f), cy: y(steps[f].risk), r: 5, fill: p.color, stroke: "#fff", "stroke-width": 1.5 }, svg("title", {}, `${p.label}: flagged at step ${f + 1}`)));
  }
  for (let i = 0; i < maxN; i += Math.ceil(maxN / 15)) {
    root.append(svg("text", { class: "axis", x: x(i), y: H - 6, "text-anchor": "middle" }, String(i + 1)));
  }
  box.replaceChildren(root);

  table.replaceChildren(
    el("thead", {}, el("tr", {}, ["Session", "Steps", "z", "vs normal", "Peak", "Verdict", ""].map((h) => el("th", {}, h)))),
    el("tbody", {}, state.pinned.map((p, k) => {
      const s = p.report.summary;
      const v = verdictOf(s);
      return el("tr", {},
        el("td", {}, el("span", { class: "swatch", style: `background:${p.color}` }), p.label),
        el("td", { class: "num" }, s.n_events),
        el("td", { class: "num" }, fmt(s.z, 1)),
        el("td", { class: "num" }, s.percentile === null ? "–" : `${fmt(s.percentile * 100, 0)}%`),
        el("td", { class: "num" }, `${fmt(s.peak_risk * 100, 0)}%`),
        el("td", {}, p.kind !== "normal" && p.kind !== "custom" && !s.flagged
          ? el("span", { class: "pill small warn", title: "A simulated fraud the barrier did not flag" }, "missed")
          : el("span", { class: `pill small ${v.cls}` }, s.flagged ? `step ${s.flagged_at + 1}` : v.text)),
        el("td", {}, el("button", { class: "icon", title: "Unpin", onclick: () => { state.pinned.splice(k, 1); renderCompare(); } }, "✕")),
      );
    })),
  );
}

// ---------- composer ----------

function setupComposer() {
  const type = $("#event-type");
  const input = $("#item-query");
  const results = $("#item-results");
  let timer = null;
  let active = -1;

  const syncType = () => {
    const t = type.value;
    input.disabled = t === "order";
    input.placeholder = t === "search" ? "Type a search query…" : t === "order" ? "Orders the current cart" : "Find an item in the store…";
    results.hidden = true;
    state.selected = null;
  };
  type.addEventListener("change", syncType);

  input.addEventListener("input", () => {
    state.selected = null;
    if (type.value === "search") return;
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { results.hidden = true; return; }
    timer = setTimeout(async () => {
      const hits = await api(`/api/items/search?q=${encodeURIComponent(q)}`);
      active = -1;
      results.replaceChildren(...hits.map((it) => {
        remember(it);
        return el("li", { onmousedown: (e) => { e.preventDefault(); pick(it); } },
          thumb(it), el("span", { class: "name", title: it.name }, it.name), el("span", { class: "price" }, price(it) || ""));
      }));
      if (!hits.length) results.replaceChildren(el("li", {}, el("span"), el("span", { class: "muted" }, "No items match."), el("span")));
      results.hidden = false;
    }, 150);
  });
  input.addEventListener("keydown", (e) => {
    const rows = [...results.querySelectorAll("li")];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      active = Math.max(0, Math.min(rows.length - 1, active + (e.key === "ArrowDown" ? 1 : -1)));
      rows.forEach((r, k) => r.classList.toggle("active", k === active));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (!results.hidden && active >= 0) rows[active].dispatchEvent(new MouseEvent("mousedown"));
      else addEvent();
    } else if (e.key === "Escape") results.hidden = true;
  });
  input.addEventListener("blur", () => setTimeout(() => (results.hidden = true), 100));

  function pick(item) {
    state.selected = item;
    input.value = item.name;
    results.hidden = true;
  }

  function addEvent() {
    const t = type.value;
    let ev;
    if (t === "order") ev = { type: "order" };
    else if (t === "search") {
      if (!input.value.trim()) return toast("Type a query first.");
      ev = { type: "search", query: input.value.trim() };
    } else {
      if (!state.selected) return toast("Pick an item from the list first.");
      ev = { type: t, item: state.selected.id };
    }
    input.value = "";
    state.selected = null;
    edit((evs) => evs.push(ev));
  }

  async function addSuggested(odd) {
    const t = ["view", "add_to_cart"].includes(type.value) ? type.value : "view";
    busy(odd ? "Picking an odd item…" : "Simulating the next action…");
    try {
      const temperature = Number($("#variety").value);
      const ev = await api("/api/next", { events: state.session.events, odd, type: t, temperature });
      if (ev.item) remember(await api(`/api/items/${encodeURIComponent(ev.item)}`));
      edit((evs) => evs.push(ev));
    } catch (err) {
      busy(null);
      toast(err.message);
    }
  }

  $("#add-event").addEventListener("click", addEvent);
  $("#add-likely").addEventListener("click", () => addSuggested(false));
  $("#add-odd").addEventListener("click", () => addSuggested(true));
  $("#variety").addEventListener("input", (e) => ($("#variety-value").textContent = `${e.target.value}×`));
  syncType();
}

function renderComposer() {
  const live = state.meta && state.meta.live;
  for (const id of ["#event-type", "#item-query", "#add-event", "#add-likely", "#add-odd", "#variety"]) $(id).disabled = !live;
  if (live) $("#event-type").dispatchEvent(new Event("change"));
  $("#composer-hint").textContent = live
    ? "Edits are scored live."
    : "Read-only: set UNBOXAI_API_KEY to edit sequences and simulate shoppers.";
}

// ---------- about ----------

function renderAbout() {
  const m = state.meta;
  const c = m.calibration;
  const r = c.report || {};
  const rows = Object.values(r.patterns || {}).map((p) => el("tr", {},
    el("td", {}, p.label), el("td", {}, p.sessions),
    el("td", {}, `${fmt(p.detection_rate * 100, 0)}%`),
    el("td", {}, p.median_delay_steps === null ? "–" : fmt(p.median_delay_steps, 0)),
    el("td", {}, fmt(p.auc_peak_risk)), el("td", {}, fmt(p.auc_session_z)),
  ));
  $("#about-body").replaceChildren(
    el("p", {}, "BehaviorGPT reads a shopper's history and ranks every item by how likely it is to be the next one they act on. A normal shopper's choices sit high in those rankings; a fraudster, a bot or a hijacked account picks things the history does not explain. This app turns that into a probability for each step, a risk score for the session, and a flag at the step it crosses the barrier."),
    el("h4", {}, "1. Probability of each choice"),
    el("p", {}, `The API returns scores, not probabilities. At each step the app asks for the model's top 100 items and, through an id filter, the scores of the chosen item and of 99 items sampled uniformly from the store (${m.store_size.toLocaleString()} items of ${m.catalog_id}). A softmax with a fitted temperature turns scores into probabilities; the tail of the catalog is estimated from the sample.`),
    el("div", { class: "formula" }, "p(item | history) = exp(s/τ) / [ Σ_top100 exp(s/τ) + (N − 100) · mean_sample exp(s/τ) ]\nτ is fitted by maximum likelihood on normal shoppers: " + Object.entries(c.tau).map(([k, v]) => `${k} τ = ${fmt(v, 4)}`).join(", ")),
    el("h4", {}, "2. Normalizing for context and length"),
    el("p", {}, "Surprisal −log p is high after a broad search and low after a specific view, so each step is compared to the entropy of the model's own prediction: excess surprisal = surprisal − entropy, zero on average for a shopper who behaves like the model expects. It is standardized with the median and spread measured on normal shoppers (u, in σ). The session z-score Σu/√n stays on the same scale for a 5-step and a 30-step session."),
    el("p", {}, "Only choices are scored: the first time an item is touched. Searches set context, and add-to-cart after a view, removals and orders are steps down the funnel for a choice already made."),
    el("h4", {}, "3. The risk barrier"),
    el("p", {}, `Risk accumulates with a one-sided CUSUM, S = max(0, S + u − ${c.cusum_k}): evidence builds while behavior stays off and drains while it is normal, so a session that turns halfway (account takeover) is caught when it turns rather than diluted by its normal start. The barrier is the CUSUM level that only ${fmt(c.target_fpr * 100, 0)}% of normal shoppers ever reach (here ${fmt(c.barrier)}); risk is shown as a share of it.`),
    el("h4", {}, "4. Explanations"),
    el("p", {}, "The steps that add the most excess surprise are listed with their probability and rank, what changed (a new category, a price far above the session's, ignored search results, carting without viewing), and the items the model expected at that point."),
    el("h4", {}, "How well it works"),
    r.false_alarm_rate === undefined
      ? el("p", { class: "muted" }, "Not calibrated yet: run scripts/calibrate.py.")
      : el("div", {},
        el("p", {}, `On ${r.normal_sessions.held_out} held-out normal sessions the barrier flagged ${fmt(r.false_alarm_rate * 100, 1)}%. Simulated fraud:`),
        el("table", {}, el("thead", {}, el("tr", {}, ["Pattern", "Sessions", "Flagged", "Median delay (steps)", "AUC risk", "AUC z"].map((h) => el("th", {}, h)))), el("tbody", {}, rows)),
        el("p", { class: "fine" }, "Shoppers are simulated (see anomaly/simulate.py): normal ones search, click results with position bias and follow recommendations; the fraud patterns are scripted. Real traffic will differ; recalibrate on your own sessions before using the barrier."),
      ),
  );
}

// ---------- render ----------

function render() {
  renderScenarioBar();
  renderTimeline();
  renderVerdict();
}

async function init() {
  setupComposer();
  $("#about-open").addEventListener("click", () => $("#about").showModal());
  $("#about-close").addEventListener("click", () => $("#about").close());
  $("#about").addEventListener("close", () => {
    if (location.hash === "#how-it-works") history.replaceState(null, "", location.pathname + location.search);
  });
  $("#pin").addEventListener("click", pinCurrent);
  $("#clear").addEventListener("click", () => setSession({ label: "Custom session", events: [] }));

  const [meta, examples] = await Promise.all([api("/api/meta"), api("/api/examples")]);
  state.meta = meta;
  state.examples = examples;
  $("#catalog-id").textContent = meta.catalog_id;
  const badge = $("#live-badge");
  badge.textContent = meta.live ? "Live" : "Examples only";
  badge.title = meta.live ? "An UNBOXAI_API_KEY is set: edits and simulations are scored live" : "No API key: only the bundled examples can be scored";
  badge.classList.toggle("live", meta.live);
  renderAbout();
  if (location.hash === "#how-it-works") $("#about").showModal();
  renderComposer();
  render();
  renderCompare();

  // pin every example, so normal and abnormal sit side by side from the start
  for (const ex of examples) {
    await loadExample(ex);
    pinCurrent();
  }
  const wanted = new URLSearchParams(location.search).get("example");
  const first = examples.find((e) => e.id === wanted)
    || examples.find((e) => e.kind === "account_takeover") || examples[0];
  if (first) await loadExample(first);
}

init().catch((err) => toast(err.message));
