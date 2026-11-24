const ARXIV_ID = "2611.29723";

const LOSS_COLOR = {
  "ApexID Hierarchical": "var(--apex)",
  "ArcFace": "var(--af)",
  "Dynamic ArcFace": "var(--daf)",
  "SphereFace2": "var(--sf2)",
};

const CORPUS = [
  ["HappyWhale", "Whale; dolphin", 41407, 6329, "marine"],
  ["CzechLynx", "Lynx", 37829, 319, "cat"],
  ["HumpbackWhaleID", "Whale", 13624, 2931, "marine"],
  ["WhaleSharkID", "Whale shark", 5958, 444, "marine"],
  ["IPanda50", "Panda", 5167, 48, "other"],
  ["LeopardID2022", "Leopard", 4953, 287, "cat"],
  ["BalearicLizard", "Lizard", 4359, 750, "herp"],
  ["GiraffeZebraID", "Giraffe; zebra", 4333, 1001, "savanna"],
  ["CTai", "Chimpanzee", 3609, 65, "other"],
  ["HyenaID2022", "Hyena", 2422, 241, "savanna"],
  ["NDD20", "Dolphin", 2122, 78, "marine"],
  ["SealID", "Seal", 1668, 55, "marine"],
  ["NyalaData", "Nyala", 1442, 172, "savanna"],
  ["Jaguars of Pantanal", "Jaguar", 1178, 31, "cat"],
  ["SalamanderID2025", "Salamander", 1078, 277, "herp"],
  ["Giraffes", "Giraffe", 892, 156, "savanna"],
  ["ReunionTurtles", "Sea turtle", 237, 79, "herp"],
  ["AmvrakikosTurtles", "Sea turtle", 141, 47, "herp"],
  ["ZakynthosTurtles", "Sea turtle", 114, 38, "herp"],
];
const TOTAL_IMAGES = 132533;
const GROUP_SHADES = {
  marine: ["#1f5f8b", "#2f77a8", "#4b90bf", "#6ba7cf", "#8dbcdc"],
  cat: ["#a8641c", "#c27c1f", "#d99a47"],
  savanna: ["#6d5a3d", "#8a7352", "#a68d69", "#c0a986"],
  herp: ["#127c7a", "#2b9590", "#52aca6", "#7cc2bc", "#a3d5d0"],
  other: ["#6f4f8f", "#8a6aa9"],
};

const CLOSED = [
  ["ConvNeXt-Small", [
    ["ArcFace", 62.24, 29.54, 14.57, 30.08, 59.36],
    ["Dynamic ArcFace", 61.75, 29.23, 14.51, 29.62, 58.72],
    ["SphereFace2", 63.60, 30.24, 15.31, 30.83, 60.78],
    ["ApexID Hierarchical", 64.89, 30.84, 15.71, 31.72, 62.13],
  ]],
  ["MIEW-ID v3", [
    ["ArcFace", 63.65, 32.09, 17.71, 32.60, 60.82],
    ["Dynamic ArcFace", 62.72, 31.51, 17.31, 32.06, 60.07],
    ["SphereFace2", 63.40, 31.89, 17.78, 32.44, 60.64],
    ["ApexID Hierarchical", 63.88, 32.54, 18.28, 33.23, 61.62],
    ["Pretrained", 58.14, 27.40, 13.74, 27.74, 55.54],
  ]],
  ["MegaDescriptor-L", [
    ["ArcFace", 58.20, 27.13, 13.05, 27.53, 55.52],
    ["Dynamic ArcFace", 55.44, 25.51, 11.75, 25.49, 52.34],
    ["SphereFace2", 57.13, 26.61, 12.47, 26.79, 54.43],
    ["ApexID Hierarchical", 61.18, 29.07, 14.69, 29.85, 58.52],
    ["Pretrained", 54.01, 23.62, 10.66, 24.11, 50.82],
  ]],
];
const METRICS = ["r1", "map", "minp", "idmap", "idr1"];
const METRIC_NAME = { r1: "Rank-1", map: "mAP", minp: "mINP", idmap: "ID-bal. mAP", idr1: "ID-bal. R-1" };

const OPEN = [
  ["ConvNeXt-S", [
    ["ArcFace", 0.424, 0.596, 0.447, 0.392, 0.098, 0.207],
    ["Dynamic ArcFace", 0.436, 0.597, 0.460, 0.397, 0.305, 0.338],
    ["SphereFace2", 0.441, 0.588, 0.463, 0.404, 0.305, 0.338],
    ["ApexID Hierarchical", 0.432, 0.591, 0.457, 0.403, 0.319, 0.322],
  ]],
  ["MIEW-ID v3", [
    ["ArcFace", 0.433, 0.582, 0.457, 0.409, 0.320, 0.400],
    ["Dynamic ArcFace", 0.444, 0.593, 0.479, 0.396, 0.331, 0.419],
    ["SphereFace2", 0.427, 0.594, 0.452, 0.400, 0.431, 0.537],
    ["ApexID Hierarchical", 0.430, 0.593, 0.465, 0.406, 0.379, 0.437],
    ["Pretrained", 0.451, 0.592, 0.474, 0.416, 0.405, 0.521],
  ]],
  ["MegaDescriptor-L", [
    ["ArcFace", 0.414, 0.595, 0.437, 0.384, 0.419, 0.501],
    ["Dynamic ArcFace", 0.443, 0.609, 0.473, 0.382, 0.475, 0.498],
    ["SphereFace2", 0.411, 0.567, 0.427, 0.383, 0.321, 0.346],
    ["ApexID Hierarchical", 0.434, 0.601, 0.463, 0.384, 0.307, 0.316],
    ["Pretrained", 0.428, 0.580, 0.457, 0.398, 0.317, 0.321],
  ]],
];

const ABLATION = {
  lambdas: ["0.025", "0.05", "0.075", "0.10", "0.15", "0.25", "0.50"],
  base: [63.603, 30.244, 15.310, 30.825, 60.779],
  rows: [
    [64.257, 30.499, 15.334, 31.113, 61.260],
    [64.262, 30.586, 15.375, 31.334, 61.382],
    [64.193, 30.542, 15.383, 31.114, 61.047],
    [64.602, 30.662, 15.531, 31.235, 61.409],
    [64.478, 30.546, 15.333, 31.257, 61.730],
    [64.796, 30.724, 15.493, 31.510, 62.303],
    [64.888, 30.835, 15.715, 31.724, 62.125],
  ],
};

const TRANSFER = [
  ["ConvNeXt-Small", [
    ["ArcFace", 15.39, 5.89], ["Dynamic ArcFace", 14.54, 5.32], ["SphereFace2", 15.38, 5.72], ["ApexID Hierarchical", 17.69, 6.63],
  ]],
  ["MIEW-ID v3", [
    ["ArcFace", 14.21, 4.74], ["Dynamic ArcFace", 12.66, 4.29], ["SphereFace2", 14.40, 4.73], ["ApexID Hierarchical", 14.21, 4.87], ["Original checkpoint", 14.12, 4.82],
  ]],
  ["MegaDescriptor-L", [
    ["ArcFace", 10.65, 4.12], ["Dynamic ArcFace", 9.38, 3.72], ["SphereFace2", 9.57, 3.87], ["ApexID Hierarchical", 13.79, 4.94], ["Original checkpoint", 8.63, 3.34],
  ]],
];

const SVG_NS = "http://www.w3.org/2000/svg";
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const fmt = new Intl.NumberFormat("en-US");

function el(tag, attrs = {}, children = []) {
  const node = tag.startsWith("svg:") ? document.createElementNS(SVG_NS, tag.slice(4)) : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

function radioGroup(container, onChange) {
  const buttons = [...container.querySelectorAll('[role="radio"]')];
  const select = (button, focus) => {
    for (const b of buttons) {
      const on = b === button;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    if (focus) button.focus();
    onChange(button);
  };
  buttons.forEach((b, i) => {
    b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1;
    b.addEventListener("click", () => select(b, false));
    b.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      select(buttons[(i + step + buttons.length) % buttons.length], true);
    });
  });
}

function whenVisible(node, callback) {
  if (reduceMotion || !("IntersectionObserver" in window)) return callback();
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); callback(); }
  }, { rootMargin: "0px 0px -15% 0px" });
  io.observe(node);
}

function onWidthChange(node, callback) {
  if (!("ResizeObserver" in window)) return;
  let last = node.clientWidth;
  let frame = 0;
  new ResizeObserver(() => {
    const w = node.clientWidth;
    if (w === last) return;
    last = w;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(callback);
  }).observe(node);
}

function initPaperLink() {
  const link = document.getElementById("paper-link");
  if (!ARXIV_ID) {
    link.addEventListener("click", (event) => event.preventDefault());
    link.title = "The arXiv preprint will be linked here once it is public";
    return;
  }
  link.href = `https://arxiv.org/abs/${ARXIV_ID}`;
  link.removeAttribute("aria-disabled");
  link.querySelector("small").textContent = `arXiv:${ARXIV_ID}`;
}

function initRunhead() {
  const links = [...document.querySelectorAll(".runhead-links a")];
  const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const heads = targets.map((t) => t.querySelector("h2") || t);
  const setActive = () => {
    const y = window.scrollY + window.innerHeight * 0.3;
    let active = -1;
    heads.forEach((h, i) => { if (h.getBoundingClientRect().top + window.scrollY <= y) active = i; });
    links.forEach((a, i) => { if (i === active) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
  };
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { setActive(); ticking = false; });
  }, { passive: true });
  setActive();
}

function initCorpus() {
  const strip = document.getElementById("corpus-strip");
  const readout = document.getElementById("corpus-readout");
  const tbody = document.querySelector("#corpus-table tbody");
  const used = {};
  const rows = CORPUS.map(([name, species, images, ids, group]) => {
    const shades = GROUP_SHADES[group];
    const color = shades[(used[group] = (used[group] ?? -1) + 1) % shades.length];
    return { name, species, images, ids, color };
  });

  const segments = new Map();
  const tableRows = new Map();

  const show = (row) => {
    for (const [name, seg] of segments) seg.classList.toggle("is-hot", row && name === row.name);
    for (const [name, tr] of tableRows) tr.classList.toggle("is-hot", row && name === row.name);
    readout.replaceChildren();
    if (!row) {
      readout.append(el("span", { text: "19 source datasets · 132,533 images · 13,348 identities" }));
      return;
    }
    const share = ((row.images / TOTAL_IMAGES) * 100).toFixed(row.images / TOTAL_IMAGES < 0.01 ? 2 : 1);
    readout.append(
      el("strong", { text: row.name }),
      el("span", { text: row.species }),
      el("span", { class: "tab", text: `${fmt.format(row.images)} images (${share}%)` }),
      el("span", { class: "tab", text: `${fmt.format(row.ids)} identities` }),
    );
  };

  for (const row of rows) {
    const seg = el("button", {
      type: "button", role: "listitem",
      "aria-label": `${row.name}: ${fmt.format(row.images)} images, ${fmt.format(row.ids)} identities`,
      style: { flexGrow: String(row.images), flexBasis: "0", "--c": row.color },
    });
    seg.style.setProperty("--c", row.color);
    seg.addEventListener("mouseenter", () => show(row));
    seg.addEventListener("focus", () => show(row));
    seg.addEventListener("click", () => show(row));
    strip.append(seg);
    segments.set(row.name, seg);
  }
  strip.addEventListener("mouseleave", () => show(null));
  strip.addEventListener("focusout", (event) => { if (!strip.contains(event.relatedTarget)) show(null); });

  const renderTable = (key, dir) => {
    const sorted = [...rows].sort((a, b) => {
      const va = a[key], vb = b[key];
      const cmp = typeof va === "number" ? va - vb : String(va).localeCompare(String(vb));
      return dir === "ascending" ? cmp : -cmp;
    });
    tbody.replaceChildren(...sorted.map((row) => {
      let tr = tableRows.get(row.name);
      if (!tr) {
        tr = el("tr", {}, [
          el("th", { scope: "row" }, [el("span", { class: "dot", style: { background: row.color }, "aria-hidden": "true" }), row.name]),
          el("td", { text: row.species }),
          el("td", { class: "num", text: fmt.format(row.images) }),
          el("td", { class: "num", text: fmt.format(row.ids) }),
        ]);
        tr.addEventListener("mouseenter", () => show(row));
        tr.addEventListener("mouseleave", () => show(null));
        tableRows.set(row.name, tr);
      }
      return tr;
    }));
  };

  const sortButtons = [...document.querySelectorAll("#corpus-table thead button")];
  for (const button of sortButtons) {
    button.addEventListener("click", () => {
      const current = button.getAttribute("aria-sort");
      const numeric = ["images", "ids"].includes(button.dataset.sort);
      const dir = current ? (current === "descending" ? "ascending" : "descending") : (numeric ? "descending" : "ascending");
      for (const b of sortButtons) b.removeAttribute("aria-sort");
      button.setAttribute("aria-sort", dir);
      renderTable(button.dataset.sort, dir);
    });
  }
  renderTable("images", "descending");
  show(null);
}

function bestIndices(rows, offset) {
  const best = [];
  for (let c = 0; c < rows[0].length - offset; c++) {
    const max = Math.max(...rows.map((r) => r[c + offset]));
    best.push(new Set(rows.map((r, i) => (r[c + offset] === max ? i : -1)).filter((i) => i >= 0)));
  }
  return best;
}

function initClosedTable() {
  const table = document.getElementById("closed-table");
  const select = document.getElementById("bar-metric");
  const allValues = CLOSED.flatMap(([, rows]) => rows);
  const families = [];

  for (const [family, rows] of CLOSED) {
    const best = bestIndices(rows, 1);
    const tbody = el("tbody", { class: "family", "data-family": family });
    tbody.append(el("tr", { class: "group" }, [el("th", { scope: "rowgroup", colspan: "7", text: family })]));
    rows.forEach((row, i) => {
      const [loss, ...vals] = row;
      const isApex = loss === "ApexID Hierarchical";
      const isOrig = loss === "Pretrained";
      const tr = el("tr", { class: isApex ? "is-apex" : null }, [el("th", { scope: "row", text: loss })]);
      vals.forEach((v, c) => {
        const td = el("td", { class: "num", "data-col": METRICS[c] });
        td.append(best[c].has(i) ? el("b", { text: v.toFixed(2) }) : document.createTextNode(v.toFixed(2)));
        tr.append(td);
      });
      const bar = el("div", { class: `bar${isOrig ? " is-orig" : ""}` }, [el("span")]);
      bar.style.setProperty("--c", LOSS_COLOR[loss] ?? "var(--orig)");
      tr.append(el("td", { class: "barcell" }, [bar]));
      tr._vals = vals;
      tr._bar = bar;
      tbody.append(tr);
    });
    table.append(tbody);
    families.push(tbody);
  }

  let visible = false;
  const draw = () => {
    const metric = select.value;
    const c = METRICS.indexOf(metric);
    const values = allValues.map((r) => r[c + 1]);
    const lo = Math.floor(Math.min(...values) - 4);
    const hi = Math.max(...values);
    table.querySelectorAll("[data-col]").forEach((cell) => cell.classList.toggle("is-active", cell.dataset.col === metric));
    for (const tr of table.querySelectorAll("tbody tr:not(.group)")) {
      const pct = visible ? ((tr._vals[c] - lo) / (hi - lo)) * 100 : 0;
      tr._bar.style.setProperty("--w", `${pct.toFixed(2)}%`);
      tr._bar.title = `${METRIC_NAME[metric]} ${tr._vals[c].toFixed(2)}%`;
    }
  };
  select.addEventListener("change", draw);
  radioGroup(document.getElementById("bb-filter"), (button) => {
    const bb = button.dataset.bb;
    for (const tbody of families) tbody.hidden = bb !== "all" && tbody.dataset.family !== bb;
  });
  draw();
  whenVisible(table, () => { visible = true; requestAnimationFrame(draw); });
}

function initOpenTable() {
  const table = document.getElementById("open-table");
  for (const [family, rows] of OPEN) {
    const best = bestIndices(rows, 1);
    const tbody = el("tbody", { class: "family" });
    tbody.append(el("tr", { class: "group" }, [el("th", { scope: "rowgroup", colspan: "7", text: family })]));
    rows.forEach((row, i) => {
      const [loss, ...vals] = row;
      const tr = el("tr", { class: loss === "ApexID Hierarchical" ? "is-apex" : null }, [el("th", { scope: "row", text: loss })]);
      vals.forEach((v, c) => {
        const td = el("td", { class: "num" });
        td.append(best[c].has(i) ? el("b", { text: v.toFixed(3) }) : document.createTextNode(v.toFixed(3)));
        tr.append(td);
      });
      tbody.append(tr);
    });
    table.append(tbody);
  }
}

function initAblation() {
  const input = document.getElementById("lambda");
  const out = document.getElementById("lambda-out");
  const metricsBox = document.getElementById("abl-metrics");
  const svg = document.getElementById("abl-plot");
  let metric = 0;

  const buttons = METRICS.map((m, c) => {
    const b = el("button", { type: "button", role: "radio", "aria-checked": String(c === 0) }, [
      el("span", { class: "m-name", text: METRIC_NAME[m] }),
      el("span", { class: "m-val" }),
      el("span", { class: "m-delta" }),
    ]);
    metricsBox.append(b);
    return b;
  });
  radioGroup(metricsBox, (button) => { metric = buttons.indexOf(button); plot(); });

  const H = 220, L = 46, R = 16, T = 16, B = 34;

  const plot = () => {
    const W = Math.max(300, Math.round(svg.parentElement.clientWidth || 640));
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const xs = ABLATION.lambdas.map((_, k) => L + (k * (W - L - R)) / (ABLATION.lambdas.length - 1));
    const i = Number(input.value);
    const series = ABLATION.rows.map((r) => r[metric]);
    const base = ABLATION.base[metric];
    const lo = Math.min(base, ...series), hi = Math.max(base, ...series);
    const pad = (hi - lo) * 0.18 || 0.5;
    const y = (v) => T + (H - T - B) * (1 - (v - (lo - pad)) / (hi + pad - (lo - pad)));
    const ticks = [lo - pad, (lo + hi) / 2, hi + pad].map((v) => Math.round(v * 10) / 10);

    svg.replaceChildren(
      ...ticks.map((t) => el("svg:line", { class: "grid", x1: L, x2: W - R, y1: y(t), y2: y(t) })),
      ...ticks.map((t) => el("svg:text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", text: t.toFixed(1) })),
      el("svg:line", { class: "axis", x1: L, x2: W - R, y1: H - B, y2: H - B }),
      el("svg:line", { class: "base", x1: L, x2: W - R, y1: y(base), y2: y(base) }),
      el("svg:text", { class: "lbl-base", x: W - R, y: y(base) - 6, "text-anchor": "end", text: `SphereFace2 ${base.toFixed(2)}` }),
      el("svg:line", { class: "marker", x1: xs[i], x2: xs[i], y1: T, y2: H - B }),
      el("svg:path", { class: "line", d: series.map((v, k) => `${k ? "L" : "M"}${xs[k].toFixed(1)},${y(v).toFixed(1)}`).join("") }),
      ...series.map((v, k) => el("svg:circle", { class: `pt${k === i ? " is-on" : ""}`, cx: xs[k], cy: y(v), r: k === i ? 5.5 : 4 })),
      ...ABLATION.lambdas.map((l, k) => el("svg:text", { x: xs[k], y: H - B + 20, "text-anchor": "middle", text: l })),
      el("svg:title", { text: `${METRIC_NAME[METRICS[metric]]} across category weights; ${series[i].toFixed(2)} at λ = ${ABLATION.lambdas[i]}` }),
    );
  };

  const update = () => {
    const i = Number(input.value);
    out.value = ABLATION.lambdas[i];
    input.setAttribute("aria-valuetext", ABLATION.lambdas[i]);
    buttons.forEach((b, c) => {
      const v = ABLATION.rows[i][c];
      const d = v - ABLATION.base[c];
      b.querySelector(".m-val").textContent = v.toFixed(2);
      b.querySelector(".m-delta").textContent = `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)} vs. SphereFace2`;
    });
    plot();
  };
  input.addEventListener("input", update);
  update();
  onWidthChange(svg.parentElement, plot);
}

function initTransfer() {
  const svg = document.getElementById("transfer-chart");
  let metric = "r1";
  let visible = false;
  let bars, best, tickGroup, labelW, plotW, axisY;

  const layout = () => {
    const W = Math.max(300, Math.round(svg.parentElement.clientWidth || 820));
    const narrow = W < 560;
    const rowH = narrow ? 24 : 26, famGap = 22, valW = narrow ? 42 : 54, top = 8;
    labelW = narrow ? 150 : 190;
    svg.classList.toggle("is-narrow", narrow);
    plotW = W - labelW - valW;
    const rowsTotal = TRANSFER.reduce((n, [, r]) => n + r.length, 0);
    const H = top + rowsTotal * rowH + (TRANSFER.length - 1) * famGap + 30;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const defs = el("svg:defs", {}, [
      el("svg:pattern", { id: "hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, [
        el("svg:rect", { width: 5, height: 5, fill: "#e3e6ea" }),
        el("svg:line", { x1: 0, y1: 0, x2: 0, y2: 5, stroke: "#9aa1ab", "stroke-width": 1.2 }),
      ]),
    ]);
    svg.replaceChildren(defs);

    bars = [];
    let y = top;
    TRANSFER.forEach(([family, rows], f) => {
      if (f) {
        svg.append(el("svg:line", { class: "sep", x1: 0, x2: W, y1: y + famGap / 2 - 1, y2: y + famGap / 2 - 1 }));
        y += famGap;
      }
      const famY = y;
      rows.forEach(([loss, r1, map]) => {
        const isApex = loss === "ApexID Hierarchical";
        const isOrig = loss === "Original checkpoint";
        svg.append(el("svg:text", { class: `row-lbl${isApex ? " is-apex" : ""}`, x: labelW - 12, y: y + rowH / 2 + 4.5, "text-anchor": "end", text: loss }));
        const rect = el("svg:rect", {
          class: "b", x: labelW, y: y + 5, height: rowH - 10, width: 0,
          fill: isOrig ? "url(#hatch)" : (LOSS_COLOR[loss] ?? "#999"),
          stroke: isOrig ? "#9aa1ab" : "none",
        });
        const val = el("svg:text", { class: `val${isApex ? " is-apex" : ""}`, y: y + rowH / 2 + 4.5 });
        svg.append(rect, val);
        bars.push({ rect, val, r1, map });
        y += rowH;
      });
      svg.append(el("svg:text", {
        class: "fam", x: 0, y: famY + ((y - famY) / 2) + 5, text: family,
        transform: `rotate(-90 ${8} ${famY + (y - famY) / 2})`, "text-anchor": "middle", dx: 0,
      }));
    });
    axisY = y + 4;
    svg.append(el("svg:line", { class: "axis", x1: labelW, x2: labelW + plotW, y1: axisY, y2: axisY }));
    tickGroup = el("svg:g");
    svg.append(tickGroup);
    best = el("svg:line", { class: "best", x1: labelW, x2: labelW, y1: top, y2: axisY });
    defs.after(best);
  };

  const draw = () => {
    const max = metric === "r1" ? 20 : 7.5;
    const step = metric === "r1" ? 5 : 2;
    const sx = (v) => (v / max) * plotW;
    tickGroup.replaceChildren();
    for (let t = 0; t <= max + 1e-9; t += step) {
      tickGroup.append(
        el("svg:line", { class: "sep", x1: labelW + sx(t), x2: labelW + sx(t), y1: axisY, y2: axisY + 5 }),
        el("svg:text", { x: labelW + sx(t), y: axisY + 19, "text-anchor": "middle", text: String(t) }),
      );
    }
    tickGroup.append(el("svg:text", { x: labelW + plotW, y: axisY + 19, dx: 22, "text-anchor": "end", text: "%" }));
    const top1 = Math.max(...bars.map((b) => b[metric]));
    for (const b of bars) {
      const w = visible ? sx(b[metric]) : 0;
      b.rect.setAttribute("width", w.toFixed(1));
      b.val.setAttribute("x", (labelW + sx(b[metric]) + 7).toFixed(1));
      b.val.textContent = b[metric].toFixed(2);
    }
    best.style.transform = `translateX(${visible ? sx(top1) : 0}px)`;
  };
  radioGroup(document.getElementById("transfer-metric"), (button) => { metric = button.dataset.m; draw(); });
  layout();
  draw();
  whenVisible(svg, () => { visible = true; requestAnimationFrame(draw); });
  onWidthChange(svg.parentElement, () => { layout(); draw(); });
}

function initScoreCam() {
  const grid = document.getElementById("cam-grid");
  const backbones = [["convnext", "ConvNeXt-S"], ["miewid", "MIEW-ID v3"], ["megadescriptor", "MegaDescriptor-L"]];
  const losses = [["af", "ArcFace"], ["daf", "Dynamic ArcFace"], ["sf2", "SphereFace2"], ["scsf2", "ApexID Hierarchical"]];
  const dsName = { elephant: "ELPephants", bristol: "Bristol Gorillas", gcn: "GCN-ID" };
  let ds = "elephant";
  const imgs = [];

  grid.append(el("div", { class: "hd", text: "Objective" }), ...backbones.map(([, n]) => el("div", { class: "hd", text: n })));
  for (const [lk, ln] of losses) {
    const isApex = lk === "scsf2";
    grid.append(el("div", { class: `rl${isApex ? " is-apex" : ""}`, text: ln }));
    for (const [bk, bn] of backbones) {
      const img = el("img", { loading: "lazy", decoding: "async", width: 900, height: 206 });
      img._key = [bk, lk, bn, ln];
      imgs.push(img);
      grid.append(el("div", { class: `cam-cell${isApex ? " is-apex" : ""}` }, [img]));
    }
  }
  const panelName = ["overlay", "heatmap", "image"];
  let panel = 0;
  const render = () => {
    for (const img of imgs) {
      const [bk, lk, bn, ln] = img._key;
      img.src = `assets/figures/scorecam/${bk}_${lk}_${ds}.webp`;
      img.alt = `ScoreCAM ${panelName[panel]} for ${bn} trained with ${ln} on a ${dsName[ds]} image`;
    }
    grid.style.setProperty("--p", String(panel));
  };
  radioGroup(document.getElementById("cam-ds"), (b) => { ds = b.dataset.ds; render(); });
  radioGroup(document.getElementById("cam-panel"), (b) => {
    panel = Number(b.dataset.p);
    grid.style.setProperty("--p", String(panel));
    for (const img of imgs) {
      const [, , bn, ln] = img._key;
      img.alt = `ScoreCAM ${panelName[panel]} for ${bn} trained with ${ln} on a ${dsName[ds]} image`;
    }
  });
  render();
}

function initCopy() {
  for (const button of document.querySelectorAll(".copy")) {
    const label = button.querySelector("span");
    const icon = button.querySelector("use");
    button.addEventListener("click", async () => {
      const text = document.getElementById(button.dataset.copy).innerText;
      try {
        await navigator.clipboard.writeText(text);
        button.classList.add("is-done");
        label.textContent = "Copied";
        icon.setAttribute("href", "#i-check");
      } catch {
        label.textContent = "Select and copy";
      }
      setTimeout(() => {
        button.classList.remove("is-done");
        label.textContent = "Copy";
        icon.setAttribute("href", "#i-copy");
      }, 1800);
    });
  }
}

initPaperLink();
initRunhead();
initCorpus();
initClosedTable();
initOpenTable();
initAblation();
initTransfer();
initScoreCam();
initCopy();
