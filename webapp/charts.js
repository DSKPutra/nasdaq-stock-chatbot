/*
 * charts.js
 * =========
 * Renderer chart untuk jawaban chatbot. Menerima "spesifikasi" chart
 * (format yang sama dengan src/charts.py di versi Python) dan
 * menghasilkan HTML+CSS -- bukan SVG/canvas -- supaya teks tetap
 * terbaca di layar HP dan tidak butuh library eksternal.
 *
 * Jenis: "bars" (batang horizontal, bertanda +/- atau tidak),
 * "grouped" (dua seri per baris), "prob" (gauge peluang naik/turun).
 * Grafik garis harga hanya ada di versi Python/Streamlit karena
 * data.json tidak menyimpan deret harga.
 */

function chartEsc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function chartFmt(v, spec) {
  const unit = spec.unit || "";
  return spec.signed ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}${unit}` : `${v.toFixed(1)}${unit}`;
}

// Menentukan rentang sumbu: simetris di sekitar 0 untuk data bertanda,
// [0, maks] (atau `domain`) untuk lainnya. `cap` memotong batang
// ekstrem supaya batang lain tetap terbaca (diberi tanda potong).
function chartScale(spec, values) {
  let m = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  if (spec.cap) m = Math.min(m, spec.cap);
  if (spec.signed) return { lo: -m, hi: m };
  if (spec.domain) return { lo: spec.domain[0], hi: spec.domain[1] };
  return { lo: 0, hi: m };
}

function chartBarFill(v, tone, sc, signed) {
  const span = sc.hi - sc.lo;
  const clipped = Math.max(Math.min(v, sc.hi), sc.lo);
  const cut = clipped !== v;
  let left, width;
  if (signed) {
    const w = (Math.abs(clipped) / sc.hi) * 50;
    left = clipped >= 0 ? 50 : 50 - w;
    width = w;
  } else {
    left = 0;
    width = ((clipped - sc.lo) / span) * 100;
  }
  return `<span class="bar-fill tone-${tone}${cut ? " cut" : ""}" style="left:${left.toFixed(2)}%;width:${Math.max(width, 0.6).toFixed(2)}%"></span>`;
}

function chartGuide(spec, sc) {
  if (!spec.guide) return "";
  const left = ((spec.guide.value - sc.lo) / (sc.hi - sc.lo)) * 100;
  return `<span class="bar-guide" style="left:${left.toFixed(2)}%"></span>`;
}

function chartLabelWidth(labels) {
  const n = Math.max(...labels.map((l) => String(l).length));
  return Math.min(Math.max(n * 7 + 6, 64), 150);
}

function chartHead(spec) {
  return `<div class="chart-title">${chartEsc(spec.title)}</div>`;
}

function chartFoot(spec, wasCut, legend) {
  let note = spec.caption || "";
  if (wasCut) note += ` Batang yang ditandai // dipotong di ±${spec.cap}${spec.unit || ""} agar batang lain tetap terbaca; angka di kanan adalah nilai sebenarnya.`;
  const guide = spec.guide ? `<span class="key"><i class="k-guide"></i>${chartEsc(spec.guide.label)} ${spec.guide.value}${spec.unit || ""}</span>` : "";
  return `${legend || ""}${guide}<p class="chart-caption">${chartEsc(note)}</p>`;
}

function chartAria(spec) {
  const rows = (spec.items || []).map((it) => `${it.label}: ${it.values ? it.values.join(" / ") : it.value}`).join("; ");
  return chartEsc(`${spec.title}. ${rows}`);
}

function renderBars(spec) {
  const values = spec.items.map((it) => it.value);
  const sc = chartScale(spec, values);
  const lw = chartLabelWidth(spec.items.map((it) => it.label));
  const rows = spec.items
    .map(
      (it) =>
        `<div class="bar-row"><span class="bar-label">${chartEsc(it.label)}</span>` +
        `<span class="bar-track${spec.signed ? " signed" : ""}">${chartBarFill(it.value, it.tone, sc, spec.signed)}${chartGuide(spec, sc)}</span>` +
        `<span class="bar-value">${chartFmt(it.value, spec)}</span></div>`
    )
    .join("");
  const wasCut = values.some((v) => v > sc.hi || v < sc.lo);
  return `<figure class="chart" role="img" aria-label="${chartAria(spec)}">${chartHead(spec)}<div class="bars" style="--lw:${lw}px">${rows}</div>${chartFoot(spec, wasCut)}</figure>`;
}

function renderGrouped(spec) {
  const values = spec.items.flatMap((it) => it.values);
  const sc = chartScale(spec, values);
  const lw = chartLabelWidth(spec.items.map((it) => it.label));
  const rows = spec.items
    .map((it) => {
      const bars = it.values.map((v, j) => chartBarFill(v, spec.series[j].tone, sc, spec.signed).replace('class="bar-fill', `data-s="${j}" class="bar-fill`)).join("");
      const vals = it.values.map((v) => `<span>${chartFmt(v, spec)}</span>`).join("");
      return (
        `<div class="bar-row grouped"><span class="bar-label">${chartEsc(it.label)}</span>` +
        `<span class="bar-track g${it.values.length}${spec.signed ? " signed" : ""}">${bars}</span>` +
        `<span class="bar-value multi">${vals}</span></div>`
      );
    })
    .join("");
  const legend = `<div class="legend">${spec.series.map((s) => `<span class="key"><i class="k-fill tone-${s.tone}"></i>${chartEsc(s.label)}</span>`).join("")}</div>`;
  const wasCut = values.some((v) => v > sc.hi || v < sc.lo);
  return `<figure class="chart" role="img" aria-label="${chartAria(spec)}">${chartHead(spec)}${legend}<div class="bars" style="--lw:${lw}px">${rows}</div>${chartFoot(spec, wasCut)}</figure>`;
}

function renderProb(spec) {
  const up = (spec.p_up * 100).toFixed(1);
  const down = ((1 - spec.p_up) * 100).toFixed(1);
  return (
    `<figure class="chart" role="img" aria-label="${chartEsc(`${spec.title}. Naik ${up}%, turun ${down}%`)}">${chartHead(spec)}` +
    `<div class="prob"><span class="prob-up" style="width:${up}%"></span><span class="prob-down" style="width:${down}%"></span></div>` +
    `<div class="prob-labels"><span class="up">NAIK ${up}%</span><span class="down">TURUN ${down}%</span></div>` +
    `<p class="chart-caption">${chartEsc(spec.caption || "")}</p></figure>`
  );
}

function renderChart(spec) {
  if (spec.kind === "bars") return renderBars(spec);
  if (spec.kind === "grouped") return renderGrouped(spec);
  if (spec.kind === "prob") return renderProb(spec);
  return "";
}

function renderCharts(specs) {
  return specs.map(renderChart).join("");
}
