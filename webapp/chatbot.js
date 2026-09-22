/*
 * chatbot.js
 * ==========
 * Port JavaScript dari src/nlu.py + src/chatbot.py (Python). Logikanya
 * sengaja dibuat sama persis dengan versi Python supaya perilaku chatbot
 * konsisten di kedua platform (Streamlit/CLI vs web statis).
 *
 * Semua ANGKA (prediksi, cluster, anomali) sudah dihitung sebelumnya oleh
 * scripts/build_webapp_data.py (Random Forest, K-Means, Isolation Forest
 * di Python) dan disimpan di data.json. File ini HANYA melakukan intent
 * classification + pembentukan kalimat jawaban -- tidak ada machine
 * learning yang dijalankan di browser.
 */

const INTENT_KEYWORDS = {
  predict: ["prediksi", "predict", "besok naik", "besok turun", "arah harga", "akan naik", "akan turun", "forecast"],
  trend: ["tren", "trend", "bagaimana performa", "grafik", "pergerakan", "sejarah harga", "history"],
  similar: ["mirip", "serupa", "similar", "seperti"],
  cluster: ["kelompok", "cluster", "kategori saham", "segmen"],
  anomaly: ["anomali", "anomaly", "lonjakan", "aneh", "tidak wajar", "ekstrem"],
  help: ["help", "bantuan", "bantu", "bisa apa", "menu", "cara pakai"],
};

const AMBIGUOUS_STOPWORDS = new Set([
  "apa", "act", "aur", "are", "can", "for", "has", "was", "the", "and",
  "all", "arm", "app", "ada", "dan", "itu", "ini", "dari", "pada",
  "untuk", "dengan", "yang", "bisa", "saja", "akan", "atau", "jika",
  "saya", "kamu", "kah", "nya", "jadi", "juga", "saat", "agar", "amp",
]);

let DATA = null; // diisi setelah data.json dimuat

function setData(json) {
  DATA = json;
}

function extractTicker(text) {
  const candidates = text.match(/\b[A-Za-z]{1,5}\b/g) || [];
  const tickerSet = new Set(DATA.tickers);

  // tahap 1: token yang ditulis kapital semua oleh pengguna
  for (const c of candidates) {
    if (c === c.toUpperCase() && c.length >= 2 && tickerSet.has(c.toUpperCase())) {
      return c.toUpperCase();
    }
  }
  // tahap 2: fallback case-insensitive, minus kata umum ambigu
  for (const c of candidates) {
    const cu = c.toUpperCase();
    if (AMBIGUOUS_STOPWORDS.has(c.toLowerCase())) continue;
    if (tickerSet.has(cu)) return cu;
  }
  return null;
}

function classifyIntent(text) {
  const lower = text.toLowerCase();
  for (const [intent, keywords] of Object.entries(INTENT_KEYWORDS)) {
    if (keywords.some((kw) => lower.includes(kw))) return intent;
  }
  if (extractTicker(text)) return "trend";
  return "unknown";
}

const HELP_TEXT =
  "Saya adalah chatbot analisis saham NASDAQ berbasis data mining. Anda bisa bertanya seperti:\n\n" +
  "<b>Tren</b>: \"Bagaimana tren AAPL 30 hari terakhir?\"\n" +
  "<b>Prediksi arah harga</b>: \"Prediksi NVDA besok naik atau turun?\"\n" +
  "<b>Saham mirip</b>: \"Saham apa yang mirip dengan AMD?\"\n" +
  "<b>Cluster</b>: \"AAPL masuk cluster apa?\"\n" +
  "<b>Anomali</b>: \"Kapan TSLA pernah mengalami pergerakan aneh?\"\n\n" +
  "<i>Versi web ini adalah demo dengan " + (DATA ? DATA.n_tickers : 46) + " saham NASDAQ populer (AAPL, MSFT, NVDA, dst). " +
  "Versi lengkap dengan 500 saham tersedia di source code (GitHub) dan versi Streamlit/VPS.</i>";

const DISCLAIMER =
  "\n\n<i>Catatan: ini hasil model data mining untuk tujuan edukasi, bukan saran investasi.</i>";

function fmtPct(x, digits = 2) {
  const sign = x >= 0 ? "+" : "";
  return `${sign}${x.toFixed(digits)}%`;
}

function answerTrend(ticker) {
  const s = DATA.stocks[ticker];
  if (!s) return `Data untuk ${ticker} tidak ditemukan.`;
  const t = s.trend;
  const arah = t.change_30d_pct >= 0 ? "naik" : "turun";
  const arahClass = t.change_30d_pct >= 0 ? "up" : "down";
  return (
    `<b>${ticker}</b> ditutup di <b>$${t.last_close.toFixed(2)}</b> pada ${t.last_date}.\n\n` +
    `- Perubahan 30 hari terakhir: <span class="${arahClass}">${fmtPct(t.change_30d_pct)}</span> (${arah})\n` +
    `- Perubahan ~1 tahun terakhir: ${fmtPct(t.change_1y_pct)}\n` +
    `- Rentang harga hari terakhir (high-low): $${t.last_high.toFixed(2)} - $${t.last_low.toFixed(2)}`
  );
}

function answerPredict(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || !s.prediction) {
    return `Data prediksi untuk ${ticker} tidak tersedia (kemungkinan riwayat harga terlalu pendek).`;
  }
  const p = s.prediction;
  const arahClass = p.direction === "NAIK" ? "up" : "down";
  return (
    `Prediksi arah harga <b>${ticker}</b> untuk hari berikutnya: ` +
    `<span class="${arahClass}"><b>${p.direction}</b></span> (probabilitas naik = ${(p.probability_up * 100).toFixed(1)}%).\n\n` +
    `Harga penutupan terakhir: $${s.trend.last_close.toFixed(2)} pada ${s.trend.last_date}.\n\n` +
    `Keandalan model (diuji pada ${p.n_test} hari terakhir, ${p.test_period_start} s/d ${p.test_period_end}):\n` +
    `- Akurasi: ${(p.accuracy * 100).toFixed(1)}% (baseline tebak 'selalu naik': ${(p.baseline_naive_up_rate * 100).toFixed(1)}%)\n` +
    `- Precision: ${(p.precision * 100).toFixed(1)}%, Recall: ${(p.recall * 100).toFixed(1)}%, F1: ${(p.f1 * 100).toFixed(1)}%\n\n` +
    `Fitur paling berpengaruh: ${p.top_features.join(", ")}.` +
    DISCLAIMER
  );
}

function answerSimilar(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || !s.similar || s.similar.length === 0) {
    return `Tidak ditemukan saham lain yang mirip dengan ${ticker} pada cluster yang sama.`;
  }
  let out = `Saham paling mirip dengan <b>${ticker}</b> (berdasarkan kelompok volatilitas, return, dan RSI):\n\n<ul>`;
  for (const simTicker of s.similar) {
    const simTrend = DATA.stocks[simTicker] ? DATA.stocks[simTicker].trend : null;
    out += `<li><b>${simTicker}</b>` + (simTrend ? ` — close terakhir $${simTrend.last_close.toFixed(2)}` : "") + `</li>`;
  }
  out += "</ul>";
  return out;
}

function answerCluster(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || s.cluster === undefined) return `Ticker ${ticker} tidak ada pada tabel cluster.`;
  const profile = DATA.cluster_profiles[String(s.cluster)];
  const members = profile.sample_members.join(", ");
  return (
    `<b>${ticker}</b> berada di <b>cluster #${s.cluster}</b> (berisi ${profile.n_members} saham).\n\n` +
    `Profil rata-rata cluster ini:\n` +
    `- Return harian rata-rata: ${(profile.mean_daily_return * 100).toFixed(3)}%\n` +
    `- Volatilitas tahunan: ${(profile.volatility_annualized * 100).toFixed(1)}%\n` +
    `- RSI rata-rata: ${profile.avg_rsi.toFixed(1)}\n` +
    `- Max drawdown rata-rata: ${(profile.max_drawdown * 100).toFixed(1)}%\n\n` +
    `Contoh anggota cluster: ${members}${profile.n_members > 15 ? "..." : ""}`
  );
}

function answerAnomaly(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || !s.anomalies || s.anomalies.length === 0) {
    return `Tidak ditemukan hari dengan pergerakan anomali yang signifikan untuk ${ticker}.`;
  }
  let out = `Hari-hari dengan pergerakan harga tidak wajar pada <b>${ticker}</b> (deteksi Isolation Forest):\n\n<ul>`;
  for (const a of s.anomalies) {
    const cls = a.return_pct >= 0 ? "up" : "down";
    out += `<li>${a.date}: <span class="${cls}">${fmtPct(a.return_pct)}</span>, close $${a.close.toFixed(2)}</li>`;
  }
  out += "</ul>";
  return out;
}

function answer(text) {
  const intent = classifyIntent(text);
  const ticker = extractTicker(text);

  if (intent === "help" || (intent === "unknown" && !ticker)) return HELP_TEXT;

  if (!ticker) {
    return "Saya belum menangkap kode ticker-nya. Coba sebutkan kode saham NASDAQ secara eksplisit, misalnya AAPL, MSFT, atau NVDA.";
  }
  if (!DATA.stocks[ticker]) {
    return `Maaf, data untuk ${ticker} tidak tersedia di dataset ini.`;
  }

  switch (intent) {
    case "trend": return answerTrend(ticker);
    case "predict": return answerPredict(ticker);
    case "similar": return answerSimilar(ticker);
    case "cluster": return answerCluster(ticker);
    case "anomaly": return answerAnomaly(ticker);
    default: return answerTrend(ticker);
  }
}
