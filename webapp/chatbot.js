/*
 * chatbot.js
 * ==========
 * Port JavaScript dari src/nlu.py + src/chatbot.py + src/explain.py
 * (Python). Logikanya sengaja dibuat semirip mungkin dengan versi
 * Python supaya perilaku chatbot konsisten di kedua platform
 * (Streamlit/CLI vs web statis).
 *
 * Semua ANGKA MENTAH (prediksi, cluster, anomali) sudah dihitung
 * sebelumnya oleh scripts/build_webapp_data.py (Random Forest, K-Means,
 * Isolation Forest di Python) dan disimpan di data.json. File ini
 * HANYA melakukan intent classification + penyusunan kalimat jawaban
 * (termasuk alasan/reasoning-nya) -- tidak ada machine learning yang
 * dijalankan di browser.
 *
 * CATATAN PENTING soal paritas dengan Python: data.json sengaja dibuat
 * ringkas (tidak menyimpan semua nilai indikator mentah per saham,
 * mis. RSI/MACD/SMA "live" per hari), supaya ukuran filenya tetap
 * kecil untuk di-fetch browser. Akibatnya beberapa narasi di sini
 * SEDIKIT lebih umum dibanding versi Python (src/chatbot.py), yang
 * menghitung ulang indikator dari CSV mentah setiap permintaan. Versi
 * lengkap (500 saham + indikator penuh) ada di source code (GitHub)
 * dan versi Streamlit/VPS.
 */

const INTENT_KEYWORDS = {
  compare: ["bandingkan", "dibanding", "dibandingkan", "perbandingan", " vs ", " vs. ", " versus "],
  overview: ["ringkasan", "rangkuman", "overview", "profil lengkap", "laporan lengkap", "ceritakan semua", "selengkapnya"],
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

// Mencari hingga maxN kode ticker valid di dalam teks, dalam urutan
// kemunculan -- dipakai untuk intent "compare" (butuh 2 ticker) dan
// fallback untuk extractTicker (butuh 1 ticker). Strategi dua tahap:
// (1) token yang ditulis KAPITAL SEMUA oleh user diprioritaskan,
// (2) baru fallback case-insensitive minus kata umum yang ambigu.
function extractTickers(text, maxN = 1) {
  const candidates = text.match(/\b[A-Za-z]{1,5}\b/g) || [];
  const tickerSet = new Set(DATA.tickers);
  const found = [];

  for (const c of candidates) {
    if (found.length >= maxN) return found;
    if (c === c.toUpperCase() && c.length >= 2 && tickerSet.has(c.toUpperCase()) && !found.includes(c.toUpperCase())) {
      found.push(c.toUpperCase());
    }
  }
  for (const c of candidates) {
    if (found.length >= maxN) return found;
    const cu = c.toUpperCase();
    if (AMBIGUOUS_STOPWORDS.has(c.toLowerCase())) continue;
    if (tickerSet.has(cu) && !found.includes(cu)) found.push(cu);
  }
  return found;
}

function extractTicker(text) {
  const found = extractTickers(text, 1);
  return found.length ? found[0] : null;
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
  "Saya adalah <b>NASDAQku.ID</b>, chatbot analisis saham NASDAQ berbasis data mining. Setiap jawaban disertai alasannya, bukan cuma angka. Anda bisa bertanya seperti:\n\n" +
  "<b>Tren & sinyal</b>: \"Bagaimana tren AAPL 30 hari terakhir?\"\n" +
  "<b>Prediksi arah harga</b>: \"Prediksi NVDA besok naik atau turun?\"\n" +
  "<b>Saham mirip</b>: \"Saham apa yang mirip dengan AMD?\"\n" +
  "<b>Cluster</b>: \"AAPL masuk cluster apa?\"\n" +
  "<b>Anomali</b>: \"Kapan TSLA pernah mengalami pergerakan aneh?\"\n" +
  "<b>Ringkasan lengkap</b>: \"Ringkasan MSFT\"\n" +
  "<b>Bandingkan dua saham</b>: \"Bandingkan AAPL vs MSFT\"\n\n" +
  "<i>Versi web ini adalah demo dengan " + (DATA ? DATA.n_tickers : 46) + " saham NASDAQ populer (AAPL, MSFT, NVDA, dst). " +
  "Versi lengkap dengan 500 saham dan sinyal teknikal penuh (RSI/MACD/SMA harian) tersedia di source code (GitHub) dan versi Streamlit/VPS.</i>";

const DISCLAIMER =
  "\n\n<i>Catatan: ini hasil model data mining untuk tujuan edukasi, bukan saran investasi.</i>";

function fmtPct(x, digits = 2) {
  const sign = x >= 0 ? "+" : "";
  return `${sign}${x.toFixed(digits)}%`;
}

// ---------------------------------------------------------------
// Lapisan "penerjemah" angka -> kalimat mudah dipahami. Lihat
// src/explain.py untuk versi Python-nya (lebih lengkap karena punya
// akses nilai indikator mentah per saham).
// ---------------------------------------------------------------

function interpretMagnitude(pct) {
  const a = Math.abs(pct);
  const arah = pct >= 0 ? "naik" : "turun";
  if (a < 1) return `nyaris stabil (${fmtPct(pct)})`;
  if (a < 5) return `${arah} tipis (${fmtPct(pct)})`;
  if (a < 15) return `${arah} cukup signifikan (${fmtPct(pct)})`;
  return `${arah} tajam (${fmtPct(pct)})`;
}

function interpretVolatility(volAnnualized) {
  const pct = volAnnualized * 100;
  if (pct < 25) return `volatilitas tahunan ${pct.toFixed(1)}% tergolong rendah -> pergerakan harga relatif tenang`;
  if (pct < 50) return `volatilitas tahunan ${pct.toFixed(1)}% tergolong sedang`;
  return `volatilitas tahunan ${pct.toFixed(1)}% tergolong tinggi -> pergerakan harga cenderung liar, risiko (dan potensi untung-rugi harian) lebih besar`;
}

function interpretConfidenceGap(accuracy, baseline) {
  const gapPp = (accuracy - baseline) * 100;
  if (gapPp < 0) {
    return `akurasi model justru ${Math.abs(gapPp).toFixed(1)} poin persentase di bawah tebakan naif ('selalu tebak naik') -> untuk saham ini, menebak arah secara acak sama saja atau malah lebih baik dari modelnya. Jangan jadikan ini sinyal utama`;
  }
  if (gapPp <= 2) {
    return `model ini hanya unggul ${gapPp.toFixed(1)} poin persentase dari tebakan naif ('selalu tebak naik') -> secara statistik nyaris tidak lebih baik dari menebak acak, konsisten dengan teori pasar efisien (Efficient Market Hypothesis). Jangan jadikan ini sinyal utama`;
  }
  if (gapPp <= 5) {
    return `model ini unggul ${gapPp.toFixed(1)} poin persentase dari tebakan naif -> sedikit lebih baik dari acak, tapi masih lemah untuk dijadikan sinyal tunggal`;
  }
  return `model ini unggul ${gapPp.toFixed(1)} poin persentase dari tebakan naif -> cukup meyakinkan dibanding baseline, meski tetap bukan jaminan (lihat juga precision/recall)`;
}

function interpretClusterArchetype(meanDailyReturn, volatilityAnnualized, maxDrawdown) {
  const CLUSTER_RETURN_HIGH = 0.0008;
  const CLUSTER_VOL_HIGH = 0.5;
  const CLUSTER_VOL_LOW = 0.25;
  const CLUSTER_DRAWDOWN_DEEP = -0.4;

  const highVol = volatilityAnnualized >= CLUSTER_VOL_HIGH;
  const lowVol = volatilityAnnualized <= CLUSTER_VOL_LOW;
  const highReturn = meanDailyReturn >= CLUSTER_RETURN_HIGH;

  let label, desc;
  if (highVol && highReturn) {
    label = "Growth volatil";
    desc = "return harian rata-rata tinggi tapi diiringi ayunan harga besar -- berpotensi cuan besar, risiko juga besar";
  } else if (highVol && !highReturn) {
    label = "Spekulatif/berisiko";
    desc = "ayunan harga besar tanpa return rata-rata yang sepadan -- cenderung lebih berisiko daripada menguntungkan";
  } else if (lowVol && highReturn) {
    label = "Growth stabil";
    desc = "return harian rata-rata positif dengan pergerakan harga relatif tenang -- kombinasi yang relatif jarang & menarik";
  } else if (lowVol && !highReturn) {
    label = "Defensif/stabil";
    desc = "pergerakan harga tenang, return rata-rata moderat -- cocok untuk profil yang menghindari gejolak";
  } else {
    label = "Menengah/seimbang";
    desc = "volatilitas dan return berada di kisaran menengah, tidak condong ke salah satu ekstrem";
  }

  if (maxDrawdown <= CLUSTER_DRAWDOWN_DEEP) {
    desc += `; pernah mengalami drawdown dalam (maks ${(maxDrawdown * 100).toFixed(1)}%), menandakan ada periode penurunan tajam di masa lalu`;
  }
  return `${label} -- ${desc}`;
}

function interpretAnomalySeverity(returnPct, typicalMovePct) {
  const a = Math.abs(returnPct);
  if (typicalMovePct && typicalMovePct > 0) {
    const multiple = a / typicalMovePct;
    const tag = multiple >= 4 ? "ekstrem" : multiple >= 2.5 ? "signifikan" : "cukup besar";
    return `${tag} -- sekitar ${multiple.toFixed(1)}x lebih besar dari perkiraan pergerakan harian normal kelompok saham ini`;
  }
  const tag = a >= 15 ? "ekstrem" : a >= 8 ? "signifikan" : "cukup besar";
  return `${tag} (${fmtPct(returnPct)} dalam sehari)`;
}

// Deskripsi generik per nama fitur (tanpa nilai mentah, karena
// data.json tidak menyimpan nilai indikator live per saham -- lihat
// catatan paritas di atas). Tetap menjelaskan APA fitur itu dan
// kenapa relevan, bukan cuma menyebut nama kolomnya.
const FEATURE_GROUP = {
  rsi_14: "rsi", macd: "macd", macd_signal: "macd",
  sma_10: "sma", sma_50: "sma", ema_12: "ema", ema_26: "ema",
  volatility_20: "volatility", return_1d: "return_1d", hl_range_pct: "hl_range",
};

const FEATURE_EXPLANATION = {
  rsi_14: "RSI 14 hari (mengukur apakah saham sedang jenuh beli/jenuh jual) menjadi salah satu sinyal utama",
  macd: "MACD (selisih rata-rata bergerak jangka pendek vs panjang, penanda momentum tren) ikut mempengaruhi arah prediksi",
  macd_signal: "garis sinyal MACD (pemicu crossover tren) ikut mempengaruhi arah prediksi",
  sma_10: "posisi harga relatif terhadap rata-rata bergerak 10 hari ikut jadi pertimbangan",
  sma_50: "posisi harga relatif terhadap rata-rata bergerak 50 hari ikut jadi pertimbangan",
  ema_12: "rata-rata bergerak eksponensial jangka pendek (EMA12) ikut jadi pertimbangan",
  ema_26: "rata-rata bergerak eksponensial jangka panjang (EMA26) ikut jadi pertimbangan",
  volatility_20: "volatilitas 20 hari terakhir (seberapa liar pergerakan harga) ikut jadi pertimbangan",
  return_1d: "return harian terakhir (momentum sehari sebelumnya) ikut jadi pertimbangan",
  hl_range_pct: "rentang pergerakan high-low harian ikut jadi pertimbangan",
};

function interpretFeatureName(name) {
  return FEATURE_EXPLANATION[name] || null;
}

function answerTrend(ticker) {
  const s = DATA.stocks[ticker];
  if (!s) return `Data untuk ${ticker} tidak ditemukan.`;
  const t = s.trend;
  const narasi30d = interpretMagnitude(t.change_30d_pct);
  return (
    `<b>${ticker}</b> ditutup di <b>$${t.last_close.toFixed(2)}</b> pada ${t.last_date}.\n\n` +
    `Dalam 30 hari terakhir harganya ${narasi30d}; dalam ~1 tahun terakhir berubah ${fmtPct(t.change_1y_pct)}.\n` +
    `Rentang harga hari terakhir (high-low): $${t.last_high.toFixed(2)} - $${t.last_low.toFixed(2)}\n\n` +
    `<i>Sinyal teknikal harian (RSI/MACD/SMA) hanya tersedia di versi lengkap (Python/Streamlit) -- lihat juga cluster &amp; prediksi di bawah untuk gambaran arah momentum.</i>`
  );
}

function answerPredict(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || !s.prediction) {
    return `Data prediksi untuk ${ticker} tidak tersedia (kemungkinan riwayat harga terlalu pendek).`;
  }
  const p = s.prediction;
  const arahClass = p.direction === "NAIK" ? "up" : "down";

  const reasons = [];
  const seenGroups = new Set();
  for (const name of p.top_features) {
    const group = FEATURE_GROUP[name] || name;
    if (seenGroups.has(group)) continue;
    const sentence = interpretFeatureName(name);
    if (sentence) {
      reasons.push(`<li>${sentence}</li>`);
      seenGroups.add(group);
    }
    if (reasons.length >= 3) break;
  }
  const reasoningBlock = reasons.length ? `<ul>${reasons.join("")}</ul>` : "(indikator pendukung tidak tersedia)";
  const confidenceNote = interpretConfidenceGap(p.accuracy, p.baseline_naive_up_rate);

  return (
    `Prediksi arah harga <b>${ticker}</b> untuk hari berikutnya: ` +
    `<span class="${arahClass}"><b>${p.direction}</b></span> (probabilitas naik = ${(p.probability_up * 100).toFixed(1)}%).\n\n` +
    `Harga penutupan terakhir: $${s.trend.last_close.toFixed(2)} pada ${s.trend.last_date}.\n\n` +
    `<b>Mengapa model condong ke arah ini</b> (indikator paling berpengaruh pada model):\n${reasoningBlock}\n` +
    `<b>Seberapa bisa diandalkan prediksi ini?</b>\n` +
    `- Diuji pada ${p.n_test} hari terakhir (${p.test_period_start} s/d ${p.test_period_end})\n` +
    `- Akurasi ${(p.accuracy * 100).toFixed(1)}% vs baseline 'selalu tebak naik' ${(p.baseline_naive_up_rate * 100).toFixed(1)}% -> ${confidenceNote}\n` +
    `- Precision ${(p.precision * 100).toFixed(1)}% (dari semua prediksi 'naik', berapa % yang benar), Recall ${(p.recall * 100).toFixed(1)}% (dari semua hari yang benar-benar naik, berapa % tertangkap model), F1 ${(p.f1 * 100).toFixed(1)}%` +
    DISCLAIMER
  );
}

function answerSimilar(ticker) {
  const s = DATA.stocks[ticker];
  if (!s || !s.similar || s.similar.length === 0) {
    return `Tidak ditemukan saham lain yang mirip dengan ${ticker} pada cluster yang sama.`;
  }
  const profile = s.cluster !== undefined ? DATA.cluster_profiles[String(s.cluster)] : null;
  let out = `Saham paling mirip dengan <b>${ticker}</b> -- satu kelompok (cluster) karakter pergerakan harga` +
    (profile ? `: <i>${interpretClusterArchetype(profile.mean_daily_return, profile.volatility_annualized, profile.max_drawdown)}</i>` : "") +
    `.\n\n<ul>`;
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
  const archetype = interpretClusterArchetype(profile.mean_daily_return, profile.volatility_annualized, profile.max_drawdown);
  return (
    `<b>${ticker}</b> berada di <b>cluster #${s.cluster}</b> (berisi ${profile.n_members} saham).\n\n` +
    `<b>Karakter cluster ini:</b> ${archetype}\n\n` +
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
  // Perkiraan pergerakan harian normal dari volatilitas tahunan cluster-nya
  // (bukan volatilitas harian spesifik saham ini -- data.json tidak
  // menyimpan itu per saham; lihat catatan paritas di atas).
  const profile = s.cluster !== undefined ? DATA.cluster_profiles[String(s.cluster)] : null;
  const typicalDailyPct = profile ? (profile.volatility_annualized / Math.sqrt(252)) * 100 : null;

  let out = `Hari-hari dengan pergerakan harga tidak wajar pada <b>${ticker}</b> (deteksi Isolation Forest)` +
    (typicalDailyPct ? `, dibandingkan perkiraan pergerakan harian normal kelompoknya (~${typicalDailyPct.toFixed(2)}%/hari)` : "") +
    `:\n\n<ul>`;
  for (const a of s.anomalies) {
    const cls = a.return_pct >= 0 ? "up" : "down";
    const severity = interpretAnomalySeverity(a.return_pct, typicalDailyPct);
    out += `<li>${a.date}: <span class="${cls}">${fmtPct(a.return_pct)}</span>, close $${a.close.toFixed(2)} -- ${severity}</li>`;
  }
  out += "</ul>";
  return out;
}

function answerOverview(ticker) {
  const s = DATA.stocks[ticker];
  if (!s) return `Data untuk ${ticker} tidak ditemukan.`;
  const nAnomalies = s.anomalies ? s.anomalies.length : 0;
  return (
    `<b>Ringkasan lengkap ${ticker}</b>\n\n` +
    answerTrend(ticker) +
    `\n\n<hr>\n\n` +
    answerPredict(ticker) +
    `\n\n<hr>\n\n` +
    answerCluster(ticker) +
    `\n\n<hr>\n\n` +
    (nAnomalies
      ? `Tercatat <b>${nAnomalies} hari</b> dengan pergerakan anomali signifikan sepanjang riwayat ${ticker}. Tanya "anomali ${ticker}" untuk rinciannya.`
      : `Tidak ada anomali besar yang tercatat untuk ${ticker}.`)
  );
}

function answerCompare(t1, t2) {
  const s1 = DATA.stocks[t1];
  const s2 = DATA.stocks[t2];
  if (!s1 || !s2) {
    const missing = !s1 ? t1 : t2;
    return `Maaf, data untuk ${missing} tidak tersedia di dataset demo ini.`;
  }

  const lines = [`<b>Perbandingan ${t1} vs ${t2}:</b>`, "<ul>"];
  lines.push(`<li>Harga terakhir: ${t1} $${s1.trend.last_close.toFixed(2)} vs ${t2} $${s2.trend.last_close.toFixed(2)}</li>`);
  lines.push(`<li>Perubahan 30 hari terakhir: ${t1} ${fmtPct(s1.trend.change_30d_pct)} vs ${t2} ${fmtPct(s2.trend.change_30d_pct)}`);
  const stronger = s1.trend.change_30d_pct > s2.trend.change_30d_pct ? t1 : t2;
  lines.push(` -- momentum 30 hari terakhir lebih kuat pada <b>${stronger}</b></li>`);

  const p1 = s1.cluster !== undefined ? DATA.cluster_profiles[String(s1.cluster)] : null;
  const p2 = s2.cluster !== undefined ? DATA.cluster_profiles[String(s2.cluster)] : null;
  if (p1 && p2) {
    lines.push(`<li>Volatilitas tahunan kelompoknya: ${t1} ~${(p1.volatility_annualized * 100).toFixed(1)}% vs ${t2} ~${(p2.volatility_annualized * 100).toFixed(1)}%`);
    const calmer = p1.volatility_annualized < p2.volatility_annualized ? t1 : t2;
    lines.push(` -- <b>${calmer}</b> berada di kelompok yang secara historis lebih tenang</li>`);
  }
  if (s1.cluster !== undefined && s2.cluster !== undefined) {
    if (s1.cluster === s2.cluster) {
      lines.push(`<li>Keduanya berada di <b>cluster #${s1.cluster}</b> yang sama -- karakter pergerakan harga mirip</li>`);
    } else {
      lines.push(`<li>${t1} di cluster #${s1.cluster}, ${t2} di cluster #${s2.cluster} -- karakter pergerakan harga berbeda</li>`);
    }
  }
  lines.push("</ul>");
  return lines.join("\n") + DISCLAIMER;
}

function answer(text) {
  const intent = classifyIntent(text);

  if (intent === "help") return HELP_TEXT;

  if (intent === "compare") {
    const tks = extractTickers(text, 2);
    if (tks.length < 2) {
      return "Untuk membandingkan, sebutkan dua kode ticker NASDAQ, misalnya \"Bandingkan AAPL vs MSFT\".";
    }
    return answerCompare(tks[0], tks[1]);
  }

  const ticker = extractTicker(text);
  if (intent === "unknown" && !ticker) return HELP_TEXT;

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
    case "overview": return answerOverview(ticker);
    default: return answerTrend(ticker);
  }
}
