"""
chatbot.py
==========
Orchestrator chatbot: menerima teks pertanyaan pengguna, memakai nlu.py
untuk menentukan intent + ticker, lalu memanggil modul data mining yang
sesuai (classification, clustering, anomaly) dan merangkai jawaban
dalam bahasa natural (Bahasa Indonesia).

Setiap jawaban tidak cuma melaporkan angka, tapi juga *alasan*-nya
dalam kalimat yang mudah dipahami (lihat src/explain.py) -- mis. bukan
cuma "RSI 72.4", tapi "RSI 72.4 -> zona jenuh beli, rawan koreksi".

Tabel ringkasan clustering (500 saham) di-cache ke
data/processed/cluster_summary.csv agar tidak perlu dihitung ulang
setiap kali chatbot dijalankan -- lihat scripts/build_cluster_cache.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import anomaly, charts, classification, clustering, data_loader, explain, features, nlu

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cluster_summary.csv"

_cluster_cache: pd.DataFrame | None = None


def _load_cluster_cache() -> pd.DataFrame:
    global _cluster_cache
    if _cluster_cache is not None:
        return _cluster_cache
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            "Cache cluster belum dibuat. Jalankan dulu: "
            "python scripts/build_cluster_cache.py"
        )
    _cluster_cache = pd.read_csv(CACHE_PATH, index_col="ticker")
    return _cluster_cache


HELP_TEXT = (
    "Saya adalah **NASDAQku.ID**, chatbot analisis saham NASDAQ berbasis data mining. "
    "Setiap jawaban disertai alasannya, bukan cuma angka. Anda bisa bertanya seperti:\n\n"
    "- **Tren & sinyal teknikal**: \"Bagaimana tren AAPL 30 hari terakhir?\"\n"
    "- **Prediksi arah harga**: \"Prediksi NVDA besok naik atau turun?\"\n"
    "- **Saham mirip**: \"Saham apa yang mirip dengan AMD?\"\n"
    "- **Kelompok/cluster**: \"AAPL masuk cluster apa?\"\n"
    "- **Anomali**: \"Kapan TSLA pernah mengalami pergerakan aneh?\"\n"
    "- **Ringkasan lengkap**: \"Ringkasan MSFT\" (gabungan semua analisis di atas)\n"
    "- **Bandingkan dua saham**: \"Bandingkan AAPL vs MSFT\"\n\n"
    "Sebutkan kode ticker NASDAQ (mis. AAPL, MSFT, NVDA) di pertanyaan Anda."
)

DISCLAIMER = "\n\n_Catatan: ini hasil model data mining untuk tujuan edukasi, bukan saran investasi._"


@dataclass
class Reply:
    """Jawaban chatbot: teks (markdown) + daftar spesifikasi chart
    (lihat src/charts.py). CLI cukup memakai `.text`; Streamlit juga
    merender `.charts`."""

    text: str
    charts: list = field(default_factory=list)


def answer(text: str) -> str:
    """Hanya teksnya (untuk CLI / pemakaian sederhana)."""
    return answer_rich(text).text


def answer_rich(text: str) -> Reply:
    parsed = nlu.parse(text)

    if parsed.intent == "help" or (parsed.intent == "unknown" and parsed.ticker is None):
        return Reply(HELP_TEXT)

    if parsed.intent == "compare":
        if not parsed.ticker or not parsed.ticker2:
            return Reply(
                "Untuk membandingkan, sebutkan dua kode ticker NASDAQ, misalnya "
                "\"Bandingkan AAPL vs MSFT\"."
            )
        try:
            return _answer_compare(parsed.ticker, parsed.ticker2)
        except Exception as e:
            return Reply(f"Maaf, terjadi kendala saat membandingkan {parsed.ticker} dan {parsed.ticker2}: {e}")

    if parsed.ticker is None:
        return Reply(
            "Saya belum menangkap kode ticker-nya. Coba sebutkan kode saham NASDAQ "
            "secara eksplisit, misalnya AAPL, MSFT, atau NVDA."
        )

    try:
        if parsed.intent == "trend":
            return _answer_trend(parsed.ticker)
        elif parsed.intent == "predict":
            return _answer_predict(parsed.ticker)
        elif parsed.intent == "similar":
            return _answer_similar(parsed.ticker)
        elif parsed.intent == "cluster":
            return _answer_cluster(parsed.ticker)
        elif parsed.intent == "anomaly":
            return _answer_anomaly(parsed.ticker)
        elif parsed.intent == "overview":
            return _answer_overview(parsed.ticker)
        else:
            return _answer_trend(parsed.ticker)  # fallback wajar
    except Exception as e:
        return Reply(f"Maaf, terjadi kendala saat memproses permintaan untuk {parsed.ticker}: {e}")


def _answer_trend(ticker: str) -> Reply:
    df = data_loader.load_ticker(ticker)
    df_ind = features.add_technical_indicators(df)
    last = df_ind.tail(1).iloc[0]
    prev_30 = df_ind.tail(31).iloc[0] if len(df_ind) > 31 else df_ind.iloc[0]
    change_30d = (last["close"] / prev_30["close"] - 1) * 100
    prev_252 = df_ind.tail(253).iloc[0] if len(df_ind) > 253 else df_ind.iloc[0]
    change_1y = (last["close"] / prev_252["close"] - 1) * 100

    narasi_30d = explain.interpret_magnitude(change_30d)

    signal_lines = []
    if pd.notna(last.get("rsi_14")):
        signal_lines.append(f"- {explain.interpret_rsi(last['rsi_14'])}")
    if pd.notna(last.get("macd")) and pd.notna(last.get("macd_signal")):
        signal_lines.append(f"- {explain.interpret_macd(last['macd'], last['macd_signal'])}")
    if pd.notna(last.get("sma_10")) and pd.notna(last.get("sma_50")):
        signal_lines.append(f"- {explain.interpret_sma_cross(last['sma_10'], last['sma_50'])}")
    if pd.notna(last.get("volatility_20")):
        signal_lines.append(f"- {explain.interpret_volatility(last['volatility_20'])}")
    signal_block = "\n".join(signal_lines) if signal_lines else "- (indikator belum cukup data historis)"

    text = (
        f"**{ticker}** ditutup di **${last['close']:.2f}** pada {last['date'].date()}.\n\n"
        f"Dalam 30 hari terakhir harganya {narasi_30d}; dalam ~1 tahun terakhir berubah {change_1y:+.2f}%.\n"
        f"Rentang harga hari terakhir (high-low): ${last['high']:.2f} - ${last['low']:.2f}\n\n"
        f"**Sinyal teknikal saat ini:**\n{signal_block}"
    )
    return Reply(text, charts.trend_charts(df_ind, change_30d, change_1y))


def _answer_predict(ticker: str) -> Reply:
    result = classification.predict_next_direction(ticker)
    m = result["metrics"]
    fv = result["feature_values"]

    reasons: list[str] = []
    seen_groups: set[str] = set()
    for name in result["top_features"].keys():
        group = explain.FEATURE_GROUP.get(name, name)
        if group in seen_groups:
            continue
        sentence = explain.interpret_feature(name, fv)
        if sentence:
            reasons.append(f"- {sentence}")
            seen_groups.add(group)
        if len(reasons) >= 3:
            break
    reasoning_block = "\n".join(reasons) if reasons else "- (indikator pendukung tidak tersedia)"

    confidence_note = explain.interpret_confidence_gap(m["accuracy"], m["baseline_naive_up_rate"])

    text = (
        f"Prediksi arah harga **{result['ticker']}** untuk hari berikutnya: "
        f"**{result['direction']}** (probabilitas naik = {result['probability_up']*100:.1f}%).\n\n"
        f"Harga penutupan terakhir: ${result['last_close']:.2f} pada {result['last_date']}.\n\n"
        f"**Mengapa model condong ke arah ini** (indikator paling berpengaruh pada model):\n"
        f"{reasoning_block}\n\n"
        f"**Seberapa bisa diandalkan prediksi ini?**\n"
        f"- Diuji pada {m['n_test']} hari terakhir ({m['test_period_start']} s/d {m['test_period_end']})\n"
        f"- Akurasi {m['accuracy']*100:.1f}% vs baseline 'selalu tebak naik' {m['baseline_naive_up_rate']*100:.1f}% "
        f"-> {confidence_note}\n"
        f"- Precision {m['precision']*100:.1f}% (dari semua prediksi 'naik', berapa % yang benar), "
        f"Recall {m['recall']*100:.1f}% (dari semua hari yang benar-benar naik, berapa % tertangkap model), "
        f"F1 {m['f1']*100:.1f}%"
        + DISCLAIMER
    )
    return Reply(text, charts.prediction_charts(result["direction"], result["probability_up"], m))


def _answer_similar(ticker: str) -> Reply:
    cache = _load_cluster_cache()
    sims = clustering.similar_stocks(ticker, cache, top_n=5)
    if sims.empty:
        return Reply(f"Tidak ditemukan saham lain yang mirip dengan {ticker} pada cluster yang sama.")

    target = cache.loc[ticker.upper()]
    lines = [
        f"Saham paling mirip dengan **{ticker}** -- satu cluster dan jarak fitur "
        "ternormalisasi (return, volatilitas, RSI, drawdown) paling dekat:\n"
    ]
    for t, row in sims.iterrows():
        vol_delta_pp = (row["volatility_annualized"] - target["volatility_annualized"]) * 100
        lebih = "lebih volatil" if vol_delta_pp > 0 else "lebih tenang"
        lines.append(
            f"- **{t}**: return harian rata-rata {row['mean_daily_return']*100:.3f}%, "
            f"volatilitas tahunan {row['volatility_annualized']*100:.1f}% "
            f"({abs(vol_delta_pp):.1f} poin % {lebih} dari {ticker}), CAGR 3th {row['cagr_3y']*100:.1f}%"
        )
    return Reply("\n".join(lines), [charts.similar_chart(ticker.upper(), float(target["volatility_annualized"]), sims)])


def _answer_cluster(ticker: str) -> Reply:
    cache = _load_cluster_cache()
    ticker = ticker.upper()
    if ticker not in cache.index:
        return Reply(f"Ticker {ticker} tidak ada pada tabel cluster.")
    cluster_id = int(cache.loc[ticker, "cluster"])
    profile = clustering.cluster_profile(cache, cluster_id)
    members = cache[cache["cluster"] == cluster_id].index.tolist()
    sample_members = ", ".join(members[:15])

    archetype = explain.interpret_cluster_archetype(
        profile["mean_daily_return"], profile["volatility_annualized"], profile["max_drawdown"]
    )

    row = cache.loc[ticker]
    vol_vs_cluster = "lebih tinggi" if row["volatility_annualized"] > profile["volatility_annualized"] else "lebih rendah"
    ret_vs_cluster = "lebih tinggi" if row["mean_daily_return"] > profile["mean_daily_return"] else "lebih rendah"

    text = (
        f"**{ticker}** berada di **cluster #{cluster_id}** (berisi {len(members)} saham).\n\n"
        f"**Karakter cluster ini:** {archetype}\n\n"
        f"Profil rata-rata cluster:\n"
        f"- Return harian rata-rata: {profile['mean_daily_return']*100:.3f}%\n"
        f"- Volatilitas tahunan: {profile['volatility_annualized']*100:.1f}%\n"
        f"- RSI rata-rata: {profile['avg_rsi']:.1f}\n"
        f"- Max drawdown rata-rata: {profile['max_drawdown']*100:.1f}%\n\n"
        f"Dibanding rata-rata cluster-nya, volatilitas {ticker} **{vol_vs_cluster}** dan "
        f"return harian rata-ratanya **{ret_vs_cluster}**.\n\n"
        f"Contoh anggota cluster: {sample_members}{'...' if len(members) > 15 else ''}"
    )
    return Reply(text, [charts.cluster_chart(cache, cluster_id, ticker)])


def _answer_anomaly(ticker: str) -> Reply:
    df = data_loader.load_ticker(ticker)
    df_ind = features.add_technical_indicators(df)
    typical_daily_pct = float(df_ind["return_1d"].dropna().std() * 100)

    anomalies = anomaly.detect_anomalies(ticker, method="isolation_forest", top_n=5)
    if anomalies.empty:
        return Reply(f"Tidak ditemukan hari dengan pergerakan anomali yang signifikan untuk {ticker}.")

    lines = [
        f"Hari-hari dengan pergerakan harga tidak wajar pada **{ticker}** (deteksi Isolation "
        f"Forest), dibandingkan pergerakan harian normalnya yang sekitar {typical_daily_pct:.2f}%/hari:\n"
    ]
    for _, row in anomalies.iterrows():
        severity = explain.interpret_anomaly_severity(row["return_1d"] * 100, typical_daily_pct)
        lines.append(
            f"- {row['date'].date()}: return {row['return_1d']*100:+.2f}%, "
            f"close ${row['close']:.2f}, rentang harian {row['hl_range_pct']*100:.2f}% -- {severity}"
        )
    return Reply("\n".join(lines), [charts.anomaly_chart(anomalies, ticker)])


def _answer_overview(ticker: str) -> Reply:
    """Ringkasan lengkap: gabungan tren, prediksi, cluster, dan jumlah
    anomali untuk satu saham dalam satu jawaban naratif -- cocok untuk
    pertanyaan terbuka semacam "ceritakan semua tentang AAPL"."""
    ticker = ticker.upper()
    parts: list[Reply] = []

    for label, fn in (
        ("Tren", _answer_trend),
        ("Prediksi arah harga", _answer_predict),
        ("Info cluster", _answer_cluster),
    ):
        try:
            parts.append(fn(ticker))
        except Exception as e:
            parts.append(Reply(f"_{label} tidak tersedia: {e}_"))

    try:
        anomalies = anomaly.detect_anomalies(ticker, method="isolation_forest", top_n=50)
        if anomalies.empty:
            parts.append(Reply(f"Tidak ada anomali besar yang tercatat untuk {ticker}."))
        else:
            parts.append(
                Reply(
                    f"Tercatat **{len(anomalies)} hari** dengan pergerakan anomali signifikan "
                    f"sepanjang riwayat {ticker}. Tanya \"anomali {ticker}\" untuk rinciannya."
                )
            )
    except Exception as e:
        parts.append(Reply(f"_Info anomali tidak tersedia: {e}_"))

    text = f"## Ringkasan lengkap {ticker}\n\n" + "\n\n---\n\n".join(p.text for p in parts)
    return Reply(text, [c for p in parts for c in p.charts])


def _answer_compare(t1: str, t2: str) -> Reply:
    """Membandingkan dua saham: harga & momentum 30 hari, volatilitas,
    dan keanggotaan cluster -- dengan kesimpulan eksplisit mana yang
    lebih kuat/lebih tenang, bukan cuma angka berdampingan."""
    t1, t2 = t1.upper(), t2.upper()
    df1 = data_loader.load_ticker(t1)
    df2 = data_loader.load_ticker(t2)

    def snapshot(df: pd.DataFrame) -> tuple[pd.Series, float, float]:
        last = df.tail(1).iloc[0]
        prev_30 = df.tail(31).iloc[0] if len(df) > 31 else df.iloc[0]
        prev_252 = df.tail(253).iloc[0] if len(df) > 253 else df.iloc[0]
        return last, (last["close"] / prev_30["close"] - 1) * 100, (last["close"] / prev_252["close"] - 1) * 100

    last1, chg1, yr1 = snapshot(df1)
    last2, chg2, yr2 = snapshot(df2)

    lines = [f"**Perbandingan {t1} vs {t2}:**\n"]
    lines.append(f"- Harga terakhir: {t1} ${last1['close']:.2f} vs {t2} ${last2['close']:.2f}")
    lines.append(f"- Perubahan 30 hari terakhir: {t1} {chg1:+.2f}% vs {t2} {chg2:+.2f}%")
    stronger = t1 if chg1 > chg2 else t2
    lines.append(f"  -> momentum 30 hari terakhir lebih kuat pada **{stronger}**")

    vol1 = vol2 = None
    cache = _load_cluster_cache()
    if t1 in cache.index and t2 in cache.index:
        vol1, vol2 = float(cache.loc[t1, "volatility_annualized"]), float(cache.loc[t2, "volatility_annualized"])
        lines.append(f"- Volatilitas tahunan: {t1} {vol1*100:.1f}% vs {t2} {vol2*100:.1f}%")
        calmer = t1 if vol1 < vol2 else t2
        lines.append(f"  -> **{calmer}** secara historis lebih tenang (risiko pergerakan harian lebih kecil)")

        c1, c2 = int(cache.loc[t1, "cluster"]), int(cache.loc[t2, "cluster"])
        if c1 == c2:
            lines.append(f"- Keduanya berada di **cluster #{c1}** yang sama -> karakter pergerakan harga mirip")
        else:
            lines.append(f"- {t1} di cluster #{c1}, {t2} di cluster #{c2} -> karakter pergerakan harga berbeda")
    else:
        missing = t1 if t1 not in cache.index else t2
        lines.append(f"_(Data cluster untuk {missing} tidak tersedia, sebagian perbandingan dilewati.)_")

    return Reply("\n".join(lines) + DISCLAIMER, charts.compare_charts(t1, t2, df1, df2, chg1, chg2, yr1, yr2, vol1, vol2))


if __name__ == "__main__":
    print(answer("Prediksi AAPL besok naik atau turun?"))
    print()
    print(answer("Ringkasan MSFT"))
    print()
    print(answer("Bandingkan AAPL vs MSFT"))
