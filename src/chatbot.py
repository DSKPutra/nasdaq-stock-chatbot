"""
chatbot.py
==========
Orchestrator chatbot: menerima teks pertanyaan pengguna, memakai nlu.py
untuk menentukan intent + ticker, lalu memanggil modul data mining yang
sesuai (classification, clustering, anomaly) dan merangkai jawaban
dalam bahasa natural (Bahasa Indonesia).

Tabel ringkasan clustering (500 saham) di-cache ke
data/processed/cluster_summary.csv agar tidak perlu dihitung ulang
setiap kali chatbot dijalankan -- lihat scripts/build_cluster_cache.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import anomaly, classification, clustering, data_loader, nlu

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
    "Saya adalah chatbot analisis saham NASDAQ berbasis data mining. "
    "Anda bisa bertanya seperti:\n\n"
    "- **Tren**: \"Bagaimana tren AAPL 6 bulan terakhir?\"\n"
    "- **Prediksi arah harga**: \"Prediksi NVDA besok naik atau turun?\"\n"
    "- **Saham mirip**: \"Saham apa yang mirip dengan AMD?\"\n"
    "- **Anomali**: \"Kapan TSLA pernah mengalami pergerakan aneh?\"\n\n"
    "Sebutkan kode ticker NASDAQ (mis. AAPL, MSFT, NVDA) di pertanyaan Anda."
)

DISCLAIMER = "\n\n_Catatan: ini hasil model data mining untuk tujuan edukasi, bukan saran investasi._"


def answer(text: str) -> str:
    parsed = nlu.parse(text)

    if parsed.intent == "help" or (parsed.intent == "unknown" and parsed.ticker is None):
        return HELP_TEXT

    if parsed.ticker is None:
        return (
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
        else:
            return _answer_trend(parsed.ticker)  # fallback wajar
    except Exception as e:
        return f"Maaf, terjadi kendala saat memproses permintaan untuk {parsed.ticker}: {e}"


def _answer_trend(ticker: str) -> str:
    df = data_loader.load_ticker(ticker)
    last = df.tail(1).iloc[0]
    prev_30 = df.tail(31).iloc[0] if len(df) > 31 else df.iloc[0]
    change_30d = (last["close"] / prev_30["close"] - 1) * 100

    prev_252 = df.tail(253).iloc[0] if len(df) > 253 else df.iloc[0]
    change_1y = (last["close"] / prev_252["close"] - 1) * 100

    arah = "naik" if change_30d >= 0 else "turun"
    return (
        f"**{ticker}** ditutup di **${last['close']:.2f}** pada {last['date'].date()}.\n\n"
        f"- Perubahan 30 hari terakhir: {change_30d:+.2f}% ({arah})\n"
        f"- Perubahan ~1 tahun terakhir: {change_1y:+.2f}%\n"
        f"- Rentang harga hari terakhir (high-low): ${last['high']:.2f} - ${last['low']:.2f}"
    )


def _answer_predict(ticker: str) -> str:
    result = classification.predict_next_direction(ticker)
    m = result["metrics"]
    return (
        f"Prediksi arah harga **{result['ticker']}** untuk hari berikutnya: "
        f"**{result['direction']}** (probabilitas naik = {result['probability_up']*100:.1f}%).\n\n"
        f"Harga penutupan terakhir: ${result['last_close']:.2f} pada {result['last_date']}.\n\n"
        f"Keandalan model (diuji pada {m['n_test']} hari terakhir, "
        f"{m['test_period_start']} s/d {m['test_period_end']}):\n"
        f"- Akurasi: {m['accuracy']*100:.1f}% (baseline tebak 'selalu naik': {m['baseline_naive_up_rate']*100:.1f}%)\n"
        f"- Precision: {m['precision']*100:.1f}%, Recall: {m['recall']*100:.1f}%, F1: {m['f1']*100:.1f}%\n\n"
        f"Fitur paling berpengaruh: {', '.join(list(result['top_features'].keys())[:3])}."
        + DISCLAIMER
    )


def _answer_similar(ticker: str) -> str:
    cache = _load_cluster_cache()
    sims = clustering.similar_stocks(ticker, cache, top_n=5)
    if sims.empty:
        return f"Tidak ditemukan saham lain yang mirip dengan {ticker} pada cluster yang sama."
    lines = [f"Saham paling mirip dengan **{ticker}** (berdasarkan kelompok volatilitas, return, dan RSI):\n"]
    for t, row in sims.iterrows():
        lines.append(
            f"- **{t}**: return harian rata-rata {row['mean_daily_return']*100:.3f}%, "
            f"volatilitas tahunan {row['volatility_annualized']*100:.1f}%, "
            f"CAGR 3th {row['cagr_3y']*100:.1f}%"
        )
    return "\n".join(lines)


def _answer_cluster(ticker: str) -> str:
    cache = _load_cluster_cache()
    ticker = ticker.upper()
    if ticker not in cache.index:
        return f"Ticker {ticker} tidak ada pada tabel cluster."
    cluster_id = int(cache.loc[ticker, "cluster"])
    profile = clustering.cluster_profile(cache, cluster_id)
    members = cache[cache["cluster"] == cluster_id].index.tolist()
    sample_members = ", ".join(members[:15])
    return (
        f"**{ticker}** berada di **cluster #{cluster_id}** (berisi {len(members)} saham).\n\n"
        f"Profil rata-rata cluster ini:\n"
        f"- Return harian rata-rata: {profile['mean_daily_return']*100:.3f}%\n"
        f"- Volatilitas tahunan: {profile['volatility_annualized']*100:.1f}%\n"
        f"- RSI rata-rata: {profile['avg_rsi']:.1f}\n"
        f"- Max drawdown rata-rata: {profile['max_drawdown']*100:.1f}%\n\n"
        f"Contoh anggota cluster: {sample_members}{'...' if len(members) > 15 else ''}"
    )


def _answer_anomaly(ticker: str) -> str:
    anomalies = anomaly.detect_anomalies(ticker, method="isolation_forest", top_n=5)
    if anomalies.empty:
        return f"Tidak ditemukan hari dengan pergerakan anomali yang signifikan untuk {ticker}."
    lines = [f"Hari-hari dengan pergerakan harga tidak wajar pada **{ticker}** (deteksi Isolation Forest):\n"]
    for _, row in anomalies.iterrows():
        lines.append(
            f"- {row['date'].date()}: return {row['return_1d']*100:+.2f}%, "
            f"close ${row['close']:.2f}, rentang harian {row['hl_range_pct']*100:.2f}%"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(answer("Prediksi AAPL besok naik atau turun?"))
