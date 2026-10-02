"""
explain.py
==========
Lapisan "penerjemah" angka teknikal/statistik menjadi kalimat Bahasa
Indonesia yang mudah dipahami orang awam (bukan cuma angka mentah).

Dipisah dari modul data mining (features/classification/clustering/
anomaly) supaya logika model dan logika bahasa tidak bercampur --
mempermudah audit laporan UTS (mana bagian "model", mana bagian
"narasi"). Semua fungsi di sini murni (tanpa efek samping, tanpa akses
data/IO), sehingga mudah diuji unit.

Logikanya diport manual ke webapp/chatbot.js dengan penyesuaian: versi
web statis tidak menyimpan semua nilai indikator mentah per saham di
data.json (supaya filenya tetap ringan), jadi sebagian narasi di sana
lebih umum dibanding versi Python ini. Lihat komentar di chatbot.js.
"""

from __future__ import annotations

# Nama fitur teknikal yang "satu kelompok" (supaya saat menjelaskan
# fitur-fitur paling berpengaruh dari model, kita tidak menyebut dua
# kali hal yang sama, mis. "macd" dan "macd_signal" adalah satu cerita).
FEATURE_GROUP = {
    "rsi_14": "rsi",
    "macd": "macd",
    "macd_signal": "macd",
    "sma_10": "sma",
    "sma_50": "sma",
    "ema_12": "ema",
    "ema_26": "ema",
    "volatility_20": "volatility",
    "return_1d": "return_1d",
    "hl_range_pct": "hl_range",
}

FEATURE_LABELS = {
    "rsi_14": "RSI 14 hari (momentum beli/jual)",
    "macd": "MACD (momentum tren jangka pendek)",
    "macd_signal": "garis sinyal MACD",
    "sma_10": "rata-rata bergerak 10 hari (SMA10)",
    "sma_50": "rata-rata bergerak 50 hari (SMA50)",
    "ema_12": "rata-rata bergerak eksponensial 12 hari (EMA12)",
    "ema_26": "rata-rata bergerak eksponensial 26 hari (EMA26)",
    "volatility_20": "volatilitas 20 hari terakhir",
    "return_1d": "return harian terakhir",
    "hl_range_pct": "rentang high-low harian",
}


def interpret_rsi(rsi: float) -> str:
    if rsi >= 70:
        return f"RSI {rsi:.1f} -> zona jenuh beli (overbought); historisnya rawan koreksi turun"
    if rsi <= 30:
        return f"RSI {rsi:.1f} -> zona jenuh jual (oversold); historisnya rawan rebound naik"
    if rsi >= 55:
        return f"RSI {rsi:.1f} -> momentum beli masih dominan, belum ekstrem"
    if rsi <= 45:
        return f"RSI {rsi:.1f} -> momentum jual masih dominan, belum ekstrem"
    return f"RSI {rsi:.1f} -> netral, tidak condong ke momentum beli atau jual"


def interpret_macd(macd: float, signal: float) -> str:
    gap = macd - signal
    if gap > 0:
        return f"MACD di atas garis sinyal (selisih +{gap:.3f}) -> momentum jangka pendek condong naik (bullish)"
    if gap < 0:
        return f"MACD di bawah garis sinyal (selisih {gap:.3f}) -> momentum jangka pendek condong turun (bearish)"
    return "MACD dan garis sinyal nyaris berimpit -> momentum jangka pendek belum menentu"


def interpret_sma_cross(sma_short: float, sma_long: float, short_label="SMA10", long_label="SMA50") -> str:
    if sma_long == 0:
        return "data rata-rata bergerak belum cukup untuk dibandingkan"
    if sma_short > sma_long:
        gap_pct = (sma_short / sma_long - 1) * 100
        return f"{short_label} {gap_pct:.1f}% di atas {long_label} -> tren jangka menengah masih naik"
    gap_pct = (sma_long / sma_short - 1) * 100
    return f"{short_label} {gap_pct:.1f}% di bawah {long_label} -> tren jangka menengah masih turun"


def interpret_volatility(vol_annualized: float) -> str:
    pct = vol_annualized * 100
    if pct < 25:
        return f"volatilitas tahunan {pct:.1f}% tergolong rendah -> pergerakan harga relatif tenang"
    if pct < 50:
        return f"volatilitas tahunan {pct:.1f}% tergolong sedang"
    return (
        f"volatilitas tahunan {pct:.1f}% tergolong tinggi -> pergerakan harga cenderung liar, "
        "risiko (dan potensi untung-rugi harian) lebih besar"
    )


def interpret_magnitude(pct: float) -> str:
    a = abs(pct)
    arah = "naik" if pct >= 0 else "turun"
    if a < 1:
        return f"nyaris stabil ({pct:+.2f}%)"
    if a < 5:
        return f"{arah} tipis ({pct:+.2f}%)"
    if a < 15:
        return f"{arah} cukup signifikan ({pct:+.2f}%)"
    return f"{arah} tajam ({pct:+.2f}%)"


def interpret_confidence_gap(accuracy: float, baseline: float) -> str:
    """Membandingkan akurasi model vs baseline naif dalam poin persentase,
    lalu menerjemahkannya ke tingkat keyakinan yang jujur (bukan cuma
    melaporkan angka)."""
    gap_pp = (accuracy - baseline) * 100
    if gap_pp < 0:
        return (
            f"akurasi model justru {abs(gap_pp):.1f} poin persentase di bawah tebakan naif "
            "('selalu tebak naik') -> untuk saham ini, menebak arah secara acak sama saja "
            "atau malah lebih baik dari modelnya. Jangan jadikan ini sinyal utama"
        )
    if gap_pp <= 2:
        return (
            f"model ini hanya unggul {gap_pp:.1f} poin persentase dari tebakan naif "
            "('selalu tebak naik') -> secara statistik nyaris tidak lebih baik dari menebak "
            "acak, konsisten dengan teori pasar efisien (Efficient Market Hypothesis). "
            "Jangan jadikan ini sinyal utama"
        )
    if gap_pp <= 5:
        return (
            f"model ini unggul {gap_pp:.1f} poin persentase dari tebakan naif -> sedikit lebih "
            "baik dari acak, tapi masih lemah untuk dijadikan sinyal tunggal"
        )
    return (
        f"model ini unggul {gap_pp:.1f} poin persentase dari tebakan naif -> cukup meyakinkan "
        "dibanding baseline, meski tetap bukan jaminan (lihat juga precision/recall)"
    )


def interpret_feature(name: str, values: dict) -> str | None:
    """Menjelaskan satu fitur teknikal memakai nilai aktualnya (dict
    `values` berisi semua kolom FEATURE_COLS dari classification.py)."""
    if name == "rsi_14" and "rsi_14" in values:
        return interpret_rsi(values["rsi_14"])
    if name in ("macd", "macd_signal") and "macd" in values and "macd_signal" in values:
        return interpret_macd(values["macd"], values["macd_signal"])
    if name in ("sma_10", "sma_50") and "sma_10" in values and "sma_50" in values:
        return interpret_sma_cross(values["sma_10"], values["sma_50"])
    if name in ("ema_12", "ema_26") and "ema_12" in values and "ema_26" in values:
        return interpret_sma_cross(values["ema_12"], values["ema_26"], "EMA12", "EMA26")
    if name == "volatility_20" and "volatility_20" in values:
        return interpret_volatility(values["volatility_20"])
    if name == "return_1d" and "return_1d" in values:
        pct = values["return_1d"] * 100
        return f"return harian terakhir {pct:+.2f}%"
    if name == "hl_range_pct" and "hl_range_pct" in values:
        return f"rentang high-low harian terakhir {values['hl_range_pct']*100:.2f}% dari harga penutupan"
    return None


# Ambang batas kualitatif untuk memberi "nama" sebuah cluster saham.
# Dipilih dari pengamatan kasar atas distribusi data historis saham
# NASDAQ -- bukan angka baku, hanya untuk membantu interpretasi awam.
_CLUSTER_RETURN_HIGH = 0.0008   # ~20%/tahun return harian rata-rata
_CLUSTER_VOL_HIGH = 0.5         # 50% volatilitas tahunan
_CLUSTER_VOL_LOW = 0.25         # 25% volatilitas tahunan
_CLUSTER_DRAWDOWN_DEEP = -0.4   # -40% drawdown


def interpret_cluster_archetype(mean_daily_return: float, volatility_annualized: float, max_drawdown: float) -> str:
    high_vol = volatility_annualized >= _CLUSTER_VOL_HIGH
    low_vol = volatility_annualized <= _CLUSTER_VOL_LOW
    high_return = mean_daily_return >= _CLUSTER_RETURN_HIGH

    if high_vol and high_return:
        label, desc = "Growth volatil", (
            "return harian rata-rata tinggi tapi diiringi ayunan harga besar -- berpotensi "
            "cuan besar, risiko juga besar"
        )
    elif high_vol and not high_return:
        label, desc = "Spekulatif/berisiko", (
            "ayunan harga besar tanpa return rata-rata yang sepadan -- cenderung lebih "
            "berisiko daripada menguntungkan"
        )
    elif low_vol and high_return:
        label, desc = "Growth stabil", (
            "return harian rata-rata positif dengan pergerakan harga relatif tenang -- "
            "kombinasi yang relatif jarang & menarik"
        )
    elif low_vol and not high_return:
        label, desc = "Defensif/stabil", (
            "pergerakan harga tenang, return rata-rata moderat -- cocok untuk profil yang "
            "menghindari gejolak"
        )
    else:
        label, desc = "Menengah/seimbang", (
            "volatilitas dan return berada di kisaran menengah, tidak condong ke salah satu ekstrem"
        )

    if max_drawdown <= _CLUSTER_DRAWDOWN_DEEP:
        desc += (
            f"; pernah mengalami drawdown dalam (maks {max_drawdown*100:.1f}%), menandakan "
            "ada periode penurunan tajam di masa lalu"
        )

    return f"{label} -- {desc}"


def interpret_anomaly_severity(return_pct: float, typical_move_pct: float | None = None) -> str:
    """`typical_move_pct` idealnya std return harian (dalam %, belum
    diannualisasi) dari saham yang sama, supaya "ekstrem" dinilai relatif
    terhadap kebiasaan saham itu sendiri, bukan ambang batas universal."""
    a = abs(return_pct)
    if typical_move_pct and typical_move_pct > 0:
        multiple = a / typical_move_pct
        tag = "ekstrem" if multiple >= 4 else "signifikan" if multiple >= 2.5 else "cukup besar"
        return f"{tag} -- sekitar {multiple:.1f}x lebih besar dari pergerakan harian normal saham ini"
    tag = "ekstrem" if a >= 15 else "signifikan" if a >= 8 else "cukup besar"
    return f"{tag} ({return_pct:+.2f}% dalam sehari)"
