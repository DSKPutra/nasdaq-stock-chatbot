"""
features.py
============
Feature engineering dari data OHLC harian menjadi indikator teknikal.
Karena dataset tidak punya volume, semua fitur diturunkan dari
open/high/low/close saja.

Fungsi utama:
    add_technical_indicators(df) -> DataFrame df + kolom indikator
    summary_features(df)         -> 1 baris ringkasan statistik (untuk
                                     clustering antar saham)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan kolom indikator teknikal ke DataFrame OHLC.

    Kolom baru:
        return_1d      : return harian (close-to-close)
        log_return_1d  : log return harian
        sma_10/50      : simple moving average
        ema_12/26      : exponential moving average
        macd           : ema_12 - ema_26
        macd_signal    : EMA-9 dari macd
        rsi_14         : Relative Strength Index
        volatility_20  : rolling std return 20 hari (annualisasi sederhana)
        bb_upper/lower : Bollinger Bands (SMA20 +/- 2*std20)
        hl_range_pct   : (high-low)/close, proxy rentang harian
        target_up      : label biner, 1 jika close besok > close hari ini
    """
    out = df.copy()
    out["return_1d"] = out["close"].pct_change()
    out["log_return_1d"] = np.log(out["close"]).diff()

    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["rsi_14"] = _rsi(out["close"], 14)

    out["volatility_20"] = out["return_1d"].rolling(20).std() * np.sqrt(252)

    sma20 = out["close"].rolling(20).mean()
    std20 = out["close"].rolling(20).std()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20

    out["hl_range_pct"] = (out["high"] - out["low"]) / out["close"]

    # label untuk klasifikasi arah harga besok (1 = naik, 0 = turun/tetap)
    out["target_up"] = (out["close"].shift(-1) > out["close"]).astype(int)

    return out


def summary_features(df_with_indicators: pd.DataFrame) -> dict:
    """Meringkas satu saham (yang sudah punya indikator) menjadi satu baris
    fitur statistik untuk clustering antar saham."""
    d = df_with_indicators.dropna(subset=["return_1d"])
    if len(d) < 60:
        return {}
    recent = d.tail(252 * 3)  # ~3 tahun terakhir agar representatif & adil
    return {
        "mean_daily_return": recent["return_1d"].mean(),
        "volatility_annualized": recent["return_1d"].std() * np.sqrt(252),
        "avg_rsi": recent["rsi_14"].mean(),
        "avg_hl_range_pct": recent["hl_range_pct"].mean(),
        "pct_days_up": (recent["return_1d"] > 0).mean(),
        "skew_return": recent["return_1d"].skew(),
        "max_drawdown": _max_drawdown(recent["close"]),
        "cagr_3y": _cagr(recent["close"]),
        "n_obs": len(recent),
    }


def _max_drawdown(close: pd.Series) -> float:
    cummax = close.cummax()
    drawdown = (close - cummax) / cummax
    return drawdown.min()


def _cagr(close: pd.Series) -> float:
    if len(close) < 2 or close.iloc[0] <= 0:
        return np.nan
    years = len(close) / 252
    if years <= 0:
        return np.nan
    return (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1
