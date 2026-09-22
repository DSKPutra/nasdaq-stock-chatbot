"""
anomaly.py
==========
Deteksi hari-hari dengan pergerakan harga tidak wajar (anomali) pada
satu saham, memakai dua pendekatan sederhana namun umum dipakai:
    1. Z-score pada return harian (naive, mudah dijelaskan)
    2. Isolation Forest pada beberapa fitur sekaligus (return,
       hl_range_pct, volatility_20) -- menangkap anomali multivariat.

Fungsi utama:
    detect_anomalies(ticker, method="isolation_forest"|"zscore")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import data_loader, features

ANOMALY_FEATURE_COLS = ["return_1d", "hl_range_pct", "volatility_20"]


def detect_anomalies_zscore(df_ind: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    d = df_ind.dropna(subset=["return_1d"]).copy()
    mean, std = d["return_1d"].mean(), d["return_1d"].std()
    d["zscore"] = (d["return_1d"] - mean) / std
    anomalies = d[d["zscore"].abs() >= threshold]
    return anomalies[["date", "close", "return_1d", "zscore"]].sort_values(
        "zscore", key=lambda s: s.abs(), ascending=False
    )


def detect_anomalies_isolation_forest(df_ind: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    d = df_ind.dropna(subset=ANOMALY_FEATURE_COLS).copy()
    if len(d) < 100:
        raise ValueError("Data terlalu pendek untuk Isolation Forest.")

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    d["anomaly_flag"] = model.fit_predict(d[ANOMALY_FEATURE_COLS])  # -1 = anomali
    d["anomaly_score"] = model.decision_function(d[ANOMALY_FEATURE_COLS])  # makin kecil makin anomali

    anomalies = d[d["anomaly_flag"] == -1].copy()
    return anomalies[["date", "close", "return_1d", "hl_range_pct", "volatility_20", "anomaly_score"]].sort_values(
        "anomaly_score"
    )


def detect_anomalies(ticker: str, method: str = "isolation_forest", top_n: int = 10) -> pd.DataFrame:
    df = data_loader.load_ticker(ticker)
    df_ind = features.add_technical_indicators(df)

    if method == "zscore":
        result = detect_anomalies_zscore(df_ind)
    elif method == "isolation_forest":
        result = detect_anomalies_isolation_forest(df_ind)
    else:
        raise ValueError("method harus 'zscore' atau 'isolation_forest'")

    return result.head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    print(detect_anomalies("AAPL", method="isolation_forest", top_n=10))
