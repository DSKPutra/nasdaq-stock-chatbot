"""
clustering.py
==============
Mengelompokkan 500 saham NASDAQ berdasarkan karakteristik pergerakan
harga (return rata-rata, volatilitas, RSI rata-rata, drawdown, dsb.)
memakai K-Means.

Alur:
    build_summary_table(tickers)  -> DataFrame ringkasan per saham
    fit_clusters(summary_df, k)   -> (model, summary_df + kolom cluster)
    similar_stocks(ticker, ...)   -> daftar saham paling mirip (cluster
                                      sama + jarak fitur terdekat)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import data_loader, features

FEATURE_COLS = [
    "mean_daily_return",
    "volatility_annualized",
    "avg_rsi",
    "avg_hl_range_pct",
    "pct_days_up",
    "skew_return",
    "max_drawdown",
    "cagr_3y",
]


def build_summary_table(tickers: list[str] | None = None) -> pd.DataFrame:
    """Menghitung fitur ringkasan untuk setiap ticker. Ini proses batch
    (dipanggil sekali lalu di-cache ke parquet/csv), bukan per-request."""
    if tickers is None:
        tickers = data_loader.list_tickers()

    rows = []
    for t in tickers:
        try:
            df = data_loader.load_ticker(t)
            df = features.add_technical_indicators(df)
            summ = features.summary_features(df)
            if summ:
                summ["ticker"] = t
                rows.append(summ)
        except Exception as e:
            print(f"[clustering] Lewati {t}: {e}")

    table = pd.DataFrame(rows).set_index("ticker")
    table = table.dropna(subset=FEATURE_COLS)
    return table


def fit_clusters(summary_df: pd.DataFrame, k: int = 8, random_state: int = 42):
    """Melatih K-Means pada tabel ringkasan. Mengembalikan (model, scaler,
    summary_df dengan kolom 'cluster')."""
    X = summary_df[FEATURE_COLS].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(Xs)

    result = summary_df.copy()
    result["cluster"] = labels
    return model, scaler, result


def similar_stocks(ticker: str, summary_with_clusters: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Mencari saham paling mirip dengan `ticker`: pertama cari yang
    satu cluster, lalu urutkan berdasarkan jarak Euclidean pada fitur
    (setelah distandarisasi) agar hasil paling relevan berada di atas."""
    ticker = ticker.upper()
    if ticker not in summary_with_clusters.index:
        raise ValueError(f"Ticker '{ticker}' tidak ada pada tabel ringkasan.")

    target_cluster = summary_with_clusters.loc[ticker, "cluster"]
    same_cluster = summary_with_clusters[summary_with_clusters["cluster"] == target_cluster]
    same_cluster = same_cluster.drop(index=ticker, errors="ignore")

    if same_cluster.empty:
        return same_cluster

    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(summary_with_clusters[FEATURE_COLS])
    all_scaled_df = pd.DataFrame(all_scaled, index=summary_with_clusters.index, columns=FEATURE_COLS)

    target_vec = all_scaled_df.loc[ticker].values
    dists = same_cluster.index.map(
        lambda t: np.linalg.norm(all_scaled_df.loc[t].values - target_vec)
    )
    same_cluster = same_cluster.copy()
    same_cluster["distance"] = list(dists)
    return same_cluster.sort_values("distance").head(top_n)


def cluster_profile(summary_with_clusters: pd.DataFrame, cluster_id: int) -> pd.Series:
    """Rata-rata fitur untuk satu cluster -> membantu memberi 'nama'
    intuitif ke cluster (mis. 'volatil & bertumbuh cepat')."""
    subset = summary_with_clusters[summary_with_clusters["cluster"] == cluster_id]
    return subset[FEATURE_COLS].mean()


if __name__ == "__main__":
    # smoke test cepat pada subset kecil ticker
    sample_tickers = data_loader.list_tickers()[:40]
    table = build_summary_table(sample_tickers)
    model, scaler, result = fit_clusters(table, k=5)
    print(result[["cluster"] + FEATURE_COLS].head(10))
