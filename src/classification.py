"""
classification.py
===================
Memprediksi arah harga (naik/turun) hari berikutnya untuk satu saham,
memakai indikator teknikal sebagai fitur dan Random Forest sebagai
model. Validasi memakai walk-forward split (bukan random split) supaya
tidak terjadi kebocoran data (data leakage) dari masa depan ke masa
lalu -- ini penting untuk data time series dan sering jadi poin
penilaian di UTS Data Mining.

Fungsi utama:
    build_dataset(df_with_indicators)  -> (X, y) siap latih
    train_and_evaluate(ticker)         -> model, metrik evaluasi
    predict_next_direction(ticker)     -> prediksi + probabilitas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from . import data_loader, features

FEATURE_COLS = [
    "return_1d",
    "sma_10",
    "sma_50",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rsi_14",
    "volatility_20",
    "hl_range_pct",
]


def build_dataset(df_ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Menyiapkan matriks fitur X, label y, dan kolom tanggal (untuk
    split berbasis waktu). Baris terakhir (tanpa label, karena besok
    belum terjadi) dipisah untuk prediksi live."""
    d = df_ind.dropna(subset=FEATURE_COLS).copy()
    live_row = d.iloc[[-1]]  # baris terbaru, dipakai untuk prediksi ke depan
    train_part = d.iloc[:-1].dropna(subset=["target_up"])
    X = train_part[FEATURE_COLS]
    y = train_part["target_up"]
    dates = train_part["date"]
    return X, y, dates, live_row


def train_and_evaluate(ticker: str, test_size: float = 0.2, n_estimators: int = 300):
    """Melatih Random Forest dengan walk-forward split (train = masa lalu,
    test = periode paling akhir) dan mengembalikan model + metrik."""
    df = data_loader.load_ticker(ticker)
    if len(df) < 300:
        raise ValueError(
            f"Data {ticker} terlalu pendek ({len(df)} baris) untuk melatih model yang andal."
        )
    df_ind = features.add_technical_indicators(df)
    X, y, dates, live_row = build_dataset(df_ind)

    if len(X) < 100:
        raise ValueError(f"Data valid {ticker} setelah indikator terlalu sedikit ({len(X)} baris).")

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=6, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "baseline_naive_up_rate": y_test.mean(),  # jika selalu tebak "naik"
        "test_period_start": str(dates.iloc[split_idx].date()),
        "test_period_end": str(dates.iloc[-1].date()),
    }

    feature_importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "live_row": live_row,
        "last_close": float(df_ind["close"].iloc[-1]),
        "last_date": str(df_ind["date"].iloc[-1].date()),
    }


def predict_next_direction(ticker: str) -> dict:
    """Melatih model untuk `ticker` lalu memprediksi arah hari
    berikutnya berdasarkan data paling akhir. Mengembalikan label,
    probabilitas, dan metrik evaluasi model (agar jujur soal
    keandalannya)."""
    result = train_and_evaluate(ticker)
    model = result["model"]
    live_X = result["live_row"][FEATURE_COLS]
    proba_up = float(model.predict_proba(live_X)[0][1])
    direction = "NAIK" if proba_up >= 0.5 else "TURUN"

    return {
        "ticker": ticker.upper(),
        "direction": direction,
        "probability_up": proba_up,
        "last_close": result["last_close"],
        "last_date": result["last_date"],
        "metrics": result["metrics"],
        "top_features": result["feature_importance"].head(5).to_dict(),
        # nilai aktual tiap fitur pada baris terbaru -- dipakai untuk
        # menjelaskan *mengapa* model condong ke arah tertentu, bukan
        # cuma menyebut nama fiturnya (lihat src/explain.py).
        "feature_values": {col: float(result["live_row"][col].iloc[0]) for col in FEATURE_COLS},
    }


if __name__ == "__main__":
    out = predict_next_direction("AAPL")
    import json
    print(json.dumps(out, indent=2, default=float))
