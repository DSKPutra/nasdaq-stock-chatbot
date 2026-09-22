"""
data_loader.py
================
Membaca data harga OHLC NASDAQ (satu file CSV per ticker) dari folder
`data/raw/`. Dataset asli: ticker, date, open, high, low, close
(tanpa volume).

Fungsi utama:
    list_tickers()              -> daftar semua ticker yang tersedia
    load_ticker(ticker)         -> DataFrame OHLC satu saham, terurut
                                    berdasarkan tanggal
    load_many(tickers=None)     -> dict {ticker: DataFrame}, default
                                    memuat semua ticker (butuh waktu &
                                    memori lebih besar -> gunakan untuk
                                    proses batch/caching, bukan tiap
                                    request chatbot)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# folder data ada di ../data/raw relatif terhadap file ini
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def list_tickers() -> list[str]:
    """Mengembalikan daftar semua ticker (nama file tanpa .csv), terurut."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Folder data tidak ditemukan: {DATA_DIR}. "
            "Pastikan file CSV per-ticker sudah diekstrak ke data/raw/."
        )
    tickers = sorted(p.stem for p in DATA_DIR.glob("*.csv"))
    if not tickers:
        raise FileNotFoundError(f"Tidak ada file .csv di {DATA_DIR}")
    return tickers


def load_ticker(ticker: str) -> pd.DataFrame:
    """Memuat satu file CSV ticker dan mengembalikan DataFrame yang bersih.

    Kolom hasil: date (datetime64), open, high, low, close (float)
    Terurut naik berdasarkan tanggal, tanpa duplikat tanggal.
    """
    ticker = ticker.upper().strip()
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        raise ValueError(f"Ticker '{ticker}' tidak ditemukan di dataset.")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"ticker", "date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom hilang pada {ticker}.csv: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset="date").sort_values("date")
    df = df.reset_index(drop=True)
    return df[["date", "open", "high", "low", "close"]]


def load_many(tickers: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Memuat banyak ticker sekaligus ke dalam dict. Gunakan untuk proses
    batch (mis. membangun cache fitur clustering), bukan untuk satu kali
    permintaan chatbot."""
    if tickers is None:
        tickers = list_tickers()
    out = {}
    for t in tickers:
        try:
            out[t] = load_ticker(t)
        except Exception as e:  # lewati file yang bermasalah, jangan crash
            print(f"[data_loader] Lewati {t}: {e}")
    return out


if __name__ == "__main__":
    tks = list_tickers()
    print(f"Jumlah ticker: {len(tks)}")
    print("Contoh:", tks[:10])
    sample = load_ticker(tks[0])
    print(sample.head())
    print(sample.tail())
