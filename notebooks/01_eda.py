"""
01_eda.py
=========
Eksplorasi data awal (EDA) untuk laporan UTS. Skrip ini menghasilkan
beberapa grafik ringkasan ke notebooks/figures/ dan mencetak statistik
deskriptif dataset OHLC NASDAQ.

Bisa dijalankan langsung sebagai skrip:
    python notebooks/01_eda.py
Atau disalin sel-per-sel ke Jupyter Notebook (mis. Google Colab / VSCode)
untuk eksplorasi interaktif -- komentar '# %%' menandai batas sel.
"""

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_loader, features

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# %% [1] Ringkasan dataset ------------------------------------------------
tickers = data_loader.list_tickers()
print(f"Jumlah saham (ticker): {len(tickers)}")

lengths = {}
first_dates = {}
for t in tickers:
    df = data_loader.load_ticker(t)
    lengths[t] = len(df)
    first_dates[t] = df["date"].min()

meta = pd.DataFrame({"n_rows": lengths, "first_date": first_dates})
print("\nStatistik panjang riwayat data (jumlah hari) per saham:")
print(meta["n_rows"].describe())

print("\n5 saham dengan riwayat terpanjang:")
print(meta.sort_values("n_rows", ascending=False).head())

print("\n5 saham dengan riwayat terpendek (baru listing):")
print(meta.sort_values("n_rows").head())

# %% [2] Distribusi tahun mulai (kapan saham mulai listing/tercatat) ------
meta["first_year"] = meta["first_date"].dt.year
plt.figure(figsize=(10, 4))
meta["first_year"].value_counts().sort_index().plot(kind="bar")
plt.title("Jumlah Saham Berdasarkan Tahun Data Mulai Tersedia")
plt.xlabel("Tahun")
plt.ylabel("Jumlah saham")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_distribusi_tahun_mulai.png", dpi=120)
plt.close()
print(f"\nGrafik disimpan: {FIG_DIR / '01_distribusi_tahun_mulai.png'}")

# %% [3] Contoh tren harga (AAPL, NVDA, MSFT) ------------------------------
sample_tickers = [t for t in ["AAPL", "NVDA", "MSFT", "AMD"] if t in tickers]
plt.figure(figsize=(10, 5))
for t in sample_tickers:
    df = data_loader.load_ticker(t)
    df = df[df["date"] >= "2020-01-01"]
    normalized = df["close"] / df["close"].iloc[0] * 100
    plt.plot(df["date"], normalized, label=t)
plt.title("Harga Ternormalisasi Sejak 2020 (basis = 100)")
plt.xlabel("Tanggal")
plt.ylabel("Indeks harga (2020-01 = 100)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "02_tren_harga_ternormalisasi.png", dpi=120)
plt.close()
print(f"Grafik disimpan: {FIG_DIR / '02_tren_harga_ternormalisasi.png'}")

# %% [4] Distribusi return harian (memeriksa asumsi normalitas) -----------
df_aapl = data_loader.load_ticker("AAPL")
df_aapl = features.add_technical_indicators(df_aapl)
returns = df_aapl["return_1d"].dropna()

plt.figure(figsize=(8, 4))
plt.hist(returns, bins=100, density=True)
plt.title("Distribusi Return Harian AAPL (seluruh riwayat)")
plt.xlabel("Return harian")
plt.ylabel("Densitas")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_distribusi_return_aapl.png", dpi=120)
plt.close()
print(f"Grafik disimpan: {FIG_DIR / '03_distribusi_return_aapl.png'}")

print(f"\nSkewness return AAPL: {returns.skew():.3f}")
print(f"Kurtosis return AAPL: {returns.kurt():.3f}  (>>0 = fat tail / rawan outlier)")

# %% [5] Korelasi indikator teknikal dengan target arah harga --------------
corr = df_aapl[
    ["return_1d", "sma_10", "sma_50", "macd", "rsi_14", "volatility_20", "hl_range_pct", "target_up"]
].corr()["target_up"].sort_values(ascending=False)
print("\nKorelasi fitur dengan target_up (AAPL) -- catatan: korelasi linear")
print("kecil di semua fitur adalah hal yang WAJAR untuk data harga saham,")
print("dan menjadi alasan mengapa model non-linear (Random Forest) dipakai:")
print(corr)

print("\nSelesai. Lihat folder notebooks/figures/ untuk grafik yang dihasilkan.")
