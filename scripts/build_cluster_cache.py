"""
build_cluster_cache.py
=======================
Menghitung fitur ringkasan untuk seluruh (atau sebagian) ticker dan
melatih K-Means, lalu menyimpan hasilnya ke
data/processed/cluster_summary.csv agar chatbot tidak perlu menghitung
ulang setiap kali dijalankan.

Jalankan sekali di awal (atau ulangi berkala jika data diperbarui):
    python scripts/build_cluster_cache.py
    python scripts/build_cluster_cache.py --limit 100   # subset cepat untuk uji coba
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clustering, data_loader

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cluster_summary.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah ticker (untuk uji cepat)")
    parser.add_argument("--k", type=int, default=8, help="Jumlah cluster K-Means")
    args = parser.parse_args()

    tickers = data_loader.list_tickers()
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"Menghitung fitur ringkasan untuk {len(tickers)} ticker...")
    t0 = time.time()
    summary = clustering.build_summary_table(tickers)
    print(f"Selesai dalam {time.time()-t0:.1f}s. {len(summary)} ticker valid dari {len(tickers)}.")

    model, scaler, result = clustering.fit_clusters(summary, k=args.k)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH)
    print(f"Tersimpan ke {OUT_PATH}")
    print(result["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
