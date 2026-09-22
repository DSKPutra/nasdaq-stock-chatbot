"""
build_webapp_data.py
=====================
Menghitung SEMUA hasil data mining (trend, prediksi arah harga, cluster,
saham mirip, anomali) untuk 500 ticker sekali di awal, lalu menyimpannya
sebagai satu file JSON statis (webapp/data.json).

Kenapa dibuat statis (bukan dihitung on-the-fly seperti versi Streamlit)?
Karena web app versi Vercel/Netlify/Lovable adalah situs statis + JS
(tanpa server Python yang hidup terus), sehingga semua komputasi berat
(Random Forest, Isolation Forest, K-Means) harus sudah selesai sebelum
di-deploy. Ini juga membuat chatbot merespons instan di browser.

Jalankan (butuh ~10-15 menit untuk 500 ticker):
    python scripts/build_webapp_data.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import anomaly, classification, clustering, data_loader, features

OUT_PATH = Path(__file__).resolve().parent.parent / "webapp" / "data.json"
CLUSTER_CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "cluster_summary.csv"


def build_trend(df):
    last = df.tail(1).iloc[0]
    prev_30 = df.tail(31).iloc[0] if len(df) > 31 else df.iloc[0]
    prev_252 = df.tail(253).iloc[0] if len(df) > 253 else df.iloc[0]
    return {
        "last_close": round(float(last["close"]), 2),
        "last_date": str(last["date"].date()),
        "last_high": round(float(last["high"]), 2),
        "last_low": round(float(last["low"]), 2),
        "change_30d_pct": round((last["close"] / prev_30["close"] - 1) * 100, 2),
        "change_1y_pct": round((last["close"] / prev_252["close"] - 1) * 100, 2),
    }


def main():
    import pandas as pd

    tickers = data_loader.list_tickers()
    print(f"Menghitung data untuk {len(tickers)} ticker...")

    cluster_table = pd.read_csv(CLUSTER_CACHE, index_col="ticker")

    result = {}
    t_start = time.time()
    for i, t in enumerate(tickers, 1):
        try:
            df = data_loader.load_ticker(t)
            if len(df) < 100:
                continue
            trend = build_trend(df)

            entry = {"trend": trend}

            # prediksi arah harga (butuh riwayat cukup panjang)
            if len(df) >= 300:
                try:
                    pred = classification.predict_next_direction(t)
                    entry["prediction"] = {
                        "direction": pred["direction"],
                        "probability_up": round(pred["probability_up"], 4),
                        "accuracy": round(pred["metrics"]["accuracy"], 4),
                        "precision": round(pred["metrics"]["precision"], 4),
                        "recall": round(pred["metrics"]["recall"], 4),
                        "f1": round(pred["metrics"]["f1"], 4),
                        "baseline_naive_up_rate": round(pred["metrics"]["baseline_naive_up_rate"], 4),
                        "n_test": pred["metrics"]["n_test"],
                        "test_period_start": pred["metrics"]["test_period_start"],
                        "test_period_end": pred["metrics"]["test_period_end"],
                        "top_features": list(pred["top_features"].keys())[:3],
                    }
                except Exception as e:
                    print(f"  [predict] lewati {t}: {e}")

            # anomali
            try:
                anomalies = anomaly.detect_anomalies(t, method="isolation_forest", top_n=5)
                entry["anomalies"] = [
                    {
                        "date": str(row["date"].date()),
                        "close": round(float(row["close"]), 2),
                        "return_pct": round(float(row["return_1d"]) * 100, 2),
                    }
                    for _, row in anomalies.iterrows()
                ]
            except Exception as e:
                print(f"  [anomaly] lewati {t}: {e}")
                entry["anomalies"] = []

            # cluster + saham mirip
            if t in cluster_table.index:
                cid = int(cluster_table.loc[t, "cluster"])
                entry["cluster"] = cid
                try:
                    sims = clustering.similar_stocks(t, cluster_table, top_n=5)
                    entry["similar"] = list(sims.index)
                except Exception:
                    entry["similar"] = []

            result[t] = entry

        except Exception as e:
            print(f"Lewati {t}: {e}")

        if i % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  {i}/{len(tickers)} selesai ({elapsed:.0f}s, ~{elapsed/i*len(tickers):.0f}s total estimasi)")

    # profil tiap cluster (rata-rata fitur)
    cluster_profiles = {}
    for cid in sorted(cluster_table["cluster"].unique()):
        profile = clustering.cluster_profile(cluster_table, int(cid))
        members = cluster_table[cluster_table["cluster"] == cid].index.tolist()
        cluster_profiles[str(int(cid))] = {
            "mean_daily_return": round(float(profile["mean_daily_return"]), 5),
            "volatility_annualized": round(float(profile["volatility_annualized"]), 4),
            "avg_rsi": round(float(profile["avg_rsi"]), 2),
            "max_drawdown": round(float(profile["max_drawdown"]), 4),
            "n_members": len(members),
            "sample_members": members[:15],
        }

    output = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tickers": len(result),
        "tickers": sorted(result.keys()),
        "stocks": result,
        "cluster_profiles": cluster_profiles,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nSelesai dalam {time.time()-t_start:.0f}s. Tersimpan ke {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
