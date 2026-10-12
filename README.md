# NASDAQku.ID — Chatbot Analisis Saham NASDAQ (UTS Data Mining)

Chatbot yang menjawab pertanyaan seputar 500 saham NASDAQ memakai empat
teknik data mining: feature engineering indikator teknikal, klasifikasi
(prediksi arah harga), clustering (segmentasi saham), dan deteksi
anomali. Data: harga OHLC harian per saham, 1962–2026, tanpa volume.

**Demo web:** https://nasdaq-stock-chatbot-putratalks.vercel.app/ ·
tampilan `webapp/` mengikuti Dea Saka Kurnia Putra Design System (lihat
`webapp/style.css`).

## 1. Struktur Proyek

```
nasdaq_chatbot/
├── data/
│   ├── raw/                 # 500 file CSV per ticker (dari dataset asli)
│   └── processed/
│       └── cluster_summary.csv   # cache hasil clustering (dibuat oleh script)
├── notebooks/
│   ├── 01_eda.py             # eksplorasi data awal + grafik
│   └── figures/              # grafik hasil EDA (.png)
├── scripts/
│   └── build_cluster_cache.py    # hitung & simpan cache clustering
├── src/
│   ├── data_loader.py        # baca CSV per ticker
│   ├── features.py           # indikator teknikal (SMA, EMA, RSI, MACD, dll)
│   ├── classification.py     # prediksi arah harga (Random Forest)
│   ├── clustering.py         # segmentasi saham (K-Means)
│   ├── anomaly.py            # deteksi anomali (Isolation Forest / Z-score)
│   ├── nlu.py                 # deteksi intent + ekstraksi ticker dari teks
│   ├── explain.py             # terjemahan angka -> kalimat yang mudah dipahami
│   ├── charts.py              # spesifikasi + renderer chart (matplotlib) untuk jawaban
│   └── chatbot.py            # orchestrator: teks -> jawaban
├── app.py                    # antarmuka chatbot (Streamlit)
├── cli.py                    # antarmuka chatbot (terminal, tanpa Streamlit)
└── requirements.txt
```

## 2. Cara Menjalankan

```bash
# 1) install dependency
pip install -r requirements.txt

# 2) bangun cache clustering (sekali saja, ±10 detik untuk 500 saham)
python scripts/build_cluster_cache.py

# 3a) jalankan chatbot via web (disarankan untuk demo)
streamlit run app.py

# 3b) atau jalankan via terminal
python cli.py
```

Contoh pertanyaan yang bisa diajukan ke chatbot:
- "Bagaimana tren AAPL 30 hari terakhir?"
- "Prediksi NVDA besok naik atau turun?"
- "Saham apa yang mirip dengan AMD?"
- "AAPL masuk cluster apa?"
- "Kapan TSLA pernah anomali?"
- "Ringkasan lengkap MSFT" — gabungan tren + prediksi + cluster + anomali dalam satu jawaban
- "Bandingkan AAPL vs MSFT" — perbandingan dua saham sekaligus kesimpulan mana yang lebih kuat/tenang

Jawaban juga disertai **chart** bila relevan: garis harga + SMA10/SMA50
(tren), gauge peluang naik/turun dan batang akurasi vs baseline
(prediksi), volatilitas per cluster, return pada hari anomali, serta
perbandingan dua saham (garis kinerja relatif + batang). Versi Streamlit
merender chart lewat matplotlib (`src/charts.py`); versi web statis
merendernya sebagai HTML/CSS (`webapp/charts.js`) dari spesifikasi yang
sama, tanpa grafik garis harga karena `data.json` tidak menyimpan deret
harga.

Setiap jawaban disertai **alasannya** dalam kalimat yang mudah dipahami
(bukan cuma angka mentah) -- mis. bukan cuma "RSI 72.4", tapi "RSI 72.4
-> zona jenuh beli, historisnya rawan koreksi turun". Lihat `src/explain.py`.

## 3. Metodologi (ringkasan untuk laporan)

### 3.1 Data & Preprocessing
Dataset berisi 500 saham NASDAQ, kolom `ticker, date, open, high, low,
close` (tanpa volume), rentang 1962–2026 (panjang riwayat bervariasi
per saham, dari puluhan hari untuk saham baru listing hingga >16.000
hari untuk saham seperti AEP dan HON). Preprocessing meliputi:
parsing tanggal, penghapusan duplikat tanggal, dan pengurutan
kronologis (`src/data_loader.py`).

### 3.2 Feature Engineering (`src/features.py`)
Karena tidak ada kolom volume, seluruh fitur diturunkan dari OHLC:
return harian, SMA(10,50), EMA(12,26), MACD, RSI(14), volatilitas
rolling 20 hari (annualized), Bollinger Bands, dan rentang
high-low harian. Fitur ini menjadi input untuk klasifikasi maupun
clustering.

### 3.3 Klasifikasi Arah Harga (`src/classification.py`)
- **Model**: Random Forest Classifier (label biner: harga besok naik
  atau tidak).
- **Validasi**: *walk-forward split* (80% data awal untuk training,
  20% data paling akhir untuk testing) — bukan random split, karena
  data time series tidak boleh "melihat masa depan" saat training
  (data leakage).
- **Evaluasi**: akurasi, precision, recall, F1, dibandingkan dengan
  baseline naif (selalu menebak "naik"). Pada percobaan (mis. MSFT),
  akurasi model (~52-53%) hanya sedikit di atas baseline — ini
  **temuan yang jujur dan penting**: prediksi arah harga saham jangka
  pendek memang sangat sulit dan mendekati random walk, konsisten
  dengan teori pasar efisien (Efficient Market Hypothesis). Laporan
  UTS sebaiknya membahas temuan ini, bukan hanya melaporkan angka
  akurasi.

### 3.4 Clustering Saham (`src/clustering.py`)
- **Fitur**: return harian rata-rata, volatilitas tahunan, RSI
  rata-rata, rentang high-low rata-rata, persentase hari naik,
  skewness return, max drawdown, dan CAGR 3 tahun terakhir (dihitung
  per saham dari `src/features.py::summary_features`).
- **Model**: K-Means (k=8 default), fitur distandarisasi (StandardScaler)
  sebelum clustering.
- **Kegunaan**: menjawab "saham mirip dengan X" (jarak Euclidean dalam
  cluster yang sama) dan "X termasuk kelompok saham seperti apa".

### 3.5 Deteksi Anomali (`src/anomaly.py`)
- **Isolation Forest** (default) pada kombinasi return harian, rentang
  high-low, dan volatilitas rolling — menangkap anomali multivariat.
- **Z-score** sebagai alternatif yang lebih sederhana/transparan untuk
  dibandingkan di laporan.
- Contoh hasil pada TSLA menangkap tanggal 19 Maret 2020 (crash awal
  COVID-19) dan 9 April 2025 — menunjukkan model berhasil menangkap
  peristiwa pasar yang memang secara faktual ekstrem.

### 3.6 NLU & Chatbot (`src/nlu.py`, `src/chatbot.py`, `src/explain.py`)
Intent classification berbasis keyword matching (rule-based) untuk 8
intent: trend, predict, similar, cluster, anomaly, **overview**
(ringkasan lengkap gabungan semua analisis), **compare** (bandingkan
dua saham), dan help. Ekstraksi ticker dilakukan dua tahap untuk
menghindari kata umum Bahasa Indonesia yang kebetulan sama dengan kode
ticker asli (mis. "apa" vs ticker APA — Apa Corporation): diprioritaskan
token yang ditulis kapital oleh pengguna, dengan daftar stopword
sebagai pengaman. Intent `compare` memakai varian dua-ticker dari
fungsi ekstraksi yang sama (`extract_tickers`).

`src/explain.py` memisahkan logika "model" dari logika "bahasa":
setiap angka (RSI, MACD, volatilitas, gap akurasi vs baseline, dst.)
diterjemahkan jadi kalimat dengan ambang batas kualitatif eksplisit
(mis. RSI ≥70 → jenuh beli, volatilitas tahunan ≥50% → tinggi), supaya
jawaban chatbot menjelaskan *mengapa*, bukan cuma melaporkan angka.
Modul ini murni (tanpa I/O) sehingga mudah diuji unit secara terpisah.

## 4. Keterbatasan & Pengembangan Lanjutan
- Tidak ada data volume perdagangan -> tidak bisa menghitung indikator
  berbasis volume (OBV, VWAP).
- Prediksi arah harga tidak boleh dipakai sebagai saran investasi;
  akurasinya mendekati baseline naif, sesuai ekspektasi teoritis.
- NLU berbasis keyword bisa dikembangkan menjadi TF-IDF + classifier
  atau embedding jika ingin menangani variasi bahasa yang lebih luas.
- Bisa ditambahkan: perbandingan beberapa model klasifikasi (Logistic
  Regression, XGBoost) dan evaluasi hyperparameter tuning untuk
  memperkaya analisis di laporan.

## 5. Hak Cipta

© 2026 **Dea Saka Kurnia Putra**. Seluruh hak cipta dilindungi.

NASDAQku.ID dirancang dan dikembangkan oleh Dea Saka Kurnia Putra,
termasuk identitas visual (logo monogram P dan lockup "Dea Saka Kurnia
Putra — A brighter tomorrow through meaningful work") yang merupakan
merek personal pemiliknya. Kode dan aset dalam repositori ini tidak
boleh digunakan ulang atau didistribusikan ulang tanpa izin tertulis
dari pemilik hak cipta.
