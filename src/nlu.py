"""
nlu.py
======
Natural Language Understanding sederhana berbasis aturan (rule-based /
keyword matching) untuk chatbot. Tugasnya dua hal:
    1. Menentukan intent pertanyaan pengguna (tren, prediksi, mirip,
       cluster, anomali, bantuan)
    2. Mengekstrak ticker saham yang disebut dalam pertanyaan

Pendekatan ini sengaja dibuat ringan & transparan (mudah dijelaskan di
laporan UTS) dibanding memakai model embedding besar, karena jumlah
intent-nya kecil dan pola pertanyaannya cukup terbatas. Bisa
dikembangkan lebih lanjut memakai TF-IDF + classifier bila diperlukan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import data_loader

_TICKERS_CACHE: set[str] | None = None


def _known_tickers() -> set[str]:
    global _TICKERS_CACHE
    if _TICKERS_CACHE is None:
        _TICKERS_CACHE = set(data_loader.list_tickers())
    return _TICKERS_CACHE


INTENT_KEYWORDS = {
    "predict": ["prediksi", "predict", "besok naik", "besok turun", "arah harga", "akan naik", "akan turun", "forecast"],
    "trend": ["tren", "trend", "bagaimana performa", "grafik", "pergerakan", "sejarah harga", "history"],
    "similar": ["mirip", "serupa", "similar", "seperti"],
    "cluster": ["kelompok", "cluster", "kategori saham", "segmen"],
    "anomaly": ["anomali", "anomaly", "lonjakan", "aneh", "tidak wajar", "ekstrem"],
    "help": ["help", "bantuan", "bantu", "bisa apa", "menu", "cara pakai"],
}

# Kata umum (Indonesia & Inggris) yang kebetulan sama dengan kode ticker asli
# (mis. "apa" == ticker APA, "act" == ticker ACT, "aur" == ticker AUR).
# Token-token ini hanya dianggap ticker jika user MENULISNYA dengan huruf
# kapital semua di kalimat asli; kalau tidak, diabaikan supaya chatbot tidak
# salah tangkap.
_AMBIGUOUS_STOPWORDS = {
    "apa", "act", "aur", "are", "can", "for", "has", "was", "the", "and",
    "all", "arm", "app", "ada", "dan", "itu", "ini", "dari", "pada",
    "untuk", "dengan", "yang", "bisa", "saja", "akan", "atau", "jika",
    "saya", "kamu", "kah", "nya", "jadi", "juga", "saat", "agar", "amp",
}


@dataclass
class ParsedQuery:
    intent: str
    ticker: str | None
    raw_text: str


def extract_ticker(text: str) -> str | None:
    """Mencari kode ticker yang valid di dalam teks pertanyaan.

    Strategi dua tahap agar kata umum Bahasa Indonesia yang kebetulan
    sama dengan kode ticker (mis. "apa" vs ticker APA) tidak salah
    tertangkap:
      1. Cari token yang ditulis KAPITAL SEMUA oleh user (mis. "AMD",
         "NVDA") dan cocok dengan ticker asli -> prioritas utama.
      2. Jika tidak ada, cari token apa pun yang cocok dengan ticker,
         KECUALI token tersebut ada di daftar kata umum yang ambigu.
    """
    tickers = _known_tickers()
    candidates = re.findall(r"\b[A-Za-z]{1,5}\b", text)

    # tahap 1: token yang memang ditulis kapital semua oleh pengguna
    for c in candidates:
        if c.isupper() and len(c) >= 2 and c in tickers:
            return c

    # tahap 2: fallback case-insensitive, minus kata umum yang ambigu
    for c in candidates:
        cu = c.upper()
        if c.lower() in _AMBIGUOUS_STOPWORDS:
            continue
        if cu in tickers:
            return cu
    return None


def classify_intent(text: str) -> str:
    lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    # default: kalau menyebut ticker tapi tidak jelas maksudnya -> anggap trend
    if extract_ticker(text):
        return "trend"
    return "unknown"


def parse(text: str) -> ParsedQuery:
    intent = classify_intent(text)
    ticker = extract_ticker(text)
    return ParsedQuery(intent=intent, ticker=ticker, raw_text=text)


if __name__ == "__main__":
    tests = [
        "Prediksi AAPL besok naik atau turun?",
        "Bagaimana tren NVDA 3 bulan terakhir?",
        "Saham apa yang mirip dengan AMD?",
        "Kelompokkan saham semikonduktor",
        "Kapan TSLA pernah anomali?",
        "Bisa bantu apa saja?",
    ]
    for t in tests:
        print(t, "->", parse(t))
