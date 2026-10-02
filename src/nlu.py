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
    "compare": ["bandingkan", "dibanding", "dibandingkan", "perbandingan", " vs ", " vs. ", " versus "],
    "overview": ["ringkasan", "rangkuman", "overview", "profil lengkap", "laporan lengkap", "ceritakan semua", "selengkapnya"],
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
    ticker2: str | None = None


def extract_tickers(text: str, max_n: int = 2) -> list[str]:
    """Mencari hingga `max_n` kode ticker valid di dalam teks, dalam
    urutan kemunculan. Strategi dua tahap agar kata umum Bahasa
    Indonesia yang kebetulan sama dengan kode ticker (mis. "apa" vs
    ticker APA) tidak salah tertangkap:
      1. Cari token yang ditulis KAPITAL SEMUA oleh user (mis. "AMD",
         "NVDA") dan cocok dengan ticker asli -> prioritas utama.
      2. Jika belum cukup, cari token apa pun yang cocok dengan ticker,
         KECUALI token tersebut ada di daftar kata umum yang ambigu.
    """
    tickers = _known_tickers()
    candidates = re.findall(r"\b[A-Za-z]{1,5}\b", text)
    found: list[str] = []

    for c in candidates:
        if len(found) >= max_n:
            return found
        if c.isupper() and len(c) >= 2 and c in tickers and c not in found:
            found.append(c)

    for c in candidates:
        if len(found) >= max_n:
            return found
        cu = c.upper()
        if c.lower() in _AMBIGUOUS_STOPWORDS:
            continue
        if cu in tickers and cu not in found:
            found.append(cu)

    return found


def extract_ticker(text: str) -> str | None:
    """Ticker pertama yang ditemukan di teks, atau None. Lihat
    `extract_tickers` untuk strateginya."""
    found = extract_tickers(text, max_n=1)
    return found[0] if found else None


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
    if intent == "compare":
        tks = extract_tickers(text, max_n=2)
        ticker = tks[0] if len(tks) > 0 else None
        ticker2 = tks[1] if len(tks) > 1 else None
        return ParsedQuery(intent=intent, ticker=ticker, ticker2=ticker2, raw_text=text)
    ticker = extract_ticker(text)
    return ParsedQuery(intent=intent, ticker=ticker, raw_text=text)


if __name__ == "__main__":
    tests = [
        "Prediksi AAPL besok naik atau turun?",
        "Bagaimana tren NVDA 3 bulan terakhir?",
        "Saham apa yang mirip dengan AMD?",
        "Kelompokkan saham semikonduktor",
        "Kapan TSLA pernah anomali?",
        "Ringkasan lengkap MSFT",
        "Bandingkan AAPL vs MSFT",
        "Bisa bantu apa saja?",
    ]
    for t in tests:
        print(t, "->", parse(t))
