"""
app.py
======
Antarmuka chatbot berbasis Streamlit.

Jalankan dengan:
    streamlit run app.py

Prasyarat: cache clustering sudah dibuat (sekali saja):
    python scripts/build_cluster_cache.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import chatbot, data_loader

_ICON = Path(__file__).resolve().parent / "webapp" / "assets" / "mark-256.jpeg"
st.set_page_config(
    page_title="NASDAQku.ID — Chatbot Analisis Saham NASDAQ",
    page_icon=str(_ICON) if _ICON.exists() else None,
    layout="centered",
)

st.title("NASDAQku.ID")
st.caption(
    "Chatbot analisis saham NASDAQ berbasis data mining: klasifikasi arah harga, "
    "clustering saham, dan deteksi anomali dari data OHLC harian 500 saham NASDAQ (1962-2026)."
)

with st.sidebar:
    st.header("Tentang")
    st.markdown(
        "Proyek UTS Data Mining — chatbot analisis saham US (NASDAQ).\n\n"
        "**Kemampuan:**\n"
        "- Tren harga\n"
        "- Prediksi arah harga (Random Forest)\n"
        "- Saham serupa (K-Means clustering)\n"
        "- Deteksi anomali (Isolation Forest)\n\n"
        "_Bukan saran investasi — untuk tujuan edukasi._"
    )
    try:
        tickers = data_loader.list_tickers()
        st.metric("Jumlah saham tersedia", len(tickers))
        with st.expander("Lihat daftar ticker"):
            st.write(", ".join(tickers))
    except Exception as e:
        st.error(str(e))

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": chatbot.HELP_TEXT}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan sesuatu, mis. 'Prediksi AAPL besok naik atau turun?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis..."):
            reply = chatbot.answer(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
