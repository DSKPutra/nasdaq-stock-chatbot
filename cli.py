"""
cli.py
======
Antarmuka chatbot berbasis terminal (alternatif ringan tanpa Streamlit,
juga berguna untuk demo cepat / debugging).

Jalankan dengan:
    python cli.py
"""

from src import chatbot


def main():
    print("=" * 60)
    print(" NASDAQku.ID — Chatbot Analisis Saham NASDAQ (mode terminal)")
    print(" Ketik 'exit' atau 'keluar' untuk berhenti.")
    print("=" * 60)
    print(chatbot.HELP_TEXT)
    print()

    while True:
        try:
            text = input("Anda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break

        if not text:
            continue
        if text.lower() in {"exit", "keluar", "quit"}:
            print("Sampai jumpa!")
            break

        reply = chatbot.answer(text)
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    main()
