"""
charts.py
=========
Pembuat chart untuk jawaban chatbot. Dipisah jadi dua lapis:

1. Builder (fungsi `*_charts`) -> menghasilkan *spesifikasi* chart berupa
   dict sederhana (tanpa dependensi plotting), mis.
       {"kind": "bars", "title": ..., "caption": ..., "items": [...]}
   Format yang sama dipakai webapp/charts.js, jadi tampilan konsisten
   di web statis dan Streamlit.
2. Renderer `render(spec)` -> matplotlib Figure bertema design system
   (ink/paper, hijau hanya sebagai aksen). Dipakai app.py (Streamlit).

Jenis chart: "bars" (batang horizontal, bertanda +/- atau tidak),
"grouped" (dua seri per baris, untuk perbandingan), "prob" (gauge
probabilitas naik/turun), dan "line" (garis harga + SMA; hanya ada di
versi Python karena data.json web tidak menyimpan deret harga).
"""

from __future__ import annotations

from typing import Any

TONES = {
    "ink": "#080A09",
    "mid": "#7C807D",
    "muted": "#A5A8A5",
    "pos": "#12A227",
    "neg": "#B3261E",
    "accent": "#21D138",
}
_PAPER, _HAIRLINE, _TEXT = "#FFFFFF", "#D8D8D3", "#171B19"

Spec = dict[str, Any]


def _tone_for(value: float) -> str:
    return "pos" if value >= 0 else "neg"


# ---------------------------------------------------------------- builders

def trend_charts(df_ind, change_30d: float, change_1y: float, days: int = 180) -> list[Spec]:
    """Garis harga penutupan + SMA10/SMA50 (`days` hari terakhir) dan
    batang perubahan harga 30 hari / ~1 tahun."""
    tail = df_ind.tail(days)
    charts: list[Spec] = [
        {
            "kind": "line",
            "title": f"Harga penutupan {days} hari terakhir",
            "caption": (
                "Garis tebal = harga penutupan. Garis hijau (SMA10) adalah rata-rata 10 hari, "
                "garis abu-abu (SMA50) rata-rata 50 hari. Saat SMA10 berada di atas SMA50, tren "
                "jangka menengah cenderung naik; sebaliknya cenderung turun."
            ),
            "x": [str(d.date()) for d in tail["date"]],
            "series": [
                {"label": "Close", "values": [float(v) for v in tail["close"]], "tone": "ink"},
                {"label": "SMA10", "values": [None if v != v else float(v) for v in tail["sma_10"]], "tone": "pos"},
                {"label": "SMA50", "values": [None if v != v else float(v) for v in tail["sma_50"]], "tone": "mid"},
            ],
        },
        price_change_chart(change_30d, change_1y),
    ]
    return charts


def price_change_chart(change_30d: float, change_1y: float) -> Spec:
    return {
        "kind": "bars",
        "title": "Perubahan harga",
        "caption": "Hijau = naik, merah = turun. Makin panjang batang, makin besar perubahannya.",
        "signed": True,
        "cap": 150,
        "unit": "%",
        "items": [
            {"label": "30 hari", "value": round(change_30d, 2), "tone": _tone_for(change_30d)},
            {"label": "~1 tahun", "value": round(change_1y, 2), "tone": _tone_for(change_1y)},
        ],
    }


def prediction_charts(direction: str, p_up: float, metrics: dict) -> list[Spec]:
    acc, base = metrics["accuracy"] * 100, metrics["baseline_naive_up_rate"] * 100
    beats = acc > base
    return [
        {
            "kind": "prob",
            "title": "Probabilitas arah harga besok",
            "caption": "Batang dibagi dua: bagian hijau = peluang naik, merah = peluang turun. Garis tengah = 50%.",
            "p_up": round(p_up, 4),
        },
        {
            "kind": "bars",
            "title": "Seberapa andal model ini?",
            "caption": (
                "Garis putus-putus = baseline (tebakan 'selalu naik'). "
                + (
                    "Akurasi model melewati garis baseline, artinya model lebih baik dari tebakan naif."
                    if beats
                    else "Akurasi model tidak melewati garis baseline, artinya model tidak lebih baik dari tebakan naif."
                )
            ),
            "unit": "%",
            "domain": [0, 100],
            "guide": {"value": round(base, 1), "label": "baseline"},
            "items": [
                {"label": "Akurasi model", "value": round(acc, 1), "tone": "ink"},
                {"label": "Baseline (selalu naik)", "value": round(base, 1), "tone": "muted"},
                {"label": "Precision", "value": round(metrics["precision"] * 100, 1), "tone": "mid"},
                {"label": "Recall", "value": round(metrics["recall"] * 100, 1), "tone": "mid"},
                {"label": "F1", "value": round(metrics["f1"] * 100, 1), "tone": "mid"},
            ],
        },
    ]


def cluster_chart(cache, cluster_id: int, ticker: str) -> Spec:
    by_cluster = cache.groupby("cluster")["volatility_annualized"].agg(["mean", "size"])
    items = [
        {
            "label": f"Cluster #{int(cid)} ({int(row['size'])})",
            "value": round(float(row["mean"]) * 100, 1),
            "tone": "ink" if int(cid) == int(cluster_id) else "muted",
        }
        for cid, row in by_cluster.iterrows()
    ]
    return {
        "kind": "bars",
        "title": "Volatilitas tahunan rata-rata per cluster",
        "caption": (
            f"Batang gelap = cluster {ticker}. Angka dalam kurung = jumlah saham di cluster itu. "
            "Makin panjang batang, makin liar pergerakan harga rata-rata sahamnya."
        ),
        "unit": "%",
        "cap": 150,
        "items": items,
    }


def similar_chart(ticker: str, target_vol: float, sims) -> Spec:
    items = [{"label": f"{ticker} (acuan)", "value": round(target_vol * 100, 1), "tone": "ink"}]
    for t, row in sims.iterrows():
        items.append({"label": str(t), "value": round(float(row["volatility_annualized"]) * 100, 1), "tone": "mid"})
    return {
        "kind": "bars",
        "title": f"Volatilitas tahunan: {ticker} vs saham mirip",
        "caption": "Saham yang mirip seharusnya punya panjang batang yang berdekatan dengan saham acuan.",
        "unit": "%",
        "cap": 150,
        "items": items,
    }


def anomaly_chart(anomalies, ticker: str) -> Spec:
    items = [
        {
            "label": str(row["date"].date()),
            "value": round(float(row["return_1d"]) * 100, 2),
            "tone": _tone_for(float(row["return_1d"])),
        }
        for _, row in anomalies.iterrows()
    ]
    return {
        "kind": "bars",
        "title": f"Return harian pada hari-hari anomali {ticker}",
        "caption": "Hijau = hari naik, merah = hari turun. Hari anomali bisa juga ditandai karena rentang harian yang lebar, bukan hanya return besar.",
        "signed": True,
        "cap": 100,
        "unit": "%",
        "items": items,
    }


def compare_charts(t1: str, t2: str, df1, df2, chg1: float, chg2: float, yr1: float, yr2: float,
                   vol1: float | None, vol2: float | None, days: int = 180) -> list[Spec]:
    def rebased(df):
        tail = df.tail(days)
        base = float(tail["close"].iloc[0])
        return [round(float(v) / base * 100, 2) for v in tail["close"]], [str(d.date()) for d in tail["date"]]

    s1, x1 = rebased(df1)
    s2, x2 = rebased(df2)
    n = min(len(s1), len(s2))
    charts: list[Spec] = [
        {
            "kind": "line",
            "title": f"Kinerja relatif {days} hari terakhir (awal = 100)",
            "caption": "Kedua harga disamakan titik awalnya (100) supaya bisa dibandingkan adil walau harga aslinya jauh berbeda. Garis yang berakhir lebih tinggi = kinerja lebih baik di periode ini.",
            "x": x1[-n:],
            "series": [
                {"label": t1, "values": s1[-n:], "tone": "ink"},
                {"label": t2, "values": s2[-n:], "tone": "mid"},
            ],
        },
        {
            "kind": "grouped",
            "title": "Perubahan harga",
            "caption": "Hijau/merah menunjukkan naik/turun pada tiap saham; urutan batang per baris = urutan legenda.",
            "signed": True,
            "cap": 150,
            "unit": "%",
            "series": [{"label": t1, "tone": "ink"}, {"label": t2, "tone": "mid"}],
            "items": [
                {"label": "30 hari", "values": [round(chg1, 2), round(chg2, 2)]},
                {"label": "~1 tahun", "values": [round(yr1, 2), round(yr2, 2)]},
            ],
        },
    ]
    if vol1 is not None and vol2 is not None:
        charts.append(
            {
                "kind": "grouped",
                "title": "Risiko: volatilitas tahunan",
                "caption": "Makin panjang batang, makin liar pergerakan harga historisnya (risiko lebih besar).",
                "unit": "%",
                "cap": 150,
                "series": [{"label": t1, "tone": "ink"}, {"label": t2, "tone": "mid"}],
                "items": [{"label": "Volatilitas", "values": [round(vol1 * 100, 1), round(vol2 * 100, 1)]}],
            }
        )
    return charts


# ---------------------------------------------------------------- renderer

def render(spec: Spec):
    """Spesifikasi -> matplotlib Figure. Dipanggil dari app.py."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = spec["kind"]
    if kind == "prob":
        fig, ax = plt.subplots(figsize=(6.4, 1.1))
        p = spec["p_up"]
        ax.barh([0], [p], color=TONES["pos"], height=0.5)
        ax.barh([0], [1 - p], left=[p], color=TONES["neg"], height=0.5)
        ax.axvline(0.5, color=_TEXT, linewidth=1)
        ax.text(0, -0.55, f"NAIK {p*100:.1f}%", color=TONES["pos"], fontsize=9, fontweight="bold", va="top")
        ax.text(1, -0.55, f"TURUN {(1-p)*100:.1f}%", color=TONES["neg"], fontsize=9, fontweight="bold", va="top", ha="right")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.0, 0.45)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    elif kind == "line":
        fig, ax = plt.subplots(figsize=(6.4, 3.3))
        xs = range(len(spec["x"]))
        for s in spec["series"]:
            vals = [float("nan") if v is None else v for v in s["values"]]
            ax.plot(xs, vals, label=s["label"], color=TONES[s["tone"]], linewidth=2.0 if s["tone"] == "ink" else 1.4)
        step = max(1, len(spec["x"]) // 5)
        ax.set_xticks(list(xs)[::step])
        ax.set_xticklabels([spec["x"][i][5:] for i in list(xs)[::step]], fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(spec["series"]))
        _style_axes(ax, grid_axis="y")
    else:  # bars / grouped
        items = spec["items"]
        grouped = kind == "grouped"
        n_series = len(spec["series"]) if grouped else 1
        fig, ax = plt.subplots(figsize=(6.4, 1.5 + 0.42 * len(items) * n_series))
        signed = spec.get("signed", False)
        values_all = [v for it in items for v in (it["values"] if grouped else [it["value"]])]
        cap = spec.get("cap")
        domain = spec.get("domain")
        m = max(abs(v) for v in values_all) or 1
        if cap:
            m = min(m, cap)
        lo, hi = (-m, m) if signed else (domain if domain else (0, m))
        h = 0.8 / n_series
        for i, it in enumerate(items):
            vals = it["values"] if grouped else [it["value"]]
            for j, v in enumerate(vals):
                tone = (spec["series"][j]["tone"] if grouped else it["tone"])
                if grouped and signed:
                    tone = spec["series"][j]["tone"]
                y = i + (j - (n_series - 1) / 2) * h
                shown = max(min(v, hi), lo)
                ax.barh(y, shown, height=h * 0.85, color=TONES[tone])
                label = f"{v:+.2f}{spec.get('unit','')}" if signed else f"{v:.1f}{spec.get('unit','')}"
                if v != shown:
                    label += " (dipotong)"
                ax.text(hi + (hi - lo) * 0.02, y, label, va="center", fontsize=8, family="monospace", color=_TEXT)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([it["label"] for it in items], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(lo, hi)
        if signed:
            ax.axvline(0, color=_TEXT, linewidth=1)
        if spec.get("guide"):
            g = spec["guide"]
            ax.axvline(g["value"], color=TONES["mid"], linestyle="--", linewidth=1)
        if grouped:
            from matplotlib.patches import Patch

            ax.legend(
                handles=[Patch(color=TONES[s["tone"]], label=s["label"]) for s in spec["series"]],
                frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, -0.08), ncol=n_series,
            )
        _style_axes(ax, grid_axis="x")

    fig.patch.set_facecolor(_PAPER)
    ax.set_title(spec["title"], loc="left", fontsize=10, color=_TEXT, pad=10)
    if kind in ("prob", "line"):
        fig.tight_layout()
    else:
        fig.subplots_adjust(top=0.84, left=0.26, right=0.80, bottom=0.22)
    return fig


def _style_axes(ax, grid_axis: str) -> None:
    ax.set_facecolor(_PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_HAIRLINE)
    ax.tick_params(colors=_TEXT, length=0, labelsize=8)
    ax.grid(axis=grid_axis, color=_HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
