"""Draw docs/results.svg: the README results table as a diverging bar chart.

No plotting dependency, just SVG text, so it runs anywhere. The numbers are the
full-corpus deltas from the README table (each one is measured in the stage
README named next to it). Ranges are drawn as a bar from best to worst variant.

    python scripts/plot_results.py
"""

from pathlib import Path

# (label, stage, low, high): delta nDCG@10 against the pipeline the technique was
# added to. That is dense bge-base (0.759) for every row except reranking on
# BM25, which is measured against BM25 alone (0.665 -> 0.706).
ROWS = [
    ("Query decomposition", "08", -0.095, -0.095),
    ("Step-back prompting", "08", -0.066, -0.066),
    ("HyDE", "08", -0.048, -0.048),
    ("Cross-encoder rerank on dense", "07", -0.033, -0.033),
    ("BGE query prefix", "03", -0.014, -0.014),
    ("RRF fusion (4 variants)", "06", -0.024, -0.009),
    ("Chunking (6 configs)", "04", -0.020, -0.010),
    ("Multi-query + RRF", "08", 0.001, 0.001),
    ("Weighted hybrid 0.7/0.3", "06", 0.010, 0.010),
    ("Cross-encoder rerank on BM25", "07", 0.041, 0.041),
]

W, ROW_H, TOP, LEFT, RIGHT = 760, 34, 84, 250, 40
LO, HI = -0.12, 0.05
H = TOP + ROW_H * len(ROWS) + 46
NEG, POS, INK, MUTED, GRID = "#c2410c", "#15803d", "#1f2937", "#6b7280", "#e5e7eb"


def x(v: float) -> float:
    return LEFT + (v - LO) / (HI - LO) * (W - LEFT - RIGHT)


def main() -> None:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="17" font-weight="600" fill="{INK}">'
        "Change in nDCG@10 from adding each technique, BEIR SciFact</text>",
        f'<text x="24" y="50" font-size="12.5" fill="{MUTED}">'
        "Each delta is against dense bge-base (0.759), except rerank on BM25, "
        "which is against BM25 alone (0.665).</text>",
        f'<text x="24" y="68" font-size="12.5" fill="{MUTED}">'
        "300 test queries; the Stage 8 rows use a paired 50-query subsample.</text>",
    ]
    for t in (-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05):
        stroke = MUTED if t == 0 else GRID
        out.append(f'<line x1="{x(t):.1f}" y1="{TOP - 6}" x2="{x(t):.1f}" '
                   f'y2="{TOP + ROW_H * len(ROWS)}" stroke="{stroke}"/>')
        out.append(f'<text x="{x(t):.1f}" y="{TOP + ROW_H * len(ROWS) + 18}" font-size="11.5" '
                   f'fill="{MUTED}" text-anchor="middle">{t:+.3f}</text>')
    for i, (label, stage, low, high) in enumerate(ROWS):
        y = TOP + i * ROW_H
        a, b = sorted((low, high))
        x0, x1 = x(min(a, 0.0)), x(max(b, 0.0))
        color = MUTED if abs(high) < 0.005 else POS if high > 0 else NEG
        if low != high:  # a range across variants: solid best-to-worst, faded to zero
            out.append(f'<rect x="{x(b):.1f}" y="{y + 7}" width="{x(0) - x(b):.1f}" '
                       f'height="{ROW_H - 14}" fill="{color}" opacity="0.35" rx="2"/>')
            x0, x1 = x(a), x(b)
        out.append(f'<rect x="{x0:.1f}" y="{y + 7}" width="{x1 - x0:.1f}" '
                   f'height="{ROW_H - 14}" fill="{color}" rx="2"/>')
        out.append(f'<text x="{LEFT - 12}" y="{y + ROW_H / 2 + 4.5}" font-size="13" '
                   f'fill="{INK}" text-anchor="end">{label} '
                   f'<tspan fill="{MUTED}" font-size="11.5">S{stage}</tspan></text>')
        value = f"{low:+.3f}" if low == high else f"{low:+.3f} to {high:+.3f}"
        tx, anchor = (x(b) + 6, "start") if high > 0 else (x(a) - 6, "end")
        out.append(f'<text x="{tx:.1f}" y="{y + ROW_H / 2 + 4.5}" font-size="11.5" '
                   f'fill="{INK}" text-anchor="{anchor}">{value}</text>')
    out.append(f'<text x="24" y="{H - 10}" font-size="11" fill="{MUTED}">'
               "github.com/sandeepvijayarao09/rag-from-scratch · scripts/plot_results.py</text>")
    out.append("</svg>")
    path = Path(__file__).resolve().parents[1] / "docs" / "results.svg"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
