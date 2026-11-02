"""Draw docs/results.svg: the tested v2 results on bge-base as a forest plot.

No plotting dependency, just SVG text, so it runs anywhere. Every number is read
from results/v2/summary.json (built by v2/report.py): the delta in nDCG@10, its
paired-bootstrap 95% CI, and the Holm-corrected verdict. bge-base is the model
v1 used, so these rows are the v1 claims re-tested.

    python scripts/plot_results.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EMB = "bge-base"

# (label, stage key in summary.json, config). Each delta is against dense
# bge-base (0.759), except the BM25 rows, which are against BM25 alone (0.665).
# For chunking and RRF the worst of the v1 variants is shown.
ROWS = [
    ("Query decomposition", "s08_query_rewriting", "decompose (v1 form)"),
    ("HyDE", "s08_query_rewriting", "hyde (v1 form)"),
    ("Step-back prompting", "s08_query_rewriting", "step_back (v1 form)"),
    ("Rerank on dense, bge-reranker-base", "s07_reranking", "dense -> bge-reranker-base (replace)"),
    ("RRF fusion, k=60 (worst of 3)", "s06_hybrid", "RRF k=60"),
    ("Chunking, sentence 200 (worst of 6)", "s04_chunking", "sentence 200/1"),
    ("BGE query prefix", "s03_benchmark", "query format: alt"),
    ("Late interaction, ColBERT", "s10_late_interaction", "colbert (full corpus)"),
    ("Contextual retrieval (600 docs)", "s09_contextual", "contextualised (gemma4:12b blurbs)"),
    ("Multi-query + RRF", "s08_query_rewriting", "multi_query (v1 form)"),
    ("Weighted hybrid 0.7/0.3", "s06_hybrid", "weighted 0.7/0.3"),
    ("Rerank on dense, Qwen3-Reranker-4B", "s07_reranking", "dense -> qwen3-reranker-4b (replace)"),
    ("Rerank on BM25, bge-reranker-base", "s07_reranking_vs_bm25", "bm25 -> bge-reranker-base (replace)"),
    ("Rerank on BM25, Qwen3-Reranker-4B", "s07_reranking_vs_bm25", "bm25 -> qwen3-reranker-4b (replace)"),
]

W, ROW_H, TOP, LEFT, RIGHT = 800, 32, 104, 290, 40
LO, HI = -0.13, 0.15
NEG, POS, INK, MUTED, GRID = "#c2410c", "#15803d", "#1f2937", "#6b7280", "#e5e7eb"
COLOR = {"worse": NEG, "better": POS, "no detectable difference": MUTED}


def x(v: float) -> float:
    return LEFT + (v - LO) / (HI - LO) * (W - LEFT - RIGHT)


def main() -> None:
    summary = json.loads((REPO / "results" / "v2" / "summary.json").read_text())[EMB]
    rows = [(label, stage, summary[stage]["rows"][cfg]) for label, stage, cfg in ROWS]
    H = TOP + ROW_H * len(rows) + 46
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="17" font-weight="600" fill="{INK}">'
        "Change in nDCG@10 from adding each technique, BEIR SciFact, bge-base</text>",
        f'<text x="24" y="50" font-size="12.5" fill="{MUTED}">'
        "Dot: mean change on 300 test queries. Line: paired-bootstrap 95% CI. Colour: "
        "Holm-corrected verdict within each stage.</text>",
        f'<text x="24" y="68" font-size="12.5" fill="{MUTED}">'
        "Against dense bge-base (0.759), except the BM25 rows (against BM25 alone, 0.665).</text>",
        f'<text x="24" y="86" font-size="12.5"><tspan fill="{NEG}">&#9679; significantly worse</tspan>'
        f'<tspan dx="14" fill="{MUTED}">&#9679; no detectable difference</tspan>'
        f'<tspan dx="14" fill="{POS}">&#9679; significantly better</tspan></text>',
    ]
    for t in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15):
        stroke = MUTED if t == 0 else GRID
        out.append(f'<line x1="{x(t):.1f}" y1="{TOP - 6}" x2="{x(t):.1f}" '
                   f'y2="{TOP + ROW_H * len(rows)}" stroke="{stroke}"/>')
        out.append(f'<text x="{x(t):.1f}" y="{TOP + ROW_H * len(rows) + 18}" font-size="11.5" '
                   f'fill="{MUTED}" text-anchor="middle">{t:+.2f}</text>')
    for i, (label, stage, r) in enumerate(rows):
        y = TOP + i * ROW_H + ROW_H / 2
        lo, hi = r["ci95"]
        color = COLOR[r["verdict"]]
        out.append(f'<line x1="{x(lo):.1f}" y1="{y:.1f}" x2="{x(hi):.1f}" y2="{y:.1f}" '
                   f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(r["delta"]):.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        out.append(f'<text x="{LEFT - 12}" y="{y + 4.5:.1f}" font-size="13" fill="{INK}" '
                   f'text-anchor="end">{label} '
                   f'<tspan fill="{MUTED}" font-size="11.5">S{stage[1:3]}</tspan></text>')
        out.append(f'<text x="{x(hi) + 8:.1f}" y="{y + 4.5:.1f}" font-size="11.5" '
                   f'fill="{INK}">{r["delta"]:+.3f}</text>')
    out.append(f'<text x="24" y="{H - 10}" font-size="11" fill="{MUTED}">'
               "github.com/sandeepvijayarao09/rag-from-scratch · scripts/plot_results.py</text>")
    out.append("</svg>")
    path = REPO / "docs" / "results.svg"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
