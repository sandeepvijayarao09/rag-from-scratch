"""Turn every v2 result file into one set of tested comparisons.

For each embedding model and stage, every config is compared to that stage's
control on the same queries (paired bootstrap, 10,000 resamples), with Holm's
correction applied within each stage. Then the ten headline techniques are
collected into one grid: technique x embedding model, as recommended and as
repaired, each labelled worse / better / no detectable difference.

Writes:
    results/v2/summary.json   every comparison, for the guide's explorer
    results/v2/RESULTS.md     the tables
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.v2.bench import compare, holm, verdict  # noqa: E402
from rag.v2.models import EMBEDDING_ORDER, RERANKER_ORDER  # noqa: E402

RESULTS = Path("results/v2")
CONTROL = {"s03_benchmark": "dense baseline", "s04_chunking": "whole documents",
           "s05_bm25": "bm25 default (k1=1.2, b=0.75)", "s06_hybrid": "dense",
           "s07_reranking": "dense (no rerank)", "s08_query_rewriting": "dense",
           "s09_contextual": "whole (600-doc subset)", "s10_late_interaction": "dense"}

# technique -> (stage, as-recommended config(s), repaired config). The first
# config listed is the one the v1 headline used.
HEADLINE = {
    "Chunking": ("s04_chunking", ["recursive 200/40", "sentence 100/1"],
                 "repaired: chunk long docs only, blended"),
    "RRF fusion": ("s06_hybrid", ["RRF k=60", "RRF k=10"], "repaired: weighted RRF (tuned)"),
    "Rerank after dense": ("s07_reranking", [f"dense -> {r} (replace)" for r in RERANKER_ORDER],
                           "dense -> qwen3-reranker-4b (interpolate, tuned)"),
    "HyDE": ("s08_query_rewriting", ["hyde (v1 form)"], "hyde (repaired, tuned)"),
    "Step-back": ("s08_query_rewriting", ["step_back (v1 form)"], "step_back (repaired, tuned)"),
    "Decomposition": ("s08_query_rewriting", ["decompose (v1 form)"], "decompose (repaired, tuned)"),
    "Query instruction": ("s03_benchmark", ["query format: alt"], None),
    "Multi-query": ("s08_query_rewriting", ["multi_query (v1 form)"], None),
    "Weighted hybrid": ("s06_hybrid", ["weighted 0.7/0.3"], "repaired: weighted fusion (tuned)"),
    "Rerank after BM25": ("s07_reranking", [f"bm25 -> {r} (replace)" for r in RERANKER_ORDER],
                          "bm25 -> qwen3-reranker-4b (interpolate, tuned)"),
}


def stage_comparisons(data: dict, stage: str, control_override: str | None = None) -> dict:
    ctrl = control_override or CONTROL[stage]
    if ctrl not in data:
        return {}
    rows, identical = {}, []
    for cfg, entry in data.items():
        if cfg == ctrl or cfg.endswith("[train]") or entry.get("n") != data[ctrl].get("n"):
            continue
        r = compare(data, cfg, ctrl)
        if r["better"] == 0 and r["worse"] == 0:
            identical.append(cfg)   # same top 10 as the control (e.g. the format Stage 3 chose)
            continue
        rows[cfg] = r
        rows[cfg]["meta"] = entry.get("meta", {})
    adj = holm({k: v["p"] for k, v in rows.items()}) if rows else {}
    for k, v in rows.items():
        v["p_holm"] = adj[k]
        v["verdict"] = verdict(v["delta"], *v["ci95"], v["p_holm"])
    return {"control": ctrl, "control_mean": compare(data, ctrl, ctrl)["control_mean"],
            "rows": rows, "identical": identical}


def main():
    summary = {}
    for emb in EMBEDDING_ORDER:
        d = RESULTS / emb
        if not d.exists():
            continue
        summary[emb] = {}
        for f in sorted(d.glob("s*.json")):
            stage = f.stem
            data = json.loads(f.read_text())
            summary[emb][stage] = stage_comparisons(data, stage)
            if stage == "s07_reranking":
                # BM25-first-stage rows are compared against BM25 alone as well.
                bm = {k: v for k, v in data.items() if k.startswith("bm25")}
                summary[emb]["s07_reranking_vs_bm25"] = stage_comparisons(bm, stage, "bm25 (no rerank)")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1))
    write_markdown(summary)


def cell(summary, emb, stage, cfg):
    st = summary.get(emb, {}).get(stage, {})
    row = st.get("rows", {}).get(cfg)
    if not row:
        # "=" ran but returned the control's top 10 on every query; "n/a" never ran
        # (e.g. the chunk repair on a model that truncates no documents).
        return "= 0" if cfg in st.get("identical", []) else "n/a"
    mark = {"worse": "▼", "better": "▲", "no detectable difference": "·"}[row["verdict"]]
    return f"{mark} {row['delta']:+.3f}"


def write_markdown(summary):
    lines = ["# v2 results", "",
             "Δ nDCG@10 against each stage's control, on the 300 SciFact test queries.",
             "▼ significantly worse · ▲ significantly better · · no detectable difference",
             "(paired bootstrap 95% CI, Holm-corrected within each stage). Repairs and any",
             "tuned setting were chosen on the train split only.",
             "= 0 same top 10 as the control on every query · n/a not applicable (the chunk",
             "repair only runs on models that truncate documents; only bge-base does, 2.8%).", ""]
    lines += ["## The ten techniques, by embedding model", "",
              "| Technique | Form | " + " | ".join(EMBEDDING_ORDER) + " |",
              "|---|---|" + "---|" * len(EMBEDDING_ORDER)]
    for tech, (stage, cfgs, repaired) in HEADLINE.items():
        stage_key = "s07_reranking_vs_bm25" if tech == "Rerank after BM25" else stage
        for cfg in cfgs:
            lines.append(f"| {tech} | {cfg} | " +
                         " | ".join(cell(summary, e, stage_key, cfg) for e in EMBEDDING_ORDER) + " |")
        if repaired:
            lines.append(f"| {tech} | **{repaired}** | " +
                         " | ".join(cell(summary, e, stage_key, repaired) for e in EMBEDDING_ORDER) + " |")
    for emb, stages in summary.items():
        lines += ["", f"## {emb}", ""]
        for stage, st in stages.items():
            if not st:
                continue
            lines += [f"### {stage} (control: {st['control']}, {st['control_mean']:.4f})", "",
                      "| Config | nDCG@10 | Δ | 95% CI | p (Holm) | +/- queries | Verdict |",
                      "|---|---|---|---|---|---|---|"]
            for cfg, r in st["rows"].items():
                lo, hi = r["ci95"]
                lines.append(f"| {cfg} | {r['treatment_mean']:.4f} | {r['delta']:+.4f} | "
                             f"[{lo:+.4f}, {hi:+.4f}] | {r['p_holm']:.4f} | "
                             f"{r['better']}/{r['worse']} | {r['verdict']} |")
            lines.append("")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
