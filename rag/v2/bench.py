"""The v2 harness: one place that loads data, embeds with the right format,
retrieves, records per-query scores and runs the statistics.

Three rules every v2 stage follows, enforced here rather than by convention:

1. Per-query scores are always saved. v1 only kept means, so nothing could be
   significance-tested after the fact. Every config's nDCG@10, recall@10, MRR
   and top-10 list are written to results/v2/<embedding>/<stage>.json.

2. Anything tunable is tuned on the TRAIN split (809 queries) and scored once on
   the TEST split (300). v1 picked its own baseline on the test queries.

3. Every comparison is paired (same queries, both arms) and gets a bootstrap
   95% interval. A stage reports Holm-corrected p-values across its family.
"""

import json
import random
from pathlib import Path

import numpy as np

from ..beir import load_corpus, load_queries
from ..embed import Embedder
from ..metrics import _dcg
from ..sparse import BM25
from ..store import VectorStore, normalize
from .models import EMBEDDINGS

RESULTS = Path("results/v2")
B, SEED = 10_000, 0


# -- metrics ----------------------------------------------------------------

def ndcg10(ranked, relevant) -> float:
    grades = [relevant.get(str(d), 0) for d in ranked[:10]]
    idcg = _dcg(sorted(relevant.values(), reverse=True)[:10])
    return _dcg(grades) / idcg if idcg else 0.0


def recall10(ranked, relevant) -> float:
    return sum(1 for d in ranked[:10] if str(d) in relevant) / len(relevant)


def mrr10(ranked, relevant) -> float:
    return next((1.0 / (i + 1) for i, d in enumerate(ranked[:10]) if str(d) in relevant), 0.0)


# -- statistics -------------------------------------------------------------

def paired_bootstrap(diff: np.ndarray, seed: int = SEED):
    """95% interval on the mean paired difference and a two-sided p-value."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(B, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float(np.mean(np.abs(means - diff.mean()) >= abs(diff.mean())))
    return float(lo), float(hi), max(p, 1 / B)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    running, out = 0.0, {}
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[name]))
        out[name] = running
    return out


def verdict(delta, lo, hi, p) -> str:
    if p < 0.05 and hi < 0:
        return "worse"
    if p < 0.05 and lo > 0:
        return "better"
    return "no detectable difference"


# -- the bench --------------------------------------------------------------

class Bench:
    """Everything one embedding model needs. Build once per process."""

    def __init__(self, embedding: str, verbose: bool = True):
        self.spec = EMBEDDINGS[embedding]
        self.name = embedding
        self.docs = load_corpus(with_title=True)
        self.by_id = {d.id: d for d in self.docs}
        self.test = load_queries(split="test")
        self.train = load_queries(split="train")
        self.embedder = Embedder(model=self.spec.ollama, cache_dir=f"cache/v2/emb/{embedding}")
        if verbose:
            print(f"[{embedding}] embedding {len(self.docs)} documents", flush=True)
        vecs = self.embedder.embed([self.doc_text(d) for d in self.docs], progress=verbose)
        self.store = VectorStore(self.docs, vecs)
        self._bm25 = None
        self.out = RESULTS / embedding
        self.out.mkdir(parents=True, exist_ok=True)
        # Stage 3 picks the query format on TRAIN and writes it here; every
        # later stage uses that choice as its dense baseline.
        choice = self.out / "query_format.json"
        self.qfmt = json.loads(choice.read_text())["format"] if choice.exists() else self.spec.query

    # formatting, straight from each model card
    def doc_text(self, doc, text: str | None = None) -> str:
        title = doc.meta.get("title") or "none"
        return self.spec.doc.format(title=title, text=text if text is not None
                                    else doc.meta.get("raw_text", doc.text))

    def query_text(self, claim: str, fmt: str | None = None) -> str:
        return (fmt or self.qfmt).format(q=claim)

    def qvec(self, claim: str, fmt: str | None = None) -> np.ndarray:
        return normalize(self.embedder.embed([self.query_text(claim, fmt)])[0])

    def qvecs(self, claims: list[str], fmt: str | None = None) -> np.ndarray:
        v = self.embedder.embed([self.query_text(c, fmt) for c in claims])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def dense_vec(self, vec, k: int = 100) -> list[tuple[str, float]]:
        return [(c.doc_id, s) for c, s in self.store.search(vec, k=k)]

    def dense(self, claim: str, k: int = 100, fmt: str | None = None):
        return self.dense_vec(self.qvec(claim, fmt), k)

    def all_scores(self, vec) -> np.ndarray:
        """Cosine score for every document, in self.docs order."""
        return self.store.vectors @ vec

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25(self.docs)
        return self._bm25

    def sparse(self, claim: str, k: int = 100):
        return [(c.doc_id, s) for c, s in self.bm25.search(claim, k=k)]

    def queries(self, split: str):
        return self.test if split == "test" else self.train

    # recording
    def _path(self, stage: str) -> Path:
        return self.out / f"{stage}.json"

    def load(self, stage: str) -> dict:
        p = self._path(stage)
        return json.loads(p.read_text()) if p.exists() else {}

    def record(self, stage: str, config: str, ranked: dict[str, list[str]],
               split: str = "test", meta: dict | None = None, quiet: bool = False,
               queries=None) -> float:
        qs = self.queries(split) if queries is None else queries
        if split == "train":
            config = f"{config} [train]"   # never overwrite a test entry
        entry = {
            "split": split, "n": len(qs), "meta": meta or {},
            "ndcg10": {q.qid: ndcg10(ranked[q.qid], q.relevant) for q in qs},
            "recall10": {q.qid: recall10(ranked[q.qid], q.relevant) for q in qs},
            "mrr10": {q.qid: mrr10(ranked[q.qid], q.relevant) for q in qs},
        }
        if split == "test":
            entry["top10"] = {q.qid: [str(d) for d in ranked[q.qid][:10]] for q in qs}
        data = self.load(stage)
        data[config] = entry
        tmp = self._path(stage).with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._path(stage))
        m = float(np.mean(list(entry["ndcg10"].values())))
        if not quiet:
            r = float(np.mean(list(entry["recall10"].values())))
            print(f"  {config:<44} nDCG@10 {m:.4f}  R@10 {r:.4f}  ({split}, n={len(qs)})",
                  flush=True)
        return m

    def mean(self, ranked: dict[str, list[str]], split: str) -> float:
        return float(np.mean([ndcg10(ranked[q.qid], q.relevant) for q in self.queries(split)]))

    def tune(self, rank_fn, grid, split: str = "train", queries=None):
        """Pick the grid value with the best train nDCG. rank_fn(param, qs) -> ranked."""
        qs = queries or self.queries(split)
        best = None
        for p in grid:
            ranked = rank_fn(p, qs)
            m = float(np.mean([ndcg10(ranked[q.qid], q.relevant) for q in qs]))
            if best is None or m > best[1]:
                best = (p, m)
        return best


def compare(stage_data: dict, treat: str, ctrl: str, metric: str = "ndcg10") -> dict:
    a, b = stage_data[treat][metric], stage_data[ctrl][metric]
    qids = sorted(a)
    diff = np.array([a[q] - b[q] for q in qids])
    lo, hi, p = paired_bootstrap(diff)
    return {"treatment": treat, "control": ctrl, "n": len(qids),
            "treatment_mean": float(np.mean([a[q] for q in qids])),
            "control_mean": float(np.mean([b[q] for q in qids])),
            "delta": float(diff.mean()), "ci95": [lo, hi], "p": p,
            "better": int((diff > 1e-9).sum()), "worse": int((diff < -1e-9).sum())}


def subsample(queries, n: int, seed: int = 0):
    qs = list(queries)
    random.Random(seed).shuffle(qs)
    return sorted(qs[:n], key=lambda q: int(q.qid))
