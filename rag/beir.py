"""BEIR benchmark loading.

BEIR is the standard IR benchmark suite; SciFact is one of its smallest
datasets (5,183 scientific abstracts, 300 test queries) which makes it the
right size to embed on a laptop while still being real prose.

The reason to use it over synthetic data: bge-base-en-v1.5 has PUBLISHED
scores here. That turns your eval from "is this technique better?" into
"is my pipeline even correct?" -- if you land far from the published number,
you have a bug, not a finding. Very few learning setups give you that.

Two conventions matter for reproducing published numbers, and both are easy
to miss:

1. Documents are indexed as "title. text", not text alone. The title carries
   real signal in an abstract corpus and every published BEIR number includes it.

2. BGE retrieval models expect an INSTRUCTION PREFIX on queries (not on
   documents). Asymmetric search -- a short query and a long passage don't
   live in the same region of embedding space, and the prefix is how the
   model was trained to bridge that. Stage 3 measures what dropping it costs.
"""

import csv
import json
from pathlib import Path

from .corpus import Chunk
from .metrics import EvalQuery

# The prefix bge-*-en-v1.5 was trained with, for the QUERY side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_corpus(root: str = "data/beir/scifact", with_title: bool = True) -> list[Chunk]:
    docs = []
    with open(Path(root) / "corpus.jsonl", encoding="utf8") as f:
        for line in f:
            record = json.loads(line)
            title, text = record.get("title", "").strip(), record.get("text", "").strip()
            body = f"{title}. {text}" if (with_title and title) else text
            docs.append(Chunk(id=record["_id"], text=body, source=root,
                              meta={"title": title, "raw_text": text}))
    return docs


def load_queries(root: str = "data/beir/scifact", split: str = "test") -> list[EvalQuery]:
    """Only queries that appear in the split's qrels are scored -- queries.jsonl
    holds all 1,109, but the test split judges only 300 of them."""
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(root) / "qrels" / f"{split}.tsv", encoding="utf8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            grade = int(row["score"])
            if grade > 0:
                qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = grade

    texts = {}
    with open(Path(root) / "queries.jsonl", encoding="utf8") as f:
        for line in f:
            record = json.loads(line)
            texts[record["_id"]] = record["text"]

    return [
        EvalQuery(question=texts[qid], relevant=rel, qid=qid, variant=split)
        for qid, rel in sorted(qrels.items(), key=lambda kv: int(kv[0]))
        if qid in texts
    ]
