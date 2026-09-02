"""BM25 -- lexical retrieval, from scratch.

No neural network. BM25 scores a document by how often the query's terms appear
in it, weighted by how RARE each term is across the corpus, and damped so that
a term appearing 50 times is not 50x better than appearing once.

    score(q, d) = SUM over terms t in q:

                                    tf(t,d) * (k1 + 1)
        idf(t) * ------------------------------------------------
                 tf(t,d) + k1 * (1 - b + b * len(d) / avg_doc_len)

Three ideas, each worth understanding separately:

  idf     -- ln(1 + (N - df + 0.5)/(df + 0.5)). A term in 5 of 5000 documents
             is far more informative than one in 4000. This is exactly what
             dense embeddings are bad at: a rare identifier, error code, gene
             name or proper noun carries enormous signal, and a 768-dim vector
             trained on general text has no particular reason to preserve it.

  k1=1.2  -- saturation. tf/(tf + k1) is concave, so the 10th occurrence of a
             term adds much less than the 2nd. Without this, keyword spam wins.

  b=0.75  -- length normalisation. Long documents contain more of everything,
             so divide by length -- but only partially (b=1 is full
             normalisation, b=0 is none). 0.75 is the Lucene default.

Implementation note: this uses an INVERTED INDEX (term -> postings list), not a
dense term-document matrix. A 5,183 x 30,000 dense matrix would be 622MB of
mostly zeros; the postings list touches only the documents that actually
contain a query term, which is a handful. This is why lexical search scaled to
the web decades before anyone had a GPU.
"""

import math
import re
from collections import defaultdict

from .corpus import Chunk

# A tiny stoplist. These are in nearly every document, so their idf is near
# zero and they contribute almost nothing -- dropping them is a speed
# optimisation more than a quality one.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "this", "these", "those", "which",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS] if drop_stopwords else tokens


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []

        for idx, chunk in enumerate(chunks):
            counts: dict[str, int] = defaultdict(int)
            for token in tokenize(chunk.text):
                counts[token] += 1
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))
            self.doc_len.append(sum(counts.values()))

        self.n = len(chunks)
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

        # idf is a pure function of document frequency, so precompute it once.
        self.idf = {
            term: math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            for term, post in self.postings.items()
        }

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, tf in postings:
                norm = 1 - self.b + self.b * self.doc_len[idx] / self.avgdl
                scores[idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], s) for i, s in top]

    def __len__(self) -> int:
        return self.n
