"""Stage 4a: why this corpus needs chunking at all.

bge-base-en-v1.5 accepts 512 tokens. Longer input is not rejected -- it is
SILENTLY TRUNCATED. You get a vector back, it looks fine, and it was built
from only the first part of your document. No error, no warning.

This is the honest, measurable motivation for chunking, as opposed to the
usual hand-wave about "semantic coherence". Run it before choosing a strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag import load_corpus, load_queries  # noqa: E402
from rag.chunking import BGE_MAX_TOKENS, TOKENS_PER_WORD  # noqa: E402

if __name__ == "__main__":
    docs = load_corpus()
    lengths = sorted(len(d.text.split()) for d in docs)
    est_tokens = [round(n * TOKENS_PER_WORD) for n in lengths]

    pct = lambda p: lengths[int(len(lengths) * p) - 1]
    print(f"SciFact: {len(docs)} documents\n")
    print(f"  words   min={lengths[0]:<5} p50={pct(.50):<5} p90={pct(.90):<5} "
          f"p99={pct(.99):<5} max={lengths[-1]}")
    print(f"  ~tokens (x{TOKENS_PER_WORD})            p90={round(pct(.90)*TOKENS_PER_WORD):<5} "
          f"p99={round(pct(.99)*TOKENS_PER_WORD):<5} max={round(lengths[-1]*TOKENS_PER_WORD)}")

    over = sum(1 for t in est_tokens if t > BGE_MAX_TOKENS)
    print(f"\n  over the {BGE_MAX_TOKENS}-token limit: {over}/{len(docs)} "
          f"({100*over/len(docs):.1f}%) -- silently truncated")

    if over:
        lost = sum(max(0, t - BGE_MAX_TOKENS) for t in est_tokens)
        print(f"  estimated tokens never embedded: {lost:,} "
              f"({100*lost/sum(est_tokens):.1f}% of the corpus)")

    queries = load_queries()
    qlen = sorted(len(q.question.split()) for q in queries)
    print(f"\n  queries: p50={qlen[len(qlen)//2]} words, max={qlen[-1]} words")
    print("\n  Note the asymmetry: ~10-word claims searching ~250-word abstracts.\n"
          "  That length mismatch is exactly what the BGE query prefix exists to bridge.")
