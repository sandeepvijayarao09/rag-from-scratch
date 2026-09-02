"""Chunking strategies.

Sizes are measured in WORDS, not tokens. That is a deliberate simplification
(no tokenizer dependency) but you must know the conversion: English runs about
1.3 BERT tokens per word, and bge-base-en-v1.5 has a 512-token limit. So a
~390-word chunk is the practical ceiling. Anything longer is SILENTLY TRUNCATED
by the encoder -- no error, no warning, just a vector built from a prefix of
your text. `03_truncation_check.py` measures how much of SciFact this affects,
which is the real motivation for chunking a corpus like this.

The strategies, in increasing order of respect for the text:

  whole     -- no splitting. The baseline.
  fixed     -- N words, sliding with overlap. Ignores all structure; will cut
               a sentence, a clause, or a number in half without noticing.
  sentence  -- pack whole sentences up to a budget. Never splits mid-sentence,
               but ignores paragraph structure.
  recursive -- try paragraph breaks first, then sentences, then words; only
               descend when a piece is still too big. This is what
               LangChain's RecursiveCharacterTextSplitter does, and it is the
               sensible default for most prose.

OVERLAP exists because a chunk boundary can orphan the context a sentence
needs to be interpretable ("This effect was not observed in the control
group" -- which effect?). Overlap buys that back at the cost of index size
and duplicate hits.
"""

import re

from .corpus import Chunk

TOKENS_PER_WORD = 1.3   # rough English estimate for BERT-family tokenizers
BGE_MAX_TOKENS = 512


def _words(text: str) -> list[str]:
    return text.split()


def _sentences(text: str) -> list[str]:
    # Good enough for prose; a real system would use a proper segmenter.
    # Avoids splitting on common abbreviations and decimals.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_whole(text: str, size: int = 0, overlap: int = 0) -> list[str]:
    return [text]


def chunk_fixed(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = _words(text)
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out = [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    # The final window can be a tiny sliver already covered by its predecessor.
    if len(out) > 1 and len(_words(out[-1])) < overlap:
        out.pop()
    return out


def chunk_sentence(text: str, size: int = 200, overlap: int = 1) -> list[str]:
    """Pack whole sentences up to `size` words; carry `overlap` sentences over."""
    sentences = _sentences(text)
    if not sentences:
        return [text]

    chunks, current, count = [], [], 0
    for sentence in sentences:
        n = len(_words(sentence))
        if current and count + n > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            count = sum(len(_words(s)) for s in current)
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_recursive(text: str, size: int = 200, overlap: int = 40,
                    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")) -> list[str]:
    """Split on the most semantic separator that works, descending only as needed."""
    if len(_words(text)) <= size:
        return [text]

    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        pieces, out, current = text.split(sep), [], ""
        for piece in pieces:
            candidate = f"{current}{sep}{piece}" if current else piece
            if len(_words(candidate)) <= size:
                current = candidate
            else:
                if current:
                    out.append(current)
                # Still too big on its own? descend to the next separator.
                current = piece if len(_words(piece)) <= size else ""
                if not current:
                    out.extend(chunk_recursive(piece, size, overlap, separators[i + 1:]))
        if current:
            out.append(current)
        return [c for c in out if c.strip()]

    return chunk_fixed(text, size, overlap)


STRATEGIES = {
    "whole": chunk_whole,
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "recursive": chunk_recursive,
}


def apply(docs: list[Chunk], strategy: str, size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split each document, preserving `doc_id` so hits can be scored at doc level."""
    split = STRATEGIES[strategy]
    out = []
    for doc in docs:
        for i, piece in enumerate(split(doc.text, size, overlap)):
            out.append(Chunk(id=f"{doc.id}#{i}", text=piece, doc_id=doc.id,
                             source=doc.source, meta=doc.meta))
    return out
