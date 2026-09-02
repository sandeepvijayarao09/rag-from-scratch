"""Corpus loading.

Document ids are strings everywhere. Stage 2's cat-facts used ints, but BEIR
corpus ids look like "31715818", and having one id type across the whole repo
is worth more than the mild ugliness of str("0").
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text.

    `id`     -- unique per chunk, what retrieval returns and eval scores against
    `doc_id` -- the document this chunk came from. Identical to `id` when a
                document is not split. Stage 4 needs these to diverge: BEIR's
                relevance judgements are at DOCUMENT level, so a chunked
                retriever has to map its hits back up to parent documents
                before it can be scored.
    """
    id: str
    text: str
    doc_id: str = ""
    source: str = ""
    meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not self.doc_id:
            object.__setattr__(self, "doc_id", self.id)


def load_cat_facts(path: str = "cat-facts.txt") -> list[Chunk]:
    """Stage 0/1 corpus: one self-contained fact per line, so no chunking needed."""
    with open(path, encoding="utf8") as f:
        lines = [line.strip() for line in f]
    return [
        Chunk(id=str(i), text=line, source=path)
        for i, line in enumerate(l for l in lines if l)
    ]
