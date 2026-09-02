"""Contextual retrieval (Anthropic, 2024): fix chunks that lost their context.

THE PROBLEM. Split a document and each piece stops being self-describing:

    "The company's revenue grew by 3% over the previous quarter."

Which company? Which quarter? The chunk is unfindable by any query naming
either, because neither appears in it. Standard chunking creates this problem
by construction, and no retrieval technique can fix it -- the information is
simply not in the indexed text.

THE FIX. Before embedding, ask an LLM to write a short situating blurb using
the WHOLE document, and prepend it:

    "This chunk is from ACME Corp's Q2 2023 SEC filing; the previous quarter's
     revenue was $314M. The company's revenue grew by 3% over the previous
     quarter."

Now the chunk answers queries about ACME and about Q2 2023. Anthropic reported
this cutting retrieval failures substantially, and more when combined with BM25
(the blurb adds proper nouns, which is exactly what idf rewards).

THE COST, which is the entire catch. One LLM call PER CHUNK, at indexing time.
For SciFact's 5,183 abstracts at ~8s per local call that is over 11 hours. This
is the most expensive technique in the repo by a wide margin, and the reason it
is normally run with prompt caching against a fast hosted model, where the
document is cached once and each chunk costs only its own tokens.

Two consequences worth internalising:
  - It is an INDEX-time cost, not a query-time cost. You pay once. Compare that
    to query transformation, which pays on every single query forever.
  - It is the technique most improved by prompt caching, because the same long
    document is re-sent for every chunk within it.

Everything here is cached to disk, so an interrupted run resumes.
"""

import hashlib
import json
from pathlib import Path

import ollama

CONTEXT_MODEL = "gemma4:e4b"

PROMPT = """<document>
{document}
</document>

Here is a chunk taken from that document:

<chunk>
{chunk}
</chunk>

Write a short standalone blurb (one or two sentences, under 60 words) that
situates this chunk within the document: what it is about, what entities it
concerns, and anything the chunk refers to but does not name.

The blurb will be prepended to the chunk to make it searchable on its own.
Write ONLY the blurb, nothing else."""


class ContextualIndexer:
    def __init__(self, model: str = CONTEXT_MODEL, cache_dir: str = "cache"):
        self.model = model
        self.path = Path(cache_dir) / "contextual.json"
        self.path.parent.mkdir(exist_ok=True)
        self.cache: dict[str, str] = (
            json.loads(self.path.read_text()) if self.path.exists() else {})

    def _key(self, document: str, chunk: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{document}\x00{chunk}".encode()).hexdigest()

    def contextualize(self, document: str, chunk: str) -> str:
        key = self._key(document, chunk)
        if key in self.cache:
            return self.cache[key]
        try:
            blurb = ollama.chat(
                model=self.model,
                messages=[{"role": "user",
                           "content": PROMPT.format(document=document[:6000], chunk=chunk)}],
                options={"temperature": 0.0},
            )["message"]["content"].strip()
        except Exception:
            blurb = ""
        self.cache[key] = blurb
        return blurb

    def apply(self, chunks, documents: dict[str, str], progress: bool = False,
              save_every: int = 25):
        """Return new chunks with the situating blurb prepended to each text."""
        from .corpus import Chunk

        out = []
        for i, chunk in enumerate(chunks, 1):
            blurb = self.contextualize(documents.get(chunk.doc_id, chunk.text), chunk.text)
            text = f"{blurb}\n\n{chunk.text}" if blurb else chunk.text
            out.append(Chunk(id=chunk.id, text=text, doc_id=chunk.doc_id,
                             source=chunk.source, meta={**chunk.meta, "blurb": blurb}))
            if i % save_every == 0:
                self.save()
                if progress:
                    print(f"\r  contextualized {i}/{len(chunks)}", end="", flush=True)
        self.save()
        if progress:
            print()
        return out

    def save(self) -> None:
        self.path.write_text(json.dumps(self.cache))
