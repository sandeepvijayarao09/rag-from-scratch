"""Batched, cached embedding.

Two changes from Stage 1's naive_rag.py:

1. Batching. `ollama.embed` accepts a list, so 149 chunks can go in a handful
   of round-trips instead of 149. The model runs the same number of forward
   passes; you're just not paying HTTP + scheduling overhead per item.

2. A content-addressed disk cache. Embeddings are a pure function of
   (model, text), so re-running an experiment should cost zero inference.
   This is what makes Stages 5-8 tolerable -- you'll re-index dozens of times
   while comparing retrieval techniques, and only *new* text should cost you.
"""

import hashlib
from pathlib import Path

import numpy as np
import ollama

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL, cache_dir: str = "cache"):
        self.model = model
        self.cache_path = Path(cache_dir) / "embeddings.npz"
        self.cache_path.parent.mkdir(exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            with np.load(self.cache_path) as data:
                self._cache = {k: data[k] for k in data.files}

    def embed(self, texts: list[str], batch_size: int = 32,
              prefix: str = "", progress: bool = False) -> np.ndarray:
        """Return an (N, D) float32 array, one row per input text.

        `prefix` is prepended to every input. BGE retrieval models are
        ASYMMETRIC: they were trained with an instruction prefix on queries
        and none on documents, because a short question and a long passage
        do not naturally land in the same region of embedding space.
        The prefix is part of the cache key, so both variants coexist.
        """
        texts = [prefix + t for t in texts]
        missing = [t for t in dict.fromkeys(texts) if _key(self.model, t) not in self._cache]

        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            if progress:
                print(f"\r  embedding {i + len(batch)}/{len(missing)}", end="", flush=True)
            vectors = ollama.embed(model=self.model, input=batch)["embeddings"]
            for text, vector in zip(batch, vectors):
                self._cache[_key(self.model, text)] = np.asarray(vector, dtype=np.float32)

        if missing:
            if progress:
                print()
            self._save()

        return np.stack([self._cache[_key(self.model, t)] for t in texts])

    def _save(self) -> None:
        np.savez(self.cache_path, **self._cache)
