"""Late interaction with a model trained for it.

v1's Stage 10 ran MaxSim over raw bge-base token embeddings. That model was
trained to produce one pooled vector, not token vectors that match each other,
so v1 measured MaxSim on vectors that were never meant for it, and on an
800-document subset because 768-dim float32 token vectors are huge.

v2 uses GTE-ModernColBERT, trained specifically for MaxSim, with 128-dim
token vectors. Stored as float16 that is roughly 40x smaller per token, which
is what lets the whole 5,183-document corpus fit on a laptop.
"""

from pathlib import Path

import numpy as np

from .models import LATE_INTERACTION_MODEL

CACHE = Path("cache/v2/colbert")


class ColbertIndex:
    def __init__(self, model_name: str = LATE_INTERACTION_MODEL):
        from pylate import models
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = models.ColBERT(model_name_or_path=model_name, device=device)
        self.doc_ids: list[str] = []
        self.vectors: list[np.ndarray] = []

    def build(self, docs, batch_size: int = 32) -> None:
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / "index.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            self.doc_ids, self.vectors = list(data["ids"]), list(data["vecs"])
            return
        embs = self.model.encode([d.text for d in docs], batch_size=batch_size,
                                 is_query=False, show_progress_bar=True)
        self.doc_ids = [d.id for d in docs]
        self.vectors = [np.asarray(e, dtype=np.float16) for e in embs]
        np.savez(path, ids=np.array(self.doc_ids, dtype=object),
                 vecs=np.array(self.vectors, dtype=object))

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        q = np.asarray(self.model.encode([query], is_query=True,
                                         show_progress_bar=False)[0], dtype=np.float32)
        scores = np.array([float((q @ d.astype(np.float32).T).max(axis=1).sum())
                           for d in self.vectors])
        top = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def index_bytes(self) -> int:
        return int(sum(v.nbytes for v in self.vectors))
