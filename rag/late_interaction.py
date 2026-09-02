"""Late interaction / multi-vector retrieval (ColBERT-style).

THE SPECTRUM, which is the useful framing:

  Bi-encoder      one vector per document. All interaction happens AFTER
  (Stages 2-6)    encoding, via a single dot product. Cheap, precomputable,
                  lossy -- a whole abstract compressed to 768 numbers.

  Cross-encoder   no precomputation at all. Query and document go through the
  (Stage 7)       transformer together. Accurate, and O(candidates) forward
                  passes per query, so it only works as a second stage.

  Late            one vector per TOKEN, precomputed. Interaction is deferred
  interaction     to query time but stays cheap, because it is just similarity
  (this file)     between stored vectors -- no transformer runs at query time.

    MaxSim(q, d) = SUM over query tokens qi:  MAX over doc tokens dj: qi . dj

  Each query token finds its single best match anywhere in the document, and
  those maxima are summed. A rare term that appears once in a long abstract
  still contributes fully, which is exactly what mean-pooling destroys.

THE COST, and it is the reason this is a specialist tool:

  bi-encoder:       1 vector  x 768 dims x 4 bytes =    3 KB per document
  late interaction: ~250 vectors x 768 x 4 bytes   =  768 KB per document

  For SciFact's 5,183 abstracts that is 16 MB versus about 4 GB. A 250x index
  blowup. Real ColBERTv2 attacks this with aggressive residual quantization
  (~2 bits per dimension) plus centroid pruning, getting it to roughly 20x.

CAVEAT, stated plainly: this uses token embeddings from `bge-base`, which was
trained for SINGLE-VECTOR retrieval with CLS pooling. Real ColBERT checkpoints
are trained end-to-end with the MaxSim objective, so their token vectors are
shaped for exactly this operation. Expect the mechanism to work and the numbers
to underperform a trained ColBERT. That gap is itself the lesson: an
architecture is not separable from the objective it was trained under.
"""

import numpy as np


class LateInteractionIndex:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str | None = None, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device, self.max_length = device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.doc_ids: list[str] = []
        self.doc_vectors: list[np.ndarray] = []   # (n_tokens, dim) per document

    def _encode(self, texts: list[str]) -> list[np.ndarray]:
        import torch

        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state   # (B, T, D)
            hidden = torch.nn.functional.normalize(hidden, dim=-1)

        mask = batch["attention_mask"].bool()
        # Drop padding: keep only real tokens per sequence.
        return [hidden[i][mask[i]].cpu().numpy().astype(np.float32)
                for i in range(len(texts))]

    def add(self, chunks, batch_size: int = 16, progress: bool = False) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            for chunk, vectors in zip(batch, self._encode([c.text for c in batch])):
                self.doc_ids.append(chunk.doc_id)
                self.doc_vectors.append(vectors)
            if progress and (i // batch_size) % 10 == 0:
                print(f"\r  encoding {i + len(batch)}/{len(chunks)}", end="", flush=True)
        if progress:
            print()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = self._encode([query])[0]                    # (Tq, D)
        # MaxSim: for each query token take its best match in the document,
        # then sum. (Tq,D) @ (D,Td) -> (Tq,Td), max over doc axis, sum over query.
        scores = [float((q @ d.T).max(axis=1).sum()) for d in self.doc_vectors]
        order = np.argsort(-np.asarray(scores))[:k]
        return [(self.doc_ids[i], scores[i]) for i in order]

    def index_bytes(self) -> int:
        return sum(v.nbytes for v in self.doc_vectors)
