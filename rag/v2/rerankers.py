"""Rerankers for v2, all behind one interface: score(query, texts) -> floats.

Two kinds:
  cross-encoder  bge-reranker-base and bge-reranker-v2-m3. A classifier head
                 over [query, document], via sentence-transformers.
  qwen3          Qwen3-Reranker-4B is a causal LM asked "does this document
                 meet the requirement? yes/no". The score is P(yes) from the
                 logits at the final position. The prompt template is the one
                 on the model card; it is not interchangeable with the others.

Scores are cached to disk on (model, query, document), because reranking is the
slowest non-LLM step and every stage reuses the same candidate sets.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .models import QWEN3_RERANK_INSTRUCTION, RERANKERS

CACHE_DIR = Path("cache/v2/rerank")


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Reranker:
    def __init__(self, name: str, batch_size: int = 16):
        self.spec = RERANKERS[name]
        # The 4B LLM reranker needs far more memory per pair than the cross-encoders.
        self.batch_size = min(batch_size, 8) if self.spec.kind == "qwen3" else batch_size
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.json"
        self.cache: dict[str, float] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self._model = None

    def _key(self, q: str, d: str) -> str:
        return hashlib.sha256(f"{q}\x00{d}".encode()).hexdigest()[:32]

    def _load(self):
        if self._model is not None:
            return
        if self.spec.kind == "cross-encoder":
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.spec.hf, device=_device(), max_length=512)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.spec.hf, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(
                self.spec.hf, torch_dtype=torch.float16).to(_device()).eval()
            prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements "
                      "based on the Query and the Instruct provided. Note that the answer can "
                      "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self._model = dict(tok=tok, model=model,
                               pre=tok.encode(prefix, add_special_tokens=False),
                               suf=tok.encode(suffix, add_special_tokens=False),
                               yes=tok.convert_tokens_to_ids("yes"),
                               no=tok.convert_tokens_to_ids("no"))

    def _score_qwen3(self, pairs):
        import torch
        m = self._model
        body = [f"<Instruct>: {QWEN3_RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
                for q, d in pairs]
        enc = m["tok"](body, padding=False, truncation="longest_first",
                       max_length=1024 - len(m["pre"]) - len(m["suf"]),
                       return_attention_mask=False)
        ids = [m["pre"] + x + m["suf"] for x in enc["input_ids"]]
        batch = m["tok"].pad({"input_ids": ids}, padding=True, return_tensors="pt")
        batch = {k: v.to(m["model"].device) for k, v in batch.items()}
        with torch.no_grad():
            # Only the last position is needed; full logits are batch x length x 151k vocab.
            logits = m["model"](**batch, logits_to_keep=1).logits[:, -1, :]
        two = torch.stack([logits[:, m["no"]], logits[:, m["yes"]]], dim=1).float()
        return torch.softmax(two, dim=1)[:, 1].cpu().tolist()

    def score(self, query: str, texts: list[str]) -> list[float]:
        keys = [self._key(query, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            self._load()
            for s in range(0, len(todo), self.batch_size):
                idx = todo[s:s + self.batch_size]
                pairs = [(query, texts[i]) for i in idx]
                if self.spec.kind == "cross-encoder":
                    out = self._model.predict(pairs, batch_size=self.batch_size,
                                              show_progress_bar=False).tolist()
                else:
                    out = self._score_qwen3(pairs)
                for i, v in zip(idx, out):
                    self.cache[keys[i]] = float(v)
            self.save()
        return [self.cache[k] for k in keys]

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """`candidates` is [(doc_id, text)]; returns [(doc_id, score)] best first."""
        scores = self.score(query, [t for _, t in candidates])
        order = np.argsort(-np.asarray(scores))
        return [(candidates[i][0], scores[i]) for i in order]

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cache))
        tmp.replace(self.path)
