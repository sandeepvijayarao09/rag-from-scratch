"""Cached, deterministic generation for v2.

v1 sampled its query rewrites at temperature 0.3 to 0.7, so re-running a stage
gave different rewrites and slightly different numbers. v2 uses temperature 0
and a fixed seed, and caches every call on (model, prompt), so a rerun
reproduces the exact same text and costs nothing.

The cache is one JSON file per model, written every few calls, so a long run
can be interrupted and resumed without losing work.
"""

import hashlib
import json
from pathlib import Path

import ollama

CACHE_DIR = Path("cache/v2/llm")


class LLM:
    def __init__(self, model: str, save_every: int = 5):
        self.model = model
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / (model.replace(":", "-").replace("/", "_") + ".json")
        self.cache: dict[str, str] = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.save_every, self._dirty = save_every, 0
        self.think_flag = True

    def _key(self, prompt: str, schema) -> str:
        tag = json.dumps(schema.model_json_schema(), sort_keys=True) if schema else ""
        return hashlib.sha256(f"{self.model}\x00{tag}\x00{prompt}".encode()).hexdigest()

    def has(self, prompt: str, schema=None) -> bool:
        return self._key(prompt, schema) in self.cache

    def __call__(self, prompt: str, schema=None, max_tokens: int = 512) -> str:
        key = self._key(prompt, schema)
        if key not in self.cache:
            kwargs = {"format": schema.model_json_schema()} if schema else {}
            if self.think_flag:
                kwargs["think"] = False      # no hidden reasoning before a one-line rewrite
            try:
                out = ollama.chat(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.0, "seed": 0, "num_predict": max_tokens}, **kwargs)
            except ollama.ResponseError as e:
                if "think" not in str(e).lower() or not self.think_flag:
                    raise
                self.think_flag = False      # this model has no thinking mode; drop the flag
                return self(prompt, schema, max_tokens)
            self.cache[key] = out["message"]["content"].strip()
            self._dirty += 1
            if self._dirty >= self.save_every:
                self.save()
        return self.cache[key]

    def save(self) -> None:
        if self._dirty:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(self.path)       # atomic, so an interrupted write never corrupts it
            self._dirty = 0
