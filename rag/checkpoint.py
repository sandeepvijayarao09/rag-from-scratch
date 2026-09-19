"""Resumable progress for long LLM-bound jobs.

Written after killing two CRAG runs and losing all completed work both times.
The eval-set generator in Stage 2 survived the same treatment because it saved
every 10 items; nothing else did.

The pattern is the one already used for embeddings and cross-encoder scores:
the result of an LLM call is a pure function of its inputs, so key it and keep
it. The only difference here is granularity. This checkpoints at the level of
"query N of stage X is done" rather than at the level of a single model call.

Usage:

    ck = Checkpoint("crag")
    for q in queries:
        if (cached := ck.get(q.qid)) is not None:
            verdict = cached
        else:
            verdict = expensive_llm_work(q)
            ck.set(q.qid, verdict)
    ck.save()

Interrupting mid-run costs you the item in flight and nothing else.
"""

import json
from pathlib import Path


class Checkpoint:
    def __init__(self, name: str, dirname: str = "cache", save_every: int = 5):
        self.path = Path(dirname) / f"ck_{name}.json"
        self.path.parent.mkdir(exist_ok=True)
        self.save_every = save_every
        self._pending = 0
        self.data: dict = (
            json.loads(self.path.read_text()) if self.path.exists() else {})

    def get(self, key):
        return self.data.get(str(key))

    def set(self, key, value) -> None:
        self.data[str(key)] = value
        self._pending += 1
        if self._pending >= self.save_every:
            self.save()

    def save(self) -> None:
        # Write to a temp file and rename, so a kill during the write cannot
        # leave a truncated JSON file behind and destroy the whole run.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        tmp.replace(self.path)
        self._pending = 0

    def __len__(self) -> int:
        return len(self.data)

    def resumed(self) -> str:
        return f"resumed with {len(self.data)} cached" if self.data else "starting fresh"
