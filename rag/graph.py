"""GraphRAG: retrieval over a knowledge graph instead of a flat chunk list.

WHAT IT SOLVES, and it is genuinely different from everything else here.

Vector retrieval answers LOCAL questions: "what does document X say about Y?"
It finds the passage that best matches the query. It structurally cannot answer
GLOBAL questions:

    "What are the main themes in this corpus?"
    "How do these findings relate to each other?"
    "Which topics does this collection cover most?"

No single passage contains that answer, so no top-k over passages can retrieve
it. The answer is a property of the collection, not of any member of it.

THE PIPELINE:

  1. EXTRACT   LLM pulls (entity, relation, entity) triples from each document.
  2. BUILD     Entities become nodes, relations become edges. Entities
               mentioned in many documents become hubs -- the graph encodes
               cross-document structure that a chunk list throws away.
  3. CLUSTER   Detect communities of densely connected entities. Communities
               correspond to topics, discovered rather than declared.
  4. SUMMARISE LLM writes a summary per community. THESE are what get
               retrieved for global questions.
  5. QUERY     Local questions -> match entities, walk the neighbourhood.
               Global questions -> retrieve community summaries.

Community detection here is LABEL PROPAGATION, implemented directly rather than
pulled from a library: every node starts in its own community, then repeatedly
adopts whichever community is most common among its neighbours. It converges
fast, needs no parameters, and is about 20 lines. Microsoft's GraphRAG uses
Leiden, which is better but needs a dependency and obscures the idea.

THE COST is the same shape as contextual retrieval and worse: at least one LLM
call per document for extraction, plus one per community for summarisation.
This is the most expensive system in the repo. Run it on a corpus SUBSET and be
honest that you did.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import ollama
from pydantic import BaseModel

GRAPH_MODEL = "gemma4:e4b"


class Triple(BaseModel):
    subject: str
    relation: str
    object: str


class Extraction(BaseModel):
    triples: list[Triple]


EXTRACT_PROMPT = """Extract the key factual relationships from this scientific text.

<text>
{text}
</text>

Return up to 6 triples of (subject, relation, object). Use the specific
technical entities named in the text (genes, proteins, drugs, diseases,
methods) as subjects and objects, not vague words. Keep relations short verb
phrases like "increases", "inhibits", "is associated with".

Return JSON only."""

SUMMARY_PROMPT = """These entities and relationships form one cluster in a
knowledge graph built from scientific abstracts:

{facts}

Write a 2-3 sentence summary of what this cluster is ABOUT: the topic it covers
and the main relationships within it. This summary will be retrieved to answer
broad questions about the corpus, so name the key entities explicitly."""


class KnowledgeGraph:
    def __init__(self, model: str = GRAPH_MODEL, cache_dir: str = "cache"):
        self.model = model
        self.path = Path(cache_dir) / "graph.json"
        self.path.parent.mkdir(exist_ok=True)
        self.triples: list[tuple[str, str, str, str]] = []   # (subj, rel, obj, doc_id)
        self.communities: dict[str, int] = {}                # entity -> community id
        self.summaries: dict[int, str] = {}
        if self.path.exists():
            self._load()

    # ---------- 1. extract ----------
    def extract(self, chunks, progress: bool = False, save_every: int = 20) -> None:
        done = {t[3] for t in self.triples}
        todo = [c for c in chunks if c.doc_id not in done]

        for i, chunk in enumerate(todo, 1):
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user",
                               "content": EXTRACT_PROMPT.format(text=chunk.text[:3000])}],
                    format=Extraction.model_json_schema(),
                    options={"temperature": 0.0},
                )
                for t in Extraction.model_validate_json(
                        response["message"]["content"]).triples:
                    s, o = t.subject.strip().lower(), t.object.strip().lower()
                    if s and o and s != o:
                        self.triples.append((s, t.relation.strip().lower(), o, chunk.doc_id))
            except Exception:
                pass
            if i % save_every == 0:
                self.save()
                if progress:
                    print(f"\r  extracted {i}/{len(todo)} docs, "
                          f"{len(self.triples)} triples", end="", flush=True)
        self.save()
        if progress:
            print()

    # ---------- 2/3. build + cluster ----------
    def adjacency(self) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = defaultdict(set)
        for s, _, o, _ in self.triples:
            adj[s].add(o)
            adj[o].add(s)
        return adj

    def detect_communities(self, iterations: int = 20, seed: int = 0) -> dict[str, int]:
        """Label propagation. Each node repeatedly adopts the most common label
        among its neighbours; ties broken randomly. Converges in a few passes."""
        adj = self.adjacency()
        labels = {node: i for i, node in enumerate(adj)}
        rng = random.Random(seed)
        nodes = list(adj)

        for _ in range(iterations):
            rng.shuffle(nodes)
            changed = 0
            for node in nodes:
                if not adj[node]:
                    continue
                counts = Counter(labels[n] for n in adj[node])
                best = max(counts.values())
                winner = rng.choice([l for l, c in counts.items() if c == best])
                if labels[node] != winner:
                    labels[node] = winner
                    changed += 1
            if changed == 0:
                break

        # Renumber to a dense 0..n-1 range for readability.
        remap = {old: new for new, old in enumerate(sorted(set(labels.values())))}
        self.communities = {node: remap[l] for node, l in labels.items()}
        return self.communities

    def community_facts(self, community_id: int, limit: int = 25) -> list[str]:
        members = {e for e, c in self.communities.items() if c == community_id}
        return [f"{s} --{r}--> {o}" for s, r, o, _ in self.triples
                if s in members or o in members][:limit]

    # ---------- 4. summarise ----------
    def summarize(self, min_size: int = 3, progress: bool = False) -> dict[int, str]:
        sizes = Counter(self.communities.values())
        targets = [c for c, n in sizes.items() if n >= min_size and c not in self.summaries]

        for i, community_id in enumerate(targets, 1):
            facts = "\n".join(self.community_facts(community_id))
            try:
                self.summaries[community_id] = ollama.chat(
                    model=self.model,
                    messages=[{"role": "user",
                               "content": SUMMARY_PROMPT.format(facts=facts)}],
                    options={"temperature": 0.2},
                )["message"]["content"].strip()
            except Exception:
                continue
            if progress:
                print(f"\r  summarized {i}/{len(targets)} communities", end="", flush=True)
        self.save()
        if progress:
            print()
        return self.summaries

    # ---------- 5. query ----------
    def local_search(self, entities: list[str], hops: int = 1) -> list[str]:
        """Walk the neighbourhood of matched entities. Answers 'what connects to X'."""
        adj = self.adjacency()
        frontier = {e.lower() for e in entities if e.lower() in adj}
        seen = set(frontier)
        for _ in range(hops):
            frontier = {n for e in frontier for n in adj[e]} - seen
            seen |= frontier
        return [f"{s} --{r}--> {o}" for s, r, o, _ in self.triples
                if s in seen and o in seen]

    def docs_for_entities(self, entities: list[str]) -> list[str]:
        wanted = {e.lower() for e in entities}
        return list({doc for s, _, o, doc in self.triples if s in wanted or o in wanted})

    # ---------- persistence ----------
    def save(self) -> None:
        self.path.write_text(json.dumps({
            "triples": self.triples,
            "communities": self.communities,
            "summaries": {str(k): v for k, v in self.summaries.items()},
        }))

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self.triples = [tuple(t) for t in data.get("triples", [])]
        self.communities = data.get("communities", {})
        self.summaries = {int(k): v for k, v in data.get("summaries", {}).items()}

    def stats(self) -> dict:
        adj = self.adjacency()
        sizes = Counter(self.communities.values())
        return {
            "triples": len(self.triples),
            "entities": len(adj),
            "communities": len(sizes),
            "largest_community": max(sizes.values()) if sizes else 0,
            "summarized": len(self.summaries),
        }
