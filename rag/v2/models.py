"""The v2 model ladder.

v1 used one model per slot: bge-base for embeddings, bge-reranker-base for
reranking, gemma4:e4b for every generation call. Most of v1's findings come
down to one fact, which is that bge-base is already very good on SciFact, so
v2 varies each slot instead of trusting one point.

Embeddings: three models, deliberately weaker -> current -> stronger, so each
technique can be scored against a baseline with more or less headroom.

Rerankers: the v1 model plus two stronger ones, to test Stage 7's rule that a
reranker only helps when it is better than the first stage.

Generators: the v1 model and a 12B model from the same family, so the
difference between them is size and nothing else.

Every query and document format below is copied from the model's own card.
Getting these wrong silently costs points, which is the lesson of Stage 3.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str            # short label used in result paths and tables
    ollama: str          # Ollama model tag
    query: str           # recommended query format, {q} is the claim
    query_alt: str       # the ablation Stage 3 tests against it
    doc: str             # document format, {title} and {text}
    note: str


EMBEDDINGS = {
    "embeddinggemma-300m": EmbeddingSpec(
        name="embeddinggemma-300m", ollama="embeddinggemma:300m",
        # Google's model card lists a "fact checking" prompt as well as the
        # generic retrieval one. SciFact is claim verification, so Stage 3
        # tests both.
        query="task: search result | query: {q}",
        query_alt="task: fact checking | query: {q}",
        doc="title: {title} | text: {text}",
        note="Weaker baseline: 300M parameters, more headroom for techniques."),
    "bge-base": EmbeddingSpec(
        name="bge-base", ollama="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        query="{q}",
        query_alt="Represent this sentence for searching relevant passages: {q}",
        doc="{title}. {text}",
        note="The v1 model, 110M parameters. v1 found the card's prefix hurt on SciFact."),
    "qwen3-embedding-4b": EmbeddingSpec(
        name="qwen3-embedding-4b", ollama="qwen3-embedding:4b",
        query="Instruct: Given a scientific claim, retrieve documents that support "
              "or refute the claim\nQuery:{q}",
        query_alt="Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery:{q}",
        doc="{title}. {text}",
        note="Stronger baseline, 4B parameters. Qwen recommends a task-specific instruction."),
}
EMBEDDING_ORDER = ["embeddinggemma-300m", "bge-base", "qwen3-embedding-4b"]


@dataclass(frozen=True)
class RerankerSpec:
    name: str
    hf: str
    kind: str            # "cross-encoder" or "qwen3"
    note: str


RERANKERS = {
    "bge-reranker-base": RerankerSpec("bge-reranker-base", "BAAI/bge-reranker-base",
                                      "cross-encoder", "The v1 reranker, 278M."),
    "bge-reranker-v2-m3": RerankerSpec("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3",
                                       "cross-encoder", "568M, a clear step up."),
    "qwen3-reranker-4b": RerankerSpec("qwen3-reranker-4b", "Qwen/Qwen3-Reranker-4B",
                                      "qwen3", "4B LLM-based reranker, yes/no logits."),
}
RERANKER_ORDER = ["bge-reranker-base", "bge-reranker-v2-m3", "qwen3-reranker-4b"]
QWEN3_RERANK_INSTRUCTION = ("Given a scientific claim, judge whether the document "
                            "supports or refutes it")

GENERATORS = {"small": "gemma4:e4b", "large": "gemma4:12b"}
GENERATOR = GENERATORS["large"]          # >10B for every v2 generation call

LATE_INTERACTION_MODEL = "lightonai/GTE-ModernColBERT-v1"
