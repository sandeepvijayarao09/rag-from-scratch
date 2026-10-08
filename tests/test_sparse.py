import math

from rag.corpus import Chunk
from rag.sparse import BM25, tokenize


def docs(*texts):
    return [Chunk(id=str(i), text=t) for i, t in enumerate(texts)]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The BRCA1 gene, in mice!") == ["brca1", "gene", "mice"]
    assert tokenize("the cat", drop_stopwords=False) == ["the", "cat"]


def test_idf_formula_and_rare_terms_weigh_more():
    bm = BM25(docs("common rare", "common", "common", "common"))
    assert bm.idf["rare"] == math.log(1 + (4 - 1 + 0.5) / (1 + 0.5))
    assert bm.idf["rare"] > bm.idf["common"]


def test_rare_term_match_ranks_first():
    bm = BM25(docs("cell growth", "cell death", "cell telomerase growth"))
    top, _ = bm.search("telomerase cell", k=1)[0]
    assert top.id == "2"


def test_term_frequency_saturates():
    bm = BM25(docs("x " * 1, "x " * 2, "x " * 20, "y"), b=0.0)
    s = {c.id: score for c, score in bm.search("x", k=4)}
    gain_1_to_2 = s["1"] - s["0"]
    gain_2_to_20 = s["2"] - s["1"]
    # 18 extra occurrences buy less than the second one did, roughly.
    assert gain_2_to_20 < 2 * gain_1_to_2
    assert s["2"] < bm.idf["x"] * (bm.k1 + 1)  # upper bound of the tf term


def test_length_normalisation_prefers_shorter_doc():
    bm = BM25(docs("aspirin", "aspirin " + "filler " * 50))
    ranked = [c.id for c, _ in bm.search("aspirin", k=2)]
    assert ranked == ["0", "1"]


def test_unknown_terms_return_nothing():
    bm = BM25(docs("alpha", "beta"))
    assert bm.search("gamma") == []
    assert len(bm) == 2
