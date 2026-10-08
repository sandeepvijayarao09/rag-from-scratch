from rag.chunking import (
    apply,
    chunk_fixed,
    chunk_recursive,
    chunk_sentence,
    chunk_whole,
)
from rag.corpus import Chunk

WORDS = " ".join(f"w{i}" for i in range(100))


def test_whole_and_short_text_are_untouched():
    assert chunk_whole("anything") == ["anything"]
    assert chunk_fixed("a b c", size=10) == ["a b c"]
    assert chunk_recursive("a b c", size=10) == ["a b c"]


def test_fixed_windows_overlap_and_cover_everything():
    chunks = chunk_fixed(WORDS, size=40, overlap=10)
    assert all(len(c.split()) <= 40 for c in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(WORDS.split())


def test_fixed_drops_a_trailing_sliver():
    # 65 words, size 40, overlap 10 -> windows start at 0, 30, 60. The last one
    # is 5 words, shorter than the overlap and already covered, so it is dropped.
    text = " ".join(f"w{i}" for i in range(65))
    chunks = chunk_fixed(text, size=40, overlap=10)
    assert len(chunks) == 2
    assert chunks[-1].split()[-1] == "w64"


def test_sentence_never_splits_mid_sentence():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    chunks = chunk_sentence(text, size=6, overlap=0)
    assert chunks == ["One two three. Four five six.", "Seven eight nine. Ten eleven twelve."]


def test_sentence_overlap_carries_previous_sentence():
    text = "One two three. Four five six. Seven eight nine."
    chunks = chunk_sentence(text, size=6, overlap=1)
    assert chunks[1].startswith("Four five six.")


def test_recursive_prefers_paragraph_breaks():
    para = " ".join(["word"] * 30)
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = chunk_recursive(text, size=40)
    assert chunks == [para, para, para]


def test_apply_keeps_parent_doc_ids():
    docs = [Chunk(id="d1", text=WORDS), Chunk(id="d2", text="short")]
    out = apply(docs, "fixed", size=40, overlap=10)
    assert {c.doc_id for c in out} == {"d1", "d2"}
    assert out[0].id == "d1#0"
    assert [c.id for c in out if c.doc_id == "d2"] == ["d2#0"]
