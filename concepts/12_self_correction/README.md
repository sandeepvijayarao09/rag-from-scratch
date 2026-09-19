# Stage 12 — Self-correction (CRAG, Self-RAG)

**Tier: Pro** · prerequisite: Stage 11

Every pipeline up to here runs identically for every query and has no way to
notice it has failed. If retrieval returns noise, generation answers from noise.
CRAG grades what came back; Self-RAG decides whether to retrieve at all.

Both add LLM calls to the query path, so they are only worth it if the
judgements are accurate. That is what this stage measures, and the two
mechanisms came out at opposite ends.

## CRAG, framed as detection

The useful question is not "can an LLM grade a document" but "does grading
catch the cases where retrieval missed?" So split the queries by ground truth
first, then grade both groups and see whether the grader can tell them apart.

```
ground truth             correct   ambiguous   incorrect     detected
retrieval SUCCEEDED            2           8           6        0.12  (want correct)
retrieval FAILED               1           0           7        0.88  (want flagged)

flag rate | retrieval SUCCEEDED    0.875
flag rate | retrieval FAILED       0.875
separation                         +0.000
balanced accuracy                  0.500
```

Read the headline number on its own and CRAG looks excellent: **0.88 failure
detection**. Read it next to the other row and it evaporates. The grader flags
87.5% of documents as unusable *whether or not retrieval actually worked*. The
separation is zero and balanced accuracy is exactly chance.

A grader that always says "bad" scores 1.00 on failure detection for free. This
one is not quite that, but statistically it may as well be. Acting on its
verdict would discard 14 of 16 good retrievals to catch 7 of 8 bad ones.

This is the same trap as Stage 11's faithfulness score, which read 1.00 only
because the model answered on a third of cases. **A conditional metric is
meaningless without its base rate.** Both stages produced an impressive-looking
number that dissolved the moment the other half was measured.

## Self-RAG routing, which works

```
query type                 routed correctly
SciFact claims           8/8 =      1.00  (want retrieve)
no-lookup prompts        8/8 =      1.00  (want skip)
```

Sixteen for sixteen. Given claims that need the corpus and prompts that do not
("what is 17 percent of 4,200", "convert this list into a markdown table"), the
router separates them perfectly.

## Why the same model aces one and fails the other

Routing asks: *is this the kind of question that needs a lookup?* That is
surface classification. The signal is in the form of the query and a 4B model
reads it easily.

Grading asks: *does this abstract settle this specific claim?* That is the
identical expert inference Stage 11 found `gemma4:e4b` could not do, where it
declined to give a verdict on 60% of correct documents. Two stages hit that
ceiling independently, by different routes.

The practical version: **spend your LLM calls where the task is classification,
not where it is expertise.** Routing is cheap and reliable. Relevance grading on
technical content needs a model that can actually read the technical content,
and if you had that model you would likely not need the grader.

## Caveats

n=24 for the CRAG split (16 succeeded, 8 failed), which is small. A separation of
exactly 0.000 does not prove the grader has literally zero signal; the
confidence interval at this sample size is wide. What it does rule out is the
*strong* signal CRAG needs in order to be worth an extra LLM call per document
on every query.

All of this is `gemma4:e4b` on biomedical abstracts. A larger model, or a corpus
where relevance is judgeable from surface features, would plausibly invert the
grading result. The routing result seems likely to hold anywhere.

## Files

`01_crag_detection.py` — both tests, checkpointed. Judgements are cached per
query, so an interrupted run resumes and re-running to change a printed metric
costs nothing.
