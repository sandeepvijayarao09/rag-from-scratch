# Stage 11 — Grounded generation

**Tier: Advanced** · prerequisite: Stage 10

Every stage before this measured retrieval. This one measures what the model
does with what retrieval handed it, and it needs a separate scoreboard because
the two fail independently. An answer can be perfectly faithful to documents
that were the wrong ones.

## The test

Three context conditions per claim:

| Condition | Context | Correct behaviour |
|---|---|---|
| gold | the document SciFact labels as relevant | give a verdict |
| distractor | top-ranked documents with the gold one removed | say NOT_ENOUGH |
| empty | nothing | say NOT_ENOUGH |

The distractor condition is the one that matters. It is topically plausible and
does not settle the claim. A system that returns confident verdicts there fails
invisibly, because no retrieval metric in this repo can see it.

## Results

```
context         gave verdict   NOT_ENOUGH            correct
gold                      12           18    12/30 = 0.40  (want verdict)
distractor                 5           25    25/30 = 0.83  (want not_enough)
empty                      0           30    30/30 = 1.00  (want not_enough)

citation present on gold verdicts: 12/12 = 1.00
```

n=30, seed 0, `gemma4:e4b`.

## The part that took three attempts

**11a.** I first framed this as question answering: answer the claim from
context, abstain if the context is insufficient. Gold scored 0.33. The model
abstained on twenty of thirty documents that were labelled relevant, which
looked like catastrophic over-refusal.

Before writing that up I printed the actual outputs. One of the abstentions was
the claim "high cardiopulmonary fitness causes increased mortality rate", whose
gold document *refutes* it. Under a question-answering prompt a refuting
document does not answer anything, so INSUFFICIENT_CONTEXT was defensible. My
eval could not represent the dataset: SciFact is claim verification, and gold
documents may support or refute.

**11b.** Rewrote it as the three-way task the corpus actually encodes
(SUPPORTED / REFUTED / NOT_ENOUGH). Gold went 0.33 to 0.40. Better, and nowhere
near fixed, so my diagnosis was partly right and mostly wrong.

**11c.** Two candidates left. Either the prompt was too strict (11b told the
model "topically related is NOT the same as settling the claim" at temperature
0), or the model cannot make the inference. Those are distinguishable, so I ran
the same claims and same gold documents under a relaxed prompt:

```
prompt             verdict rate on gold
strict (11b)     12/30 =           0.40
relaxed          14/30 =           0.47
```

Prompt work bought 0.07, then 0.07 again, then stopped. Roughly 0.53 of gold
cases still get no verdict.

## What this actually measures

The generator is the ceiling here, not the retrieval and not the prompt.
SciFact claims are technical inferences from scientific abstracts. Connecting
"cells lacking clpC have a defect in sporulation efficiency in Bacillus
subtilis" to a high-throughput genetic screen is expert reading, and
`gemma4:e4b` mostly will not commit to it.

That is worth stating precisely, because it decomposes a number people usually
report as one thing. A RAG system with *perfect* retrieval and this generator
would still fail on more than half of SciFact. Retrieval work cannot fix that
and no amount of prompt engineering moved it much either. The fix is a better
generator, or a domain-tuned one.

This is Method Rule 2 in the README paying off. If I had only measured
end-to-end accuracy I would have seen a bad number and started tuning
retrieval, which was never the problem.

## The failure mode was the opposite of the warning

Every RAG guide warns that models answer confidently from irrelevant context.
Here the model returned NOT_ENOUGH on 83% of distractor contexts and 100% of
empty ones, and every verdict it did give on gold context carried a citation.

It is not reckless. It is *too* conservative, and the cost shows up as useful
answers withheld rather than as hallucinations. Both are failures; they need
opposite fixes; and only the second one gets written about.

I would not generalise this past `gemma4:e4b` on a biomedical corpus. A larger
model, or a corpus where the inference step is shorter, could easily invert it.

## Files

| File | |
|---|---|
| `01_citation_abstention.py` | The first, wrongly framed attempt. Kept deliberately. |
| `02_claim_verification.py` | Three-way SUPPORTED / REFUTED / NOT_ENOUGH |
| `03_strictness_ablation.py` | Strict vs relaxed prompt on gold context |

`01` is still here because the sequence is the useful part. Deleting a flawed
first attempt is how you end up with a repo where everything appears to have
worked immediately.
