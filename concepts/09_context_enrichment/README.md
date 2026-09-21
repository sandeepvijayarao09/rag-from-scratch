# Stage 9 - Context enrichment

**Tier: Advanced** · prerequisite: Stage 8

Stage 4 showed chunking costs you context. Contextual retrieval (Anthropic,
2024) is the fix: before embedding, ask an LLM to write a short blurb situating
each chunk inside its parent document, and prepend it.

The problem it targets is real and unfixable by any retrieval technique:

> "The company's revenue grew by 3% over the previous quarter."

Which company? Which quarter? Unfindable by any query naming either, because
neither appears in the chunk. The information is not in the indexed text.

## The model choice is the whole implementation

One LLM call per chunk is the expensive shape, so the generator matters more
here than anywhere else in this repo. I benchmarked before running:

```
Llama-3.2-1B     0.4s/chunk
gemma4:e4b       9.2s/chunk      23x slower, comparable blurbs
```

At 9.2s the full corpus is 11 hours. At 0.4s it is minutes. Most write-ups of
this technique mention prompt caching and skip the model-size decision, which
is the larger lever by a wide margin. The 1B model's blurbs were if anything
more specific, naming "preterm infants compared to full-term infants" where the
larger model wrote "a study using diffusion tensor MRI".

## Results

600 document subset (every gold document plus distractors), 300 queries,
doc-level max pooling.

```
arm                         units   nDCG@10   vs whole     R@10      MRR
whole (no chunking)           600    0.8961              0.9593   0.8769
chunked 100/20               1784    0.8966    +0.0005   0.9660   0.8765
contextualised               1784    0.8898    -0.0063   0.9527   0.8715
```

**Contextualising made it worse**, both against plain chunks (-0.0068) and
against not chunking at all (-0.0063).

## Why, and when it would not

I predicted a small or zero effect, because Stage 4 established that SciFact
abstracts are short (p50 = 204 words), single-topic, and only 2.8% exceed the
encoder window. There was very little lost context to restore.

The result was worse than "no effect", and the likely mechanism is dilution.
The blurb adds roughly 50 words of generic framing to a 100-word chunk. On a
corpus where chunks are already self-describing, that framing carries no new
retrievable signal and shrinks the share of the embedding held by the chunk's
distinctive terms. You are paying an LLM call per chunk to make each vector
slightly blurrier.

The sharper statement of the technique, then, is this. **Contextual retrieval
restores context that chunking destroyed.** If chunking did not destroy any,
there is nothing to restore and the blurb is noise.

Where it should pay: long multi-topic documents where chunks genuinely lose
their referents. Contracts, manuals, transcripts, filings. SciFact abstracts
are the opposite of all three, which makes this close to a worst case rather
than a fair test of the idea.

## What is worth copying regardless

It is an **index-time** cost, paid once. Compare Stage 8's query rewriting,
which pays an LLM call on every query forever. When contextual retrieval does
apply, its cost model is far friendlier than it first appears, and prompt
caching plus a small model makes it cheaper still.

Blurbs are cached to disk by (model, document, chunk), so an interrupted run
resumes and re-running is free.
