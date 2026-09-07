# Execution plan: finishing stages 9, 12, 13, 14

Status: 10 of 15 stages measured. This is the plan for the remaining four
measurable stages, then publishing.

## What went wrong twice, and the fix

I killed two CRAG runs. The first because I misread `ps` output, the second
deliberately. Both times **all completed work was lost** and the job had to
restart from zero.

That is the real defect, not the parallelism. Stage 2's eval-set generator
survived being killed by SIGPIPE because it checkpoints every 10 items and
resumes. None of the remaining scripts do that.

**So: before running any of the four stages below, add a resumable disk cache
keyed on (stage, query id).** A killed job then costs only the item in flight.
This is the same content-addressed-cache pattern already used for embeddings
and cross-encoder scores, applied one layer up.

## Operating rules for the rest

1. **One job at a time.** Confirm zero python workers before starting the next.
2. **Never pipe a job through `tail` or `head`.** They buffer until the process
   exits, which hides all progress and makes a working job look hung. Write to
   a file, read the file.
3. **Never identify a process by script path.** `ps | grep myenv/bin/python`
   and `pgrep -f` both match the shell wrapper, whose command line contains the
   same string. Check for real work with CPU time and RSS instead.
4. **Checkpoint before running.** See above.
5. **Benchmark before queuing.** Measured rates on this machine:
   `gemma4:e4b` ~8s/call, `bge-base` ~62 docs/sec, `bge-reranker-base` 47.6
   pairs/sec, `MiniLM-L6` 264 pairs/sec.
6. **Smoke test at n=2-4 first.** Caught real bugs in stages 10, 11 and 12
   before they cost anything.

---

## Step 1 — Stage 12: self-correction

**Measures.** Whether CRAG grading detects retrieval failure, framed as a
detection problem: split queries by whether the gold document is actually in
the top-k, then check if the grader can tell the two groups apart. Plus the
Self-RAG router on a mix of lookup and no-lookup queries.

**Cost.** 24 queries x 3 documents = 72 grading calls, plus 16 router calls.
88 calls at ~8s is about 12 minutes.

**Expected, written before the run.** Two smoke tests at n=4 both showed the
grader flagging every *successful* retrieval as bad while catching both
failures. If that holds, "100% failure detection" is an artifact of a grader
that always says bad, and the honest reading is that it does not discriminate
at all. The router scored 1.00 on both query types, so the contrast is that
routing is easy and grading is hard, both for the same reason Stage 11 found.

**Falsification.** If success detection comes back near 1.00 at n=16, the small
samples were unlucky and CRAG works fine here.

---

## Step 2 — Stage 9: context enrichment

**Measures.** Contextual retrieval. Prepend an LLM-written situating blurb to
each chunk before embedding, then compare retrieval against the same chunks
unmodified.

**Cost.** One LLM call *per chunk*, which is the expensive shape. Needs sizing
first: benchmark `llama3.2:1b` against `gemma4:e4b` on the blurb task, because
this technique is normally run with a small fast model for exactly this reason.
If the 1B model is adequate, 60 documents chunked at size 100 is roughly 170
chunks and 5-10 minutes. If not, same corpus with gemma4 is about 23 minutes.

**Expected.** Stage 4 showed SciFact abstracts are short and self-contained, so
there is little lost context to restore. Expect a small or zero effect here and
a large one on long multi-topic documents. Worth stating that this is close to
the worst case for the technique.

---

## Step 3 — Stage 13: agentic retrieval

**Measures.** Multi-hop loop: retrieve, ask what is still missing, retrieve
again. The *trace* is the real output, since it shows the query the system
invented at each hop.

**Cost.** 20 queries at up to 3 hops, one call per hop plus a final decision.
Roughly 80 calls, about 11 minutes.

**Expected.** SciFact claims are single-hop by construction: one abstract
settles one claim. Multi-hop should not help, and if it appears to I should
distrust the eval rather than celebrate. The point is to demonstrate the
mechanism and its cost, and to check whether hop 2 restates hop 1 (a common
failure that burns a call retrieving the same documents).

---

## Step 4 — Stage 14: GraphRAG

**Measures.** Entity and relation extraction, label-propagation community
detection, community summaries. Graph statistics and the summaries themselves,
not nDCG.

**Cost.** One call per document for extraction plus one per community for
summarisation. 100 documents is about 13 minutes, plus ~15 communities at 2
minutes.

**Expected.** SciFact's eval is claim-to-abstract with document-level
judgements, which is a purely local task. GraphRAG's strength is global
questions and **will not show up in nDCG@10 at all**. Reporting a retrieval
score here would be meaningless, so the deliverable is the graph structure and
sample summaries, plus an explicit note that this benchmark cannot evaluate the
technique.

---

## Step 5 — Stage 15: no compute

Stays a documented roadmap. Measuring ANN recall loss and quantization
tradeoffs honestly needs a corpus one to two orders of magnitude larger than
SciFact. Writing that up as if 5,183 documents demonstrated it would be worse
than saying it is out of scope.

---

## Step 6 — Publish

1. Fill in the tutorial attribution URL in `concepts/01_naive_rag/README.md`
   (needs the human; I will not invent a citation)
2. Update the README results table and the ladder statuses
3. `gh repo create rag-from-scratch --public --source . --push`
4. Verify the clone-and-run path works from scratch

## Totals

About 50-60 minutes of LLM compute, strictly sequential, all of it resumable
after the checkpointing change. Nothing runs longer than ~15 minutes without
writing progress to disk.
