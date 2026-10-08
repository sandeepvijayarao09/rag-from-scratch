#!/usr/bin/env bash
# Re-run the full-corpus retrieval baselines (stages 3, 5 and 6) and save every
# printed table to results/logs/, so the headline numbers can be checked with
# one command. Needs only requirements-core.txt and the bge embedding model in
# Ollama. Embeddings are cached under cache/, so a second run takes seconds.
#
# Stages 4 and 8 keep their measured numbers in results/*.json; rerun a slice
# with the stage script's own argument (see each stage README). Stages 7 and 10
# need requirements.txt (torch) and are run per stage.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
mkdir -p results/logs
./scripts/download_data.sh

run() {
  local name="$1"; shift
  echo "==> $name"
  # Show progress live; keep it out of the saved log.
  "$PY" "$@" 2>&1 | tee /dev/stderr | tr '\r' '\n' | grep -v '^  embedding [0-9]' \
    > "results/logs/$name.txt"
}

run 03_beir_baseline       concepts/03_real_benchmark/01_beir_baseline.py
run 05_bm25                concepts/05_keyword_search/01_bm25_from_scratch.py
run 06a_complementarity    concepts/06_hybrid_search/01_complementarity.py
run 06b_hybrid_eval        concepts/06_hybrid_search/02_hybrid_eval.py

echo "done: logs in results/logs/"
