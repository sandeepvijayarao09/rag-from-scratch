#!/usr/bin/env bash
# Run every v2 stage, 1-10, in dependency order. Each step resumes from its
# caches, so if anything stops, run this again and it carries on.
#
#   ./v2/run_all.sh            # everything
#   ./v2/run_all.sh fast       # only the steps without 12B generation
#
# Logs: ../logs/v2_<step>.txt
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../venv/bin/python
LOG=../logs
mkdir -p "$LOG" results/v2
step() {
  echo "== $1 ($(date +%H:%M))"; shift
  "$@" 2>&1 | tee "$LOG/v2_$STEP.txt"
  [ "${PIPESTATUS[0]}" -eq 0 ] && echo "   ok" || echo "   FAILED: see $LOG/v2_$STEP.txt (continuing)"
}

STEP=s03 step "Stage 3: formats and baselines" $PY v2/s03_benchmark.py
STEP=s05 step "Stage 5: BM25"                   $PY v2/s05_bm25.py
STEP=s06 step "Stage 6: hybrid"                 $PY v2/s06_hybrid.py
STEP=s04 step "Stage 4: chunking"               $PY v2/s04_chunking.py
STEP=s10 step "Stage 10: late interaction"      $PY v2/s10_late_interaction.py
STEP=s07 step "Stage 7: reranking"              $PY v2/s07_reranking.py
STEP=report step "Report (partial)"             $PY v2/report.py
[ "${1:-}" = "fast" ] && exit 0

# The 12B generation steps, the long part (most of a day).
STEP=s08gen step "Stage 8: 12B rewrites"        $PY v2/s08_query_rewriting.py generate
STEP=s08 step "Stage 8: scoring"                $PY v2/s08_query_rewriting.py score
STEP=s09gen step "Stage 9: 12B blurbs"          $PY v2/s09_contextual.py generate
STEP=s09 step "Stage 9: scoring"                $PY v2/s09_contextual.py score
STEP=s0102 step "Stages 1-2: cat facts"         $PY v2/s01_s02_cat_facts.py
STEP=report step "Report"                       $PY v2/report.py
echo "Done: results/v2/RESULTS.md"
