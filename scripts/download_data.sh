#!/usr/bin/env bash
# Fetch BEIR SciFact (~8MB zipped): 5,183 abstracts, 300 labelled test queries.
# Kept out of git because it is public, versioned upstream, and regenerable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/beir && cd data/beir

if [ -d scifact ]; then echo "scifact already present"; exit 0; fi

echo "downloading BEIR SciFact..."
curl -sSL -o scifact.zip \
  "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
unzip -oq scifact.zip && rm scifact.zip
echo "done: $(wc -l < scifact/corpus.jsonl) docs, $(wc -l < scifact/queries.jsonl) queries"
