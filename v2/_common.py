"""Shared entry point for the v2 stage scripts.

Every stage runs once per embedding model:

    python v2/s03_benchmark.py                       # all three
    python v2/s03_benchmark.py --embedding bge-base  # one
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.v2.models import EMBEDDING_ORDER  # noqa: E402


def embeddings_from_args(description: str) -> list[str]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--embedding", choices=EMBEDDING_ORDER, action="append",
                    help="limit to one or more embedding models (default: all three)")
    args, _ = ap.parse_known_args()
    return args.embedding or EMBEDDING_ORDER


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)
