#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_backends import build_rag_backend, normalize_metadata_filters, parse_metadata_filters


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("items", []))
    return list(payload)


def evaluate_dataset(args: argparse.Namespace) -> int:
    items = load_eval_items(Path(args.dataset))
    if not items:
        raise ValueError(f"No evaluation items found in {args.dataset}")

    backend_cache = {}
    base_filters = parse_metadata_filters(args.rag_filter)
    hits = 0
    reciprocal_rank_sum = 0.0

    for item in items:
        collection_id = str(item.get("collection") or args.rag_collection or "").strip() or None
        cache_key = collection_id or "__all__"
        if cache_key not in backend_cache:
            backend_cache[cache_key] = build_rag_backend(
                args.rag_backend,
                args.rag_source,
                index_path=args.rag_index,
                embedding_model=args.rag_embedding_model,
                collection_id=collection_id,
                filters=base_filters,
            )

        item_filters = normalize_metadata_filters(item.get("filters"))
        chunks = backend_cache[cache_key].retrieve(
            str(item.get("question", "")),
            top_k=args.top_k,
            filters=item_filters,
        )
        retrieved_ids = [chunk.doc_id for chunk in chunks]
        expected_ids = [str(doc_id) for doc_id in item.get("expected_doc_ids", [])]
        hit_rank = None
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in expected_ids:
                hit_rank = rank
                break

        if hit_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / float(hit_rank)

        if args.verbose:
            print(f"[{item.get('id', '?')}] q={item.get('question', '')}")
            print(f"  expected={expected_ids}")
            print(f"  retrieved={retrieved_ids}")
            print(f"  hit_rank={hit_rank}")

    total = len(items)
    hit_rate = hits / float(total)
    mrr = reciprocal_rank_sum / float(total)
    print(f"items={total}")
    print(f"hit@{args.top_k}={hit_rate:.3f}")
    print(f"mrr={mrr:.3f}")
    return 0 if hits == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval quality on a QA dataset")
    parser.add_argument("--dataset", required=True, help="Path to the QA evaluation dataset JSON")
    parser.add_argument("--rag-backend", default="json-keyword", choices=["json-keyword", "vector-index"])
    parser.add_argument("--rag-source", required=True, help="Path to the knowledge JSON file")
    parser.add_argument("--rag-index", default=None, help="Vector index path when using vector-index")
    parser.add_argument("--rag-embedding-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--rag-collection", default=None, help="Default collection id")
    parser.add_argument("--rag-filter", action="append", default=None, help="Default metadata filter key=value1,value2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    raise SystemExit(evaluate_dataset(args))


if __name__ == "__main__":
    main()
