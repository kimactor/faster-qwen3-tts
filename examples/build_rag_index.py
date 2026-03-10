#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_backends import SentenceTransformerEmbedder, document_to_embedding_text, load_rag_documents


def build_index(
    source_path: Path,
    output_path: Path,
    embedding_model: str,
    *,
    collection_id: str | None = None,
) -> None:
    docs = load_rag_documents(source_path, collection_id=collection_id)
    if not docs:
        raise ValueError(f"No RAG documents found in {source_path}")

    embedder = SentenceTransformerEmbedder(embedding_model)
    texts = [document_to_embedding_text(doc) for doc in docs]
    embeddings = embedder.encode(texts)
    doc_ids = np.asarray([doc["id"] for doc in docs], dtype=object)
    model_name = np.asarray(embedding_model, dtype=object)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        doc_ids=doc_ids,
        embedding_model=model_name,
    )

    print(f"built {len(docs)} embeddings -> {output_path}")
    print(f"embedding_model={embedding_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector RAG index for the realtime assistant")
    parser.add_argument("--source", required=True, help="Path to a JSON knowledge file")
    parser.add_argument("--output", required=True, help="Output .npz index path")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="sentence-transformers model name or local path",
    )
    parser.add_argument("--collection", default=None, help="Optional collection id inside a multi-knowledge-base JSON file")
    args = parser.parse_args()

    build_index(
        Path(args.source),
        Path(args.output),
        args.embedding_model,
        collection_id=args.collection,
    )


if __name__ == "__main__":
    main()
