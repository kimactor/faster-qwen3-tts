from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


MetadataFilter = dict[str, tuple[str, ...]]


@dataclass
class RagChunk:
    doc_id: str
    title: str
    content: str
    score: float
    metadata: dict[str, Any]


class RagBackend(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: MetadataFilter | None = None,
    ) -> list[RagChunk]:
        ...


class NullRagBackend:
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: MetadataFilter | None = None,
    ) -> list[RagChunk]:
        return []


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]{2,}", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = [a + b for a, b in zip(cjk_chars, cjk_chars[1:])]
    return set(ascii_tokens) | set(cjk_chars) | set(cjk_bigrams)


def _normalize_filter_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set)):
        raw_values = values
    else:
        raw_values = [values]
    normalized = [str(value).strip().lower() for value in raw_values if str(value).strip()]
    return tuple(normalized)


def normalize_metadata_filters(filters: MetadataFilter | dict[str, Any] | None) -> MetadataFilter:
    if not filters:
        return {}
    normalized: MetadataFilter = {}
    for key, values in filters.items():
        name = str(key).strip()
        if not name:
            continue
        normalized_values = _normalize_filter_values(values)
        if normalized_values:
            normalized[name] = normalized_values
    return normalized


def parse_metadata_filters(filter_items: list[str] | tuple[str, ...] | None) -> MetadataFilter:
    parsed: dict[str, list[str]] = {}
    for item in filter_items or []:
        text = str(item).strip()
        if not text or "=" not in text:
            continue
        key, raw_values = text.split("=", 1)
        values = [part.strip() for part in raw_values.split(",") if part.strip()]
        if not key.strip() or not values:
            continue
        parsed.setdefault(key.strip(), []).extend(values)
    return normalize_metadata_filters(parsed)


def _matches_metadata(metadata: dict[str, Any], filters: MetadataFilter | None) -> bool:
    normalized_filters = normalize_metadata_filters(filters)
    if not normalized_filters:
        return True

    for key, expected_values in normalized_filters.items():
        actual = metadata.get(key)
        actual_values = _normalize_filter_values(actual)
        if not actual_values:
            return False
        if not any(value in actual_values for value in expected_values):
            return False
    return True


def list_rag_collections(source_path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    collections: list[dict[str, str]] = []
    if isinstance(payload, dict) and isinstance(payload.get("collections"), list):
        for idx, item in enumerate(payload["collections"], start=1):
            if not isinstance(item, dict):
                continue
            collection_id = str(item.get("id", f"collection-{idx}")).strip()
            if not collection_id:
                continue
            collections.append(
                {
                    "id": collection_id,
                    "title": str(item.get("title", collection_id)).strip(),
                    "description": str(item.get("description", "")).strip(),
                }
            )
    return collections


def load_rag_documents(
    source_path: str | Path,
    collection_id: str | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    docs: list[dict[str, Any]] = []

    if isinstance(payload, dict) and isinstance(payload.get("collections"), list):
        entries = payload["collections"]
    elif isinstance(payload, dict):
        entries = [{"id": "default", "title": "default", "documents": payload.get("documents", [])}]
    else:
        entries = [{"id": "default", "title": "default", "documents": payload}]

    for collection_index, collection in enumerate(entries, start=1):
        if not isinstance(collection, dict):
            continue
        current_collection_id = str(collection.get("id", f"collection-{collection_index}")).strip() or f"collection-{collection_index}"
        if collection_id and current_collection_id != collection_id:
            continue
        collection_title = str(collection.get("title", current_collection_id)).strip()
        collection_metadata = dict(collection.get("metadata") or {})
        collection_tags = list(collection.get("tags") or [])
        items = collection.get("documents", [])

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            title = str(item.get("title", f"doc-{idx+1}")).strip()
            doc_id = str(item.get("id", title)).strip()
            tags = list(item.get("tags") or [])
            merged_tags = [*collection_tags, *tags]
            metadata = {
                "collection_id": current_collection_id,
                "collection_title": collection_title,
                "tags": merged_tags,
                **collection_metadata,
                **(item.get("metadata") or {}),
            }
            docs.append(
                {
                    "id": doc_id,
                    "title": title,
                    "content": content,
                    "metadata": metadata,
                    "tokens": _tokenize(" ".join([title, content, " ".join(map(str, merged_tags))])),
                }
            )
    return docs


def document_to_embedding_text(doc: dict[str, Any]) -> str:
    tags = doc.get("metadata", {}).get("tags") or []
    parts = [str(doc.get("title", "")).strip(), str(doc.get("content", "")).strip()]
    if tags:
        parts.append(" ".join(map(str, tags)))
    return "\n".join(part for part in parts if part)


class JsonKeywordRagBackend:
    def __init__(
        self,
        source_path: str | Path,
        *,
        collection_id: str | None = None,
        default_filters: MetadataFilter | None = None,
    ):
        self.source_path = Path(source_path)
        self.collection_id = collection_id or ""
        self.default_filters = normalize_metadata_filters(default_filters)
        if self.collection_id:
            self.default_filters = normalize_metadata_filters(
                {**self.default_filters, "collection_id": (self.collection_id,)}
            )
        self._docs = load_rag_documents(self.source_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: MetadataFilter | None = None,
    ) -> list[RagChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        effective_filters = normalize_metadata_filters({**self.default_filters, **normalize_metadata_filters(filters)})

        scored: list[RagChunk] = []
        for doc in self._docs:
            if not _matches_metadata(doc["metadata"], effective_filters):
                continue
            overlap = query_tokens & doc["tokens"]
            if not overlap:
                continue
            score = float(len(overlap))
            scored.append(
                RagChunk(
                    doc_id=doc["id"],
                    title=doc["title"],
                    content=doc["content"],
                    score=score,
                    metadata=doc["metadata"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:max(1, int(top_k))]


class SentenceTransformerEmbedder:
    def __init__(self, model_name_or_path: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Vector RAG requires sentence-transformers. Install it with: pip install sentence-transformers"
            ) from exc
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class IndexedVectorRagBackend:
    def __init__(
        self,
        source_path: str | Path,
        index_path: str | Path,
        embedding_model: str,
        *,
        collection_id: str | None = None,
        default_filters: MetadataFilter | None = None,
    ):
        self.source_path = Path(source_path)
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.collection_id = collection_id or ""
        self.default_filters = normalize_metadata_filters(default_filters)
        if self.collection_id:
            self.default_filters = normalize_metadata_filters(
                {**self.default_filters, "collection_id": (self.collection_id,)}
            )
        self._docs = load_rag_documents(self.source_path)
        self._embedder = SentenceTransformerEmbedder(embedding_model)
        self._embeddings = self._load_index()

    def _load_index(self) -> np.ndarray:
        payload = np.load(self.index_path, allow_pickle=True)
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        if embeddings.shape[0] != len(self._docs):
            raise ValueError(
                f"RAG index/doc mismatch: {embeddings.shape[0]} embeddings but {len(self._docs)} documents"
            )
        if "doc_ids" in payload:
            index_doc_ids = [str(item) for item in payload["doc_ids"].tolist()]
            runtime_doc_ids = [str(doc["id"]) for doc in self._docs]
            if index_doc_ids != runtime_doc_ids:
                raise ValueError("RAG index doc_ids do not match the current knowledge file order/content")
        if "embedding_model" in payload:
            index_model = str(payload["embedding_model"].tolist())
            if index_model and index_model != self.embedding_model:
                raise ValueError(
                    f"RAG index was built with {index_model}, but runtime embedding model is {self.embedding_model}"
                )
        return embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filters: MetadataFilter | None = None,
    ) -> list[RagChunk]:
        query = (query or "").strip()
        if not query:
            return []
        effective_filters = normalize_metadata_filters({**self.default_filters, **normalize_metadata_filters(filters)})
        query_vec = self._embedder.encode([query])[0]
        scores = self._embeddings @ query_vec
        top_k = max(1, min(int(top_k), len(self._docs)))
        top_indices = np.argsort(scores)[::-1]

        chunks: list[RagChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self._docs[int(idx)]
            if not _matches_metadata(doc["metadata"], effective_filters):
                continue
            chunks.append(
                RagChunk(
                    doc_id=doc["id"],
                    title=doc["title"],
                    content=doc["content"],
                    score=score,
                    metadata=doc["metadata"],
                )
            )
            if len(chunks) >= top_k:
                break
        return chunks


def build_rag_backend(
    backend: str,
    source_path: str | Path | None,
    *,
    index_path: str | Path | None = None,
    embedding_model: str | None = None,
    collection_id: str | None = None,
    filters: MetadataFilter | dict[str, Any] | None = None,
) -> RagBackend:
    backend = (backend or "none").strip().lower()
    if backend == "none" or not source_path:
        return NullRagBackend()
    if backend == "json-keyword":
        return JsonKeywordRagBackend(
            source_path,
            collection_id=collection_id,
            default_filters=filters,
        )
    if backend == "vector-index":
        if not index_path:
            raise ValueError("vector-index backend requires --rag-index")
        if not embedding_model:
            raise ValueError("vector-index backend requires --rag-embedding-model")
        return IndexedVectorRagBackend(
            source_path,
            index_path,
            embedding_model,
            collection_id=collection_id,
            default_filters=filters,
        )
    raise ValueError(f"Unsupported RAG backend: {backend}")


def format_rag_context(chunks: list[RagChunk], max_chars: int = 1600) -> str:
    if not chunks:
        return ""

    parts: list[str] = []
    total = 0
    for idx, chunk in enumerate(chunks, start=1):
        block = f"[{idx}] {chunk.title}\n{chunk.content.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()
