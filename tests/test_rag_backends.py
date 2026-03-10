from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from rag_backends import (
    JsonKeywordRagBackend,
    RagChunk,
    build_rag_backend,
    format_rag_context,
    list_rag_collections,
    load_rag_documents,
    parse_metadata_filters,
)


SAMPLE_JSON = r"""
{
  "documents": [
    {
      "id": "doc-1",
      "title": "\u6c5f\u9634\u9a6c\u8e44\u9165",
      "content": "\u5c42\u6b21\u9165\u677e\uff0c\u9002\u5408\u505a\u5730\u65b9\u7279\u8272\u8bb2\u89e3\u3002",
      "tags": ["\u6c5f\u9634", "\u70b9\u5fc3"]
    },
    {
      "id": "doc-2",
      "title": "\u6570\u5b57\u4eba\u8bb2\u89e3",
      "content": "\u56de\u7b54\u8981\u77ed\u53e5\u4f18\u5148\u3002",
      "tags": ["\u6570\u5b57\u4eba"]
    }
  ]
}
""".encode("ascii").decode("unicode_escape")


def test_load_rag_documents_reads_json_and_tags(tmp_path):
    source = tmp_path / "knowledge.json"
    source.write_text(SAMPLE_JSON, encoding="utf-8")

    docs = load_rag_documents(source)

    assert len(docs) == 2
    assert docs[0]["id"] == "doc-1"
    assert "\u6c5f\u9634" in docs[0]["metadata"]["tags"]
    assert "\u9a6c\u8e44" in docs[0]["tokens"]


def test_json_keyword_backend_returns_matching_chunks(tmp_path):
    source = tmp_path / "knowledge.json"
    source.write_text(SAMPLE_JSON, encoding="utf-8")

    backend = build_rag_backend("json-keyword", source)
    chunks = backend.retrieve("\u8bf7\u4ecb\u7ecd\u6c5f\u9634\u9a6c\u8e44\u9165", top_k=2)

    assert isinstance(backend, JsonKeywordRagBackend)
    assert chunks
    assert chunks[0].doc_id == "doc-1"


def test_load_rag_documents_reads_collections_and_metadata_filters(tmp_path):
    source = tmp_path / "collections.json"
    source.write_text(
        """
        {
          "collections": [
            {
              "id": "brand",
              "title": "品牌库",
              "metadata": {"scene": "brand"},
              "documents": [
                {
                  "id": "doc-a",
                  "title": "品牌介绍",
                  "content": "介绍地方特色点心时要突出酥香。",
                  "metadata": {"audience": ["tourist", "family"]}
                }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    collections = list_rag_collections(source)
    docs = load_rag_documents(source)

    assert collections[0]["id"] == "brand"
    assert docs[0]["metadata"]["collection_id"] == "brand"
    assert docs[0]["metadata"]["scene"] == "brand"
    assert "tourist" in docs[0]["metadata"]["audience"]


def test_json_keyword_backend_honors_collection_and_filters(tmp_path):
    source = tmp_path / "collections.json"
    source.write_text(
        """
        {
          "collections": [
            {
              "id": "hall",
              "title": "展厅库",
              "documents": [
                {
                  "id": "hall-short",
                  "title": "短句规则",
                  "content": "展厅讲解优先短句。",
                  "metadata": {"audience": ["visitor", "family"]}
                }
              ]
            },
            {
              "id": "brand",
              "title": "品牌库",
              "documents": [
                {
                  "id": "brand-gift",
                  "title": "送礼推荐",
                  "content": "伴手礼要突出地方特色。",
                  "metadata": {"audience": ["customer"]}
                }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    backend = build_rag_backend(
        "json-keyword",
        source,
        collection_id="hall",
        filters=parse_metadata_filters(["audience=family"]),
    )
    chunks = backend.retrieve("展厅里怎么回答更自然", top_k=2)

    assert chunks
    assert chunks[0].doc_id == "hall-short"


def test_format_rag_context_respects_limit():
    context = format_rag_context(
        [
            RagChunk(doc_id="1", title="A", content="12345", score=1.0, metadata={}),
            RagChunk(doc_id="2", title="B", content="67890", score=0.9, metadata={}),
        ],
        max_chars=12,
    )

    assert context == "[1] A\n12345"
