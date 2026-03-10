# Realtime Assistant RAG Workflow

1. Prepare a JSON knowledge file like `config/rag_knowledge.sample.json` with stable `id`, `title`, `content`, and optional `tags`.
2. Start with the keyword baseline:
   - `python examples/realtime_voice_assistant.py --rag-backend json-keyword --rag-source config/rag_knowledge.sample.json --rag-debug`
   - Watch which chunks are injected and trim overly long or weak documents first.
3. For a digital-human or exhibition deployment, move from one flat file to a collection-based file like `config/rag_collections.sample.json`.
   - Use one collection per scene, booth, tenant, or product line.
   - Put stable routing metadata into each collection or document, for example `scene`, `audience`, `tone`, or `priority`.
   - At runtime switch with `--rag-collection digital_human_hall` and narrow further with repeated `--rag-filter key=value`.
4. When the chunking is stable, build a vector index:
   - `pip install sentence-transformers`
   - `python examples/build_rag_index.py --source config/rag_knowledge.sample.json --output config/rag_knowledge.sample.index.npz --embedding-model BAAI/bge-small-zh-v1.5`
5. Run the assistant with vector retrieval:
   - `python examples/realtime_voice_assistant.py --rag-backend vector-index --rag-source config/rag_knowledge.sample.json --rag-index config/rag_knowledge.sample.index.npz --rag-embedding-model BAAI/bge-small-zh-v1.5 --rag-debug`
6. Build an offline evaluation set before tuning generation:
   - collect 30-50 real user questions
   - record the expected supporting chunk ids
   - store them in a file like `config/rag_eval.sample.json`
   - run `python examples/evaluate_rag.py --dataset config/rag_eval.sample.json --rag-backend json-keyword --rag-source config/rag_collections.sample.json --top-k 3 --verbose`
7. Iterate in this order: chunk quality -> retrieval recall -> collection routing -> metadata filters -> prompt wording -> reranking.
8. Keep the runtime contract stable: custom backends should continue to implement `retrieve(query, top_k, filters=None)` so you can swap JSON, local vector, or an external RAG service without rewriting the assistant loop.
