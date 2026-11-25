# Evaluación del RAG (2026-11-25 14:23)

- **modo**: `real`
- **embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **store**: `chroma:indice_chroma/guias_viaje`
- **reranker**: `jinaai/jina-reranker-v2-base-multilingual`
- **candidatos_rerank**: `12`
- **max_caracteres_rerank**: `1200`
- **k**: `4`
- **candidatos_por_buscador**: `20`
- **rrf_k**: `60`
- **chunk_tokens**: `500`
- **chunk_overlap**: `60`
- **fragmentos_en_indice**: `246`
- **preguntas**: `15`
- **fecha**: `2026-11-25 14:23`

| Modo | Recall@1 doc | Recall@4 doc | Precision@4 doc | Recall@1 frag | Recall@4 frag | Precision@4 frag | MRR | ms/consulta |
|---|---|---|---|---|---|---|---|---|
| vectorial | 0.80 | 1.00 | 0.63 | 0.53 | 0.59 | 0.35 | 0.62 | 10 |
| bm25 | 0.80 | 1.00 | 0.80 | 0.67 | 0.73 | 0.47 | 0.72 | 1 |
| hibrido_rrf | 0.93 | 0.93 | 0.82 | 0.73 | 0.60 | 0.38 | 0.73 | 10 |
| hibrido_rrf_rerank | 1.00 | 1.00 | 0.93 | 0.73 | 0.78 | 0.48 | 0.82 | 2404 |
