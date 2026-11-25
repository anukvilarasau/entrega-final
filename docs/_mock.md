# Evaluación del RAG (2026-11-25 14:22)

- **modo**: `mock`
- **embeddings**: `hash-bolsa-de-palabras`
- **store**: `chroma:chroma/guias_viaje`
- **reranker**: `solapamiento-lexico`
- **candidatos_rerank**: `12`
- **max_caracteres_rerank**: `1200`
- **k**: `4`
- **candidatos_por_buscador**: `20`
- **rrf_k**: `60`
- **chunk_tokens**: `500`
- **chunk_overlap**: `60`
- **fragmentos_en_indice**: `272`
- **preguntas**: `15`
- **fecha**: `2026-11-25 14:22`

| Modo | Recall@1 doc | Recall@4 doc | Precision@4 doc | Recall@1 frag | Recall@4 frag | Precision@4 frag | MRR | ms/consulta |
|---|---|---|---|---|---|---|---|---|
| vectorial | 0.93 | 1.00 | 0.87 | 0.40 | 0.52 | 0.33 | 0.52 | 3 |
| bm25 | 0.80 | 1.00 | 0.83 | 0.67 | 0.75 | 0.48 | 0.72 | 0 |
| hibrido_rrf | 0.87 | 1.00 | 0.85 | 0.60 | 0.61 | 0.40 | 0.67 | 4 |
| hibrido_rrf_rerank | 0.87 | 1.00 | 0.80 | 0.53 | 0.61 | 0.42 | 0.62 | 5 |
