# light-rag-test

A sandbox for experimenting with [LightRAG](https://github.com/HKUDS/LightRAG), a lightweight retrieval-augmented generation (RAG) framework that combines knowledge-graph and vector retrieval.

## Purpose

- Try out LightRAG's indexing and query modes (`naive`, `local`, `global`, `hybrid`, `mix`)
- Compare retrieval quality across different documents and LLM/embedding backends
- Keep small, reproducible test scripts and notes in one place

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install lightrag-hku
```

See the [LightRAG documentation](https://github.com/HKUDS/LightRAG#readme) for configuring an LLM and embedding provider.

## Status

Early setup — test scripts and sample data will be added here.
