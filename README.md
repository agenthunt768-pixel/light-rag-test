# light-rag-test

A sandbox for experimenting with [LightRAG](https://github.com/HKUDS/LightRAG), a lightweight retrieval-augmented generation (RAG) framework that combines knowledge-graph and vector retrieval.

## Purpose

- Try out LightRAG's indexing and query modes (`naive`, `local`, `global`, `hybrid`, `mix`)
- Compare retrieval quality across different documents and LLM/embedding backends
- Keep small, reproducible test scripts and notes in one place

## Setup

Python 3.12 and a running [Ollama](https://ollama.com) daemon are required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models used (see [`lightrag_ollama.py`](lightrag_ollama.py)):

| Role       | Model              | Where it runs                                  |
|------------|--------------------|------------------------------------------------|
| LLM        | `kimi-k2.6:cloud`  | Ollama Cloud, via the local daemon (`ollama signin`) |
| Embeddings | `nomic-embed-text` | Locally (`ollama pull nomic-embed-text`)       |

Override with `OLLAMA_HOST`, `LLM_MODEL`, `EMBEDDING_MODEL` and `EMBEDDING_DIM`.

## Running the tests

```bash
pytest -v
```

The test in [`tests/test_lightrag_ollama.py`](tests/test_lightrag_ollama.py) indexes a short document about a fictional company, checks that the expected entities land in the knowledge graph, and then asks a question in each query mode (`naive`, `local`, `global`, `hybrid`, `mix`). The tests are skipped if Ollama isn't reachable.

For a quick demo without pytest:

```bash
python lightrag_ollama.py
```
