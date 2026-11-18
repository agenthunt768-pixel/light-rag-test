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

| Role       | Model                 | Where it runs                                        |
|------------|-----------------------|------------------------------------------------------|
| LLM        | `glm-5.3-flash:cloud` | Ollama Cloud, via the local daemon (`ollama signin`) |
| Embeddings | `nomic-embed-text`    | Locally (`ollama pull nomic-embed-text`)             |

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

## LightRAG server

```bash
cp .env.example .env      # server settings: models, entity types, timeouts
lightrag-server           # web UI and API on http://127.0.0.1:9621
```

`.env.example` binds the server to `127.0.0.1` because no login is configured. It also points extraction at the desk entity types in [`prompts/entity_type/gfm_desk.yml`](prompts/entity_type/gfm_desk.yml) and raises the embedding timeout. On a CPU-only machine, the 30-second default stopped a large ingest partway through.

## Ingesting the mistral-synthetic data pack

The pack (`mistral-synthetic-2026-09-29/`) is kept locally and is not in git. [`ingest/prepare_mistral.py`](ingest/prepare_mistral.py) converts its text sources into files the server accepts, written to `ingest_staging/`:

| Source | Files | Conversion |
|---|---:|---|
| Emails | 260 | One Markdown file per email; boilerplate and recipient lists removed, PDF attachment text appended |
| Pitchbook decks | 12 | Current version of each deck, copied as `.pptx` |
| Product grid | 31 | One file per product family and region, plus `PRODUCTS.md` |
| Research notes | 138 | One file per underlying, each note a dated section |

```bash
python ingest/prepare_mistral.py all
for f in ingest_staging/*/*; do curl -s -X POST http://127.0.0.1:9621/documents/upload -F "file=@$f"; done
```

Market data, filings, news, transcripts and chat are not ingested. On a CPU-only laptop, indexing all 441 files took about three hours, mostly local embedding time.

## Evaluating against the golden set

```bash
python ingest/eval_golden.py --mode mix
```

[`ingest/eval_golden.py`](ingest/eval_golden.py) sends the 32 golden-set questions to the running server. Each expected answer line is reduced to the product families it names (TARF, DCL, CLI, DNT, participation note…). A line counts as recalled when the answer mentions one of those families. Results go to `eval_results/`.

The golden answers were written from the bank's real emails, and some expected products never appear in this pack. Treat scores as a comparison between runs, not as absolute accuracy. First run (`mix` mode): family recall 0.52 overall, and 0.64 counting only products present in the pack.
