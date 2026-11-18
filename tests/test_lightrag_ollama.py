"""End-to-end smoke test: index a small document with LightRAG, then query it.

The document describes a made-up company, so correct answers have to come
from retrieval rather than the model's own knowledge.

Requires a running Ollama daemon that is signed in to ollama.com (for the
cloud LLM) and has the embedding model pulled. The tests are skipped when
the daemon cannot be reached.
"""

from __future__ import annotations

import ollama
import pytest
import pytest_asyncio
from lightrag import LightRAG, QueryParam

from lightrag_ollama import EMBEDDING_MODEL, OLLAMA_HOST, SAMPLE_DOC, build_rag

# Share one event loop across the module so the indexed instance can be
# reused by every test instead of re-indexing per test.
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _require_ollama() -> None:
    try:
        models = ollama.Client(host=OLLAMA_HOST).list().models
    except Exception as exc:  # connection refused, DNS, etc.
        pytest.skip(f"Ollama not reachable at {OLLAMA_HOST}: {exc}")
    names = {m.model for m in models}
    if not any(n.split(":")[0] == EMBEDDING_MODEL.split(":")[0] for n in names):
        pytest.skip(f"Embedding model {EMBEDDING_MODEL!r} not pulled; run: ollama pull {EMBEDDING_MODEL}")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def rag(tmp_path_factory) -> LightRAG:
    _require_ollama()
    instance = await build_rag(str(tmp_path_factory.mktemp("rag_storage")))
    await instance.ainsert(SAMPLE_DOC)
    yield instance
    await instance.finalize_storages()


async def test_document_is_processed(rag: LightRAG) -> None:
    counts = await rag.get_processing_status()
    assert counts.get("processed") == 1, counts
    assert not counts.get("failed"), counts


async def test_knowledge_graph_has_expected_entities(rag: LightRAG) -> None:
    labels = [label.lower() for label in await rag.chunk_entity_relation_graph.get_all_labels()]
    for expected in ("brightwater", "okafor", "heron"):
        assert any(expected in label for label in labels), f"{expected!r} not in {labels}"


@pytest.mark.parametrize("mode", ["naive", "local", "global", "hybrid", "mix"])
async def test_query_answers_from_document(rag: LightRAG, mode: str) -> None:
    answer = await rag.aquery(
        "Who founded Brightwater Labs?",
        param=QueryParam(mode=mode, enable_rerank=False),
    )
    assert "okafor" in answer.lower(), f"[{mode}] {answer}"
