"""Build a LightRAG instance backed by Ollama.

The LLM is an Ollama Cloud model (``kimi-k2.6:cloud`` by default) reached
through the local Ollama daemon, which forwards ``:cloud`` models to
ollama.com using the daemon's own sign-in. Embeddings run locally with
``nomic-embed-text``, since Ollama Cloud does not serve embedding models.

Override the defaults with environment variables:

    OLLAMA_HOST       Ollama daemon URL           (default http://localhost:11434)
    LLM_MODEL         chat model                  (default kimi-k2.6:cloud)
    EMBEDDING_MODEL   embedding model             (default nomic-embed-text)
    EMBEDDING_DIM     embedding vector dimension  (default 768)

Run ``python lightrag_ollama.py`` for a quick insert-and-query demo.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from functools import partial

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "kimi-k2.6:cloud")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# nomic-embed-text is trained with task prefixes; retrieval is noticeably
# worse without them. Other embedding models get the text unchanged.
_EMBED_PREFIXES = (
    {"query_prefix": "search_query: ", "document_prefix": "search_document: "}
    if EMBEDDING_MODEL.startswith("nomic-embed-text")
    else {}
)


async def build_rag(working_dir: str) -> LightRAG:
    """Create and initialize a LightRAG instance that stores its data in ``working_dir``."""
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={
            # Pin the host. Without it LightRAG sends ``:cloud`` model names
            # straight to ollama.com, which requires OLLAMA_API_KEY.
            "host": OLLAMA_HOST,
            # Skip the model's reasoning phase: faster, and extraction
            # prompts don't benefit from it.
            "options": {"think": False},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=2048,
            model_name=EMBEDDING_MODEL,
            supports_asymmetric=True,
            # ``.func`` is the undecorated function; wrapping the decorated
            # ollama_embed again would apply its bge-m3 defaults.
            func=partial(
                ollama_embed.func,
                embed_model=EMBEDDING_MODEL,
                host=OLLAMA_HOST,
                **_EMBED_PREFIXES,
            ),
        ),
    )
    await rag.initialize_storages()
    return rag


SAMPLE_DOC = """\
Brightwater Labs is a small robotics company founded in 2019 by Ines Okafor
in Bergen, Norway. Its main product is the Heron-3, an underwater drone that
inspects salmon farm nets for tears and biofouling.

The Heron-3 is built in partnership with Fjordline Marine, which supplies the
drone's sonar module. In 2024 Brightwater Labs hired Tomas Varga as chief
engineer; he previously designed sonar systems at Fjordline Marine.
"""


async def _demo() -> None:
    with tempfile.TemporaryDirectory() as working_dir:
        rag = await build_rag(working_dir)
        try:
            await rag.ainsert(SAMPLE_DOC)
            question = "Who founded Brightwater Labs, and what does its main product do?"
            answer = await rag.aquery(
                question, param=QueryParam(mode="hybrid", enable_rerank=False)
            )
            print(f"Q: {question}\nA: {answer}")
        finally:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(_demo())
