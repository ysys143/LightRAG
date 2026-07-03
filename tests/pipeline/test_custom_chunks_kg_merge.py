"""Regression test: ``LightRAG.ainsert_custom_chunks`` must build the KG.

Historically ``ainsert_custom_chunks`` ran entity/relation extraction (paying
the LLM cost and logging ``Chunk N of M extracted X Ent + Y Rel``) but threw
the extracted ``chunk_results`` away without calling ``merge_nodes_and_edges``.
The chunk vectors were persisted, so ``naive`` retrieval worked, but the graph
and entity/relationship vector stores stayed empty and every KG-dependent query
mode (local/global/hybrid/mix) returned no context.

This test drives the method end-to-end with a real (file-based) LightRAG
instance and a deterministic mock LLM, then asserts the graph is populated.
"""

import tempfile

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    wrap_embedding_func_with_attrs,
)

# Deterministic entity/relation extraction output in LightRAG's record format
# (records separated by newlines, fields by the tuple delimiter).
_EXTRACTION = (
    "(entity<|#|>MARIE CURIE<|#|>PERSON<|#|>A physicist who discovered polonium and radium)\n"
    "(entity<|#|>PIERRE CURIE<|#|>PERSON<|#|>A physicist who collaborated with Marie Curie)\n"
    "(relationship<|#|>MARIE CURIE<|#|>PIERRE CURIE<|#|>They collaborated on radioactivity research<|#|>collaboration<|#|>9)\n"
    "<|COMPLETE|>"
)


class _DummyTokenizer(TokenizerInterface):
    """1:1 char/token mapping so the test needs no tiktoken download."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_llm(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
):
    text = (prompt or "").lower()
    if "entity" in text or "extract" in text:
        return _EXTRACTION
    return "stub answer"


@wrap_embedding_func_with_attrs(embedding_dim=16, max_token_size=8192)
async def _mock_embed(texts):
    # Deterministic vectors; content is irrelevant to this test.
    rng = np.random.default_rng(0)
    return rng.random((len(texts), 16), dtype=np.float32)


@pytest.mark.offline
async def test_ainsert_custom_chunks_populates_knowledge_graph():
    with tempfile.TemporaryDirectory(prefix="lr_custom_chunks_") as workdir:
        rag = LightRAG(
            working_dir=workdir,
            llm_model_func=_mock_llm,
            embedding_func=_mock_embed,
            tokenizer=Tokenizer("dummy", _DummyTokenizer()),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        try:
            chunks = [
                "Marie Curie discovered polonium and radium. She won two Nobel Prizes.",
                "Pierre Curie collaborated with Marie Curie on radioactivity research.",
            ]
            await rag.ainsert_custom_chunks(
                full_text="\n\n".join(chunks),
                text_chunks=chunks,
                doc_id="doc-1",
            )

            # The extracted entities must have been merged into the graph.
            labels = await rag.chunk_entity_relation_graph.get_all_labels()
            assert "MARIE CURIE" in labels
            assert "PIERRE CURIE" in labels

            # And the entity vector store must be populated (not left empty).
            marie = await rag.chunk_entity_relation_graph.get_node("MARIE CURIE")
            assert marie is not None
        finally:
            await rag.finalize_storages()
