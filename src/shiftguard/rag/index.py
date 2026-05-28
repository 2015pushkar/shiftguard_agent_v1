"""Build/load the local Qdrant policy index using Ollama embeddings.

Embeddings come from `nomic-embed-text` via Ollama (keeps the whole stack on one
runner); vectors live in an embedded, path-backed Qdrant collection (no Docker).
Run `python -m shiftguard.rag.index` to (re)build the collection from the policy
corpus. Note: embedded Qdrant holds a file lock on its path, so only one client
may have it open at a time.
"""

from __future__ import annotations

import uuid

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..config import Settings, get_settings
from ..logging_setup import get_logger, setup_logging
from .chunking import Chunk, chunk_corpus

# Fixed namespace so a chunk's id maps to a stable Qdrant point id across rebuilds.
_POINT_NS = uuid.UUID("6f3d1c2a-0b5e-4a7d-9c8b-2e1f4a6d8c30")

# nomic-embed-text is asymmetric/instruction-tuned: documents and queries must
# carry different task prefixes so a query lands near its matching passage. The
# prefix is added only to what we embed, never to the stored/cited payload text.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

log = get_logger("rag.index")


def get_ollama_client(settings: Settings | None = None) -> OllamaClient:
    settings = settings or get_settings()
    return OllamaClient(host=settings.ollama_host)


def embed_texts(
    texts: list[str],
    settings: Settings | None = None,
    client: OllamaClient | None = None,
) -> list[list[float]]:
    """Embed a batch of texts; returns one vector per input, in order."""
    if not texts:
        return []
    settings = settings or get_settings()
    client = client or get_ollama_client(settings)
    response = client.embed(model=settings.embed_model, input=texts)
    return list(response.embeddings)


def embed_query(text: str, settings: Settings | None = None, client: OllamaClient | None = None) -> list[float]:
    return embed_texts([f"{QUERY_PREFIX}{text}"], settings=settings, client=client)[0]


def open_store(settings: Settings | None = None) -> QdrantClient:
    """Open the embedded Qdrant store. Caller is responsible for closing it."""
    settings = settings or get_settings()
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(_POINT_NS, chunk.chunk_id))


def build_index(rebuild: bool = False, settings: Settings | None = None) -> int:
    """Chunk the policy corpus, embed it, and upsert into Qdrant.

    Returns the number of chunks indexed. With `rebuild=True` the collection is
    dropped first for a clean, deterministic rebuild.
    """
    settings = settings or get_settings()
    chunks = chunk_corpus(settings.policies_dir)
    if not chunks:
        raise RuntimeError(f"no policy chunks found under {settings.policies_dir}")

    vectors = embed_texts([f"{DOC_PREFIX}{c.text}" for c in chunks], settings=settings)
    dim = len(vectors[0])
    collection = settings.qdrant_collection

    store = open_store(settings)
    try:
        if rebuild and store.collection_exists(collection):
            store.delete_collection(collection)
        if not store.collection_exists(collection):
            store.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        points = [
            PointStruct(id=_point_id(c), vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors)
        ]
        store.upsert(collection, points=points)
    finally:
        store.close()

    log.info("indexed %d chunks (dim=%d) into '%s'", len(points), dim, collection)
    return len(points)


def main() -> None:
    setup_logging("index")
    settings = get_settings()
    count = build_index(rebuild=True, settings=settings)
    log.info("done: %d chunks at %s", count, settings.qdrant_path)


if __name__ == "__main__":
    main()
