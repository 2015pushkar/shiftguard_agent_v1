"""Backend for the `search_policy` tool: embed a query, search Qdrant, return
top-k policy chunks with citations.

The embedded Qdrant store holds a file lock on its path, so a single retriever
instance keeps one open connection and reuses it across the agent loop rather
than reopening per query. Use `get_retriever()` for the shared process-wide one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient

from ..config import Settings, get_settings
from .index import embed_query, open_store


class PolicyIndexMissingError(RuntimeError):
    """Raised when the Qdrant policy collection has not been built yet."""


@dataclass(frozen=True)
class PolicyHit:
    score: float
    doc: str
    section: str
    source: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc} > {self.section}"

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "source": self.source,
            "score": round(self.score, 3),
            "text": self.text,
        }


class PolicyRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        store: QdrantClient | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.settings = settings or get_settings()
        self._store = store
        self._ollama = ollama_client

    @property
    def store(self) -> QdrantClient:
        if self._store is None:
            store = open_store(self.settings)
            if not store.collection_exists(self.settings.qdrant_collection):
                store.close()
                raise PolicyIndexMissingError(
                    f"policy collection '{self.settings.qdrant_collection}' not found at "
                    f"{self.settings.qdrant_path}. Build it with: python -m shiftguard.rag.index"
                )
            self._store = store
        return self._store

    def search(self, query: str, top_k: int | None = None) -> list[PolicyHit]:
        query = (query or "").strip()
        if not query:
            return []
        limit = top_k if top_k is not None else self.settings.top_k
        vector = embed_query(query, settings=self.settings, client=self._ollama)
        points = self.store.query_points(
            self.settings.qdrant_collection, query=vector, limit=limit
        ).points
        return [
            PolicyHit(
                score=p.score,
                doc=p.payload["doc"],
                section=p.payload["section"],
                source=p.payload["source"],
                text=p.payload["text"],
            )
            for p in points
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


@lru_cache
def get_retriever() -> PolicyRetriever:
    """Return the shared, process-wide retriever (one open Qdrant connection)."""
    return PolicyRetriever()


def search_policy(query: str, top_k: int | None = None) -> list[dict]:
    """`search_policy` tool entry point: top-k policy chunks as compact dicts."""
    return [hit.to_dict() for hit in get_retriever().search(query, top_k=top_k)]
