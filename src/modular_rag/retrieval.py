"""Default retrieval and reranking implementations."""

import re
from typing import Any, Mapping, Optional, Sequence

from .models import SearchResult
from .ports import Embedder, VectorStore


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        vector = self.embedder.embed_query(query)
        return self.store.search(vector, top_k=top_k, filters=filters)


class KeywordReranker:
    """Small local reranker showing where a model-based reranker can plug in."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> Sequence[SearchResult]:
        query_terms = set(self._TOKEN.findall(query.lower()))

        def score(result: SearchResult):
            text_terms = set(self._TOKEN.findall(result.chunk.text.lower()))
            overlap = len(query_terms & text_terms)
            return (-overlap, -result.score, result.chunk.id)

        return tuple(sorted(results, key=score)[:top_k])
