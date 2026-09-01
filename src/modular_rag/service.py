"""Query orchestration that depends only on retrieval and generation ports."""

from typing import Any, Mapping, Optional, Sequence

from .models import Citation, RAGResponse, SearchResult
from .ports import AnswerGenerator, Reranker, Retriever


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        reranker: Optional[Reranker] = None,
        default_top_k: int = 5,
        default_fetch_k: int = 20,
    ) -> None:
        if default_top_k <= 0 or default_fetch_k <= 0:
            raise ValueError("Retrieval limits must be positive.")
        if default_fetch_k < default_top_k:
            raise ValueError("default_fetch_k cannot be smaller than default_top_k.")
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.default_fetch_k = default_fetch_k

    def ask(
        self,
        question: str,
        *,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        final_k = top_k if top_k is not None else self.default_top_k
        if final_k <= 0:
            raise ValueError("top_k must be positive.")

        fetch_k = max(final_k, self.default_fetch_k if self.reranker else final_k)
        candidates = tuple(
            self.retriever.retrieve(question, top_k=fetch_k, filters=filters)
        )
        results: Sequence[SearchResult]
        if self.reranker is not None:
            results = tuple(self.reranker.rerank(question, candidates, top_k=final_k))
        else:
            results = candidates[:final_k]

        answer = self.generator.generate(question, results)
        citations = tuple(
            self._citation(number, result)
            for number, result in enumerate(results, start=1)
        )
        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            results=tuple(results),
        )

    @staticmethod
    def _citation(number: int, result: SearchResult) -> Citation:
        metadata = result.chunk.metadata
        source = str(
            metadata.get("source_name")
            or metadata.get("source_path")
            or result.chunk.document_id
        )
        excerpt = " ".join(result.chunk.text.split())[:300]
        return Citation(
            number=number,
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            source=source,
            excerpt=excerpt,
            score=result.score,
            metadata=metadata,
        )
