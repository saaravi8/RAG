"""Query orchestration that depends only on retrieval and generation ports."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
        query_methods: Optional[Mapping[str, Retriever]] = None,
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
        self._retrievers: Dict[str, Retriever] = {"standard": retriever}
        for name, method_retriever in (query_methods or {}).items():
            normalized = self._normalize_method(name)
            if normalized == "standard":
                raise ValueError("'standard' is reserved for the default retriever.")
            if normalized in self._retrievers:
                raise ValueError("Duplicate query method: {!r}.".format(normalized))
            self._retrievers[normalized] = method_retriever

    @property
    def available_methods(self) -> Tuple[str, ...]:
        return tuple(self._retrievers)

    def ask(
        self,
        question: str,
        *,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: Optional[int] = None,
        method: str = "standard",
    ) -> RAGResponse:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        final_k = top_k if top_k is not None else self.default_top_k
        if final_k <= 0:
            raise ValueError("top_k must be positive.")

        selected_method = self._normalize_method(method)
        try:
            retriever = self._retrievers[selected_method]
        except KeyError as exc:
            raise ValueError(
                "Unknown query method {!r}. Available methods: {}.".format(
                    selected_method, ", ".join(self.available_methods)
                )
            ) from exc

        fetch_k = max(final_k, self.default_fetch_k if self.reranker else final_k)
        candidates = tuple(retriever.retrieve(question, top_k=fetch_k, filters=filters))
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
    def _normalize_method(method: str) -> str:
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        normalized = method.strip().lower()
        if not normalized:
            raise ValueError("method cannot be empty.")
        return normalized

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
