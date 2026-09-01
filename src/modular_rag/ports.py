"""Protocols defining every component that an application may replace."""

from typing import Any, Mapping, Optional, Protocol, Sequence

from rag_ingestion import Document

from .models import Chunk, SearchResult, SentenceSpan, Vector, VectorRecord


class DocumentProcessor(Protocol):
    """Loads and cleans one source into a canonical document."""

    def process(
        self,
        source: Any,
        *,
        document_type: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Document:
        ...


class SentenceSegmenter(Protocol):
    """Identify ordered sentence spans in a cleaned document."""

    def segment(self, document: Document) -> Sequence[SentenceSpan]:
        ...


class DocumentIndex(Protocol):
    """Optional query-side index populated from canonical documents."""

    def replace_document(self, document: Document) -> None:
        ...

    def delete_document(self, document_id: str) -> int:
        ...


class Chunker(Protocol):
    def chunk(self, document: Document) -> Sequence[Chunk]:
        ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        ...

    def embed_query(self, text: str) -> Vector:
        ...


class VectorStore(Protocol):
    """Storage port; implementations should replace a document atomically."""

    def replace_document(
        self, document_id: str, records: Sequence[VectorRecord]
    ) -> None:
        ...

    def delete_document(self, document_id: str) -> int:
        ...

    def search(
        self,
        query_vector: Vector,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        filters: Optional[Mapping[str, Any]] = None,
    ) -> Sequence[SearchResult]:
        ...


class Reranker(Protocol):
    def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> Sequence[SearchResult]:
        ...


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: Sequence[SearchResult]) -> str:
        ...
