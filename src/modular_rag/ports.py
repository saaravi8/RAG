"""Protocols defining every component that an application may replace."""

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from rag_ingestion import Document

from .models import (
    Chunk,
    SearchResult,
    SentenceSpan,
    Vector,
    VectorRecord,
    VerificationResult,
)
from .transactions import IndexTransactionCoordinator


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


class PreparedDocumentReplacement(Protocol):
    """A staged document replacement participating in a coordinated write.

    Preparing a replacement must not change query-visible state. ``commit``
    should normally perform only a prebuilt, component-local atomic state swap.
    ``rollback`` must be non-raising and idempotent. It must discard an
    uncommitted replacement and restore the complete pre-prepare state when
    the committed candidate is still current. If a newer write has taken
    ownership, rollback must leave that newer state intact rather than restore
    a stale snapshot.
    """

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...


class PreparedDocumentDeletion(PreparedDocumentReplacement, Protocol):
    """A staged deletion exposing the primary component's removal count."""

    @property
    def deleted_count(self) -> int:
        ...


@runtime_checkable
class CoordinatedPreparedDocumentReplacement(
    PreparedDocumentReplacement, Protocol
):
    """A dynamic prepared change bound to one transaction coordinator.

    Primary and configured auxiliary participants expose their coordinator on
    the participant itself. Additional preparations have no separately
    configured participant for the Indexer to validate, so their returned
    handles must carry this explicit ownership binding.
    """

    @property
    def transaction_coordinator(self) -> IndexTransactionCoordinator:
        ...


@runtime_checkable
class TransactionalDocumentIndex(DocumentIndex, Protocol):
    """Document index capable of failure-atomic coordinated replacement."""

    transaction_coordinator: IndexTransactionCoordinator

    def prepare_replace_document(
        self, document: Document
    ) -> PreparedDocumentReplacement:
        ...

    def prepare_delete_document(
        self, document_id: str
    ) -> PreparedDocumentDeletion:
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


@runtime_checkable
class TransactionalVectorStore(VectorStore, Protocol):
    """Vector store capable of failure-atomic coordinated replacement."""

    transaction_coordinator: IndexTransactionCoordinator

    def prepare_replace_document(
        self, document_id: str, records: Sequence[VectorRecord]
    ) -> PreparedDocumentReplacement:
        ...

    def prepare_delete_document(
        self, document_id: str
    ) -> PreparedDocumentDeletion:
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


class AnswerVerifier(Protocol):
    """Check whether a generated answer is supported by its retrieved contexts."""

    def verify(
        self,
        question: str,
        answer: str,
        contexts: Sequence[SearchResult],
    ) -> VerificationResult:
        ...
