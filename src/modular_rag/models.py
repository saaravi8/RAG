"""Provider-neutral domain models used by the RAG core."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

Vector = Tuple[float, ...]


@dataclass(frozen=True)
class SentenceSpan:
    """One identified sentence with its position in the cleaned document."""

    id: str
    document_id: str
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SentenceSpan.id cannot be empty.")
        if not self.document_id:
            raise ValueError("SentenceSpan.document_id cannot be empty.")
        if not self.text.strip():
            raise ValueError("SentenceSpan.text cannot be empty.")
        if self.index < 0:
            raise ValueError("SentenceSpan.index cannot be negative.")
        if self.start_char < 0:
            raise ValueError("SentenceSpan.start_char cannot be negative.")
        if self.end_char <= self.start_char:
            raise ValueError("SentenceSpan.end_char must be after start_char.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a source document."""

    id: str
    document_id: str
    text: str
    index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Chunk.id cannot be empty.")
        if not self.document_id:
            raise ValueError("Chunk.document_id cannot be empty.")
        if not self.text.strip():
            raise ValueError("Chunk.text cannot be empty.")
        if self.index < 0:
            raise ValueError("Chunk.index cannot be negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class VectorRecord:
    """A chunk paired with the vector used to retrieve it."""

    chunk: Chunk
    vector: Vector

    def __post_init__(self) -> None:
        if not self.vector:
            raise ValueError("VectorRecord.vector cannot be empty.")
        object.__setattr__(self, "vector", tuple(float(value) for value in self.vector))


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its latest retrieval or reranking score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class IndexReport:
    """Summary returned after a document has been indexed."""

    document_id: str
    chunk_count: int


@dataclass(frozen=True)
class Citation:
    """Source information exposed alongside a generated answer."""

    number: int
    chunk_id: str
    document_id: str
    source: str
    excerpt: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class VerificationResult:
    """Provider-neutral support verdict returned by an answer verifier."""

    supported: bool
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TypeError("VerificationResult.supported must be a boolean.")
        if not isinstance(self.reason, str):
            raise TypeError("VerificationResult.reason must be a string.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class RAGResponse:
    """The answer plus the exact retrieved evidence used to produce it."""

    question: str
    answer: str
    citations: Tuple[Citation, ...]
    results: Tuple[SearchResult, ...]
