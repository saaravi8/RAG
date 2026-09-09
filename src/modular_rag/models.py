"""Provider-neutral domain models used by the RAG core."""

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

Vector = Tuple[float, ...]
_IMMUTABLE_METADATA_SCALARS = (type(None), bool, int, float, str, bytes)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError("{} must be a string.".format(name))
    if not value.strip():
        raise ValueError("{} cannot be empty.".format(name))
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("{} must be an integer.".format(name))
    return value


def _immutable_value(value: Any) -> Any:
    """Return a metadata value in the supported immutable built-in domain.

    Metadata accepts ``None``, booleans, integers, floats, strings, bytes,
    mappings with string keys, lists/tuples, and sets/frozensets. Mutable byte
    buffers are snapshotted as bytes. Containers are recursively converted to
    read-only mappings, tuples, or frozensets; arbitrary custom values are
    rejected because copying them cannot guarantee isolation.
    """

    if isinstance(value, Mapping):
        snapshot = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("metadata mapping keys must be built-in strings.")
            snapshot[key] = _immutable_value(item)
        return MappingProxyType(snapshot)
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_value(item) for item in value)
    if isinstance(value, (bytearray, memoryview)):
        try:
            return bytes(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata byte buffers must be readable.") from exc
    if type(value) in _IMMUTABLE_METADATA_SCALARS:
        return value
    raise TypeError(
        "metadata values must use immutable built-in scalars or supported "
        "built-in containers."
    )


def _immutable_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Snapshot top-level metadata under :func:`_immutable_value`'s domain."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    return _immutable_value(metadata)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError("{} must be a finite number, not a boolean.".format(name))
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("{} must be a finite number.".format(name))
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a finite number.".format(name)) from exc
    if not math.isfinite(normalized):
        raise ValueError("{} must be finite.".format(name))
    return normalized


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
        _nonempty_string(self.id, "SentenceSpan.id")
        _nonempty_string(self.document_id, "SentenceSpan.document_id")
        _nonempty_string(self.text, "SentenceSpan.text")
        index = _integer(self.index, "SentenceSpan.index")
        start_char = _integer(self.start_char, "SentenceSpan.start_char")
        end_char = _integer(self.end_char, "SentenceSpan.end_char")
        if index < 0:
            raise ValueError("SentenceSpan.index cannot be negative.")
        if start_char < 0:
            raise ValueError("SentenceSpan.start_char cannot be negative.")
        if end_char <= start_char:
            raise ValueError("SentenceSpan.end_char must be after start_char.")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a source document."""

    id: str
    document_id: str
    text: str
    index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string(self.id, "Chunk.id")
        _nonempty_string(self.document_id, "Chunk.document_id")
        _nonempty_string(self.text, "Chunk.text")
        index = _integer(self.index, "Chunk.index")
        if index < 0:
            raise ValueError("Chunk.index cannot be negative.")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class VectorRecord:
    """A chunk paired with the vector used to retrieve it."""

    chunk: Chunk
    vector: Vector

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise TypeError("VectorRecord.chunk must be a Chunk.")
        if isinstance(self.vector, (str, bytes, bytearray)):
            raise TypeError("VectorRecord.vector must be a numeric sequence.")
        try:
            values = tuple(self.vector)
        except TypeError as exc:
            raise TypeError("VectorRecord.vector must be a numeric sequence.") from exc
        if not values:
            raise ValueError("VectorRecord.vector cannot be empty.")
        object.__setattr__(
            self,
            "vector",
            tuple(
                _finite_float(value, "VectorRecord.vector coordinate")
                for value in values
            ),
        )


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its latest retrieval or reranking score."""

    chunk: Chunk
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise TypeError("SearchResult.chunk must be a Chunk.")
        object.__setattr__(self, "score", _finite_float(self.score, "SearchResult.score"))


@dataclass(frozen=True)
class IndexReport:
    """Summary returned after a document has been indexed."""

    document_id: str
    chunk_count: int

    def __post_init__(self) -> None:
        _nonempty_string(self.document_id, "IndexReport.document_id")
        chunk_count = _integer(self.chunk_count, "IndexReport.chunk_count")
        if chunk_count < 0:
            raise ValueError("IndexReport.chunk_count cannot be negative.")


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
        number = _integer(self.number, "Citation.number")
        if number <= 0:
            raise ValueError("Citation.number must be positive.")
        _nonempty_string(self.chunk_id, "Citation.chunk_id")
        _nonempty_string(self.document_id, "Citation.document_id")
        _nonempty_string(self.source, "Citation.source")
        _nonempty_string(self.excerpt, "Citation.excerpt")
        object.__setattr__(self, "score", _finite_float(self.score, "Citation.score"))
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


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
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True)
class RAGResponse:
    """The answer plus the exact retrieved evidence used to produce it."""

    question: str
    answer: str
    citations: Tuple[Citation, ...]
    results: Tuple[SearchResult, ...]
    abstained: bool = False
    abstention_reason: Optional[str] = None
    verification: Optional[VerificationResult] = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, str):
            raise TypeError("RAGResponse.question must be a string.")
        if not self.question.strip():
            raise ValueError("RAGResponse.question cannot be empty.")
        if not isinstance(self.answer, str):
            raise TypeError("RAGResponse.answer must be a string.")
        if not self.answer.strip():
            raise ValueError("RAGResponse.answer cannot be empty.")

        try:
            citations = tuple(self.citations)
        except TypeError as exc:
            raise TypeError("RAGResponse.citations must be an iterable.") from exc
        try:
            results = tuple(self.results)
        except TypeError as exc:
            raise TypeError("RAGResponse.results must be an iterable.") from exc
        if any(not isinstance(citation, Citation) for citation in citations):
            raise TypeError(
                "RAGResponse.citations must contain only Citation instances."
            )
        if any(not isinstance(result, SearchResult) for result in results):
            raise TypeError(
                "RAGResponse.results must contain only SearchResult instances."
            )
        if len(citations) != len(results):
            raise ValueError(
                "RAGResponse citations must align one-to-one with results."
            )
        for number, (citation, result) in enumerate(
            zip(citations, results), start=1
        ):
            if (
                citation.number != number
                or citation.chunk_id != result.chunk.id
                or citation.document_id != result.chunk.document_id
                or citation.score != result.score
            ):
                raise ValueError(
                    "RAGResponse citations must align with result order, "
                    "identity, and score."
                )
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "results", results)

        if not isinstance(self.abstained, bool):
            raise TypeError("RAGResponse.abstained must be a boolean.")
        if self.abstained:
            if not isinstance(self.abstention_reason, str):
                raise TypeError(
                    "An abstained RAGResponse requires a string reason."
                )
            if not self.abstention_reason.strip():
                raise ValueError(
                    "An abstained RAGResponse requires a nonempty reason."
                )
        elif self.abstention_reason is not None:
            raise ValueError(
                "A non-abstained RAGResponse cannot have an abstention reason."
            )
        if self.verification is not None and not isinstance(
            self.verification, VerificationResult
        ):
            raise TypeError(
                "RAGResponse.verification must be a VerificationResult or None."
            )
        if self.verification is not None:
            object.__setattr__(
                self,
                "verification",
                VerificationResult(
                    self.verification.supported,
                    self.verification.reason,
                    self.verification.metadata,
                ),
            )
