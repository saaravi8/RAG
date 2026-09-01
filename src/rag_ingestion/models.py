"""Canonical input and document models shared by loaders and cleaners."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class DocumentSource:
    """A lazy file source or an in-memory byte source.

    Loaders depend on this abstraction instead of depending directly on a web
    framework, upload object, or filesystem API.
    """

    path: Optional[Path] = None
    data: Optional[bytes] = None
    name: Optional[str] = None
    document_type: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data is None):
            raise ValueError("Exactly one of path or data must be provided.")
        if self.data is not None and not isinstance(self.data, bytes):
            raise TypeError("DocumentSource.data must be bytes.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        document_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DocumentSource":
        source_path = Path(path)
        return cls(
            path=source_path,
            name=source_path.name,
            document_type=document_type,
            metadata=metadata or {},
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        name: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DocumentSource":
        return cls(
            data=data,
            name=name,
            document_type=document_type,
            metadata=metadata or {},
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        name: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        encoding: str = "utf-8",
    ) -> "DocumentSource":
        return cls.from_bytes(
            text.encode(encoding),
            name=name,
            document_type=document_type,
            metadata=metadata,
        )

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        assert self.path is not None
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.read_bytes().decode(encoding)

    def with_document_type(self, document_type: str) -> "DocumentSource":
        return replace(self, document_type=document_type)

    def source_metadata(self) -> Dict[str, Any]:
        """Return user metadata plus useful source provenance."""

        result = dict(self.metadata)
        if self.name is not None:
            result.setdefault("source_name", self.name)
        if self.path is not None:
            result.setdefault("source_path", str(self.path))
        return result


@dataclass(frozen=True)
class Document:
    """The canonical representation passed between pipeline stages."""

    text: str
    document_type: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("Document.text must be a string.")
        if not self.document_type:
            raise ValueError("Document.document_type cannot be empty.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def with_text(self, text: str) -> "Document":
        """Create a transformed document without mutating the input."""

        return replace(self, text=text)

    def with_metadata(self, **updates: Any) -> "Document":
        metadata = dict(self.metadata)
        metadata.update(updates)
        return replace(self, metadata=metadata)
