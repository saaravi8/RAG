"""Exceptions raised by the document ingestion pipeline."""

from typing import Iterable


class DocumentPipelineError(Exception):
    """Base class for document pipeline failures."""


class MissingDocumentTypeError(DocumentPipelineError):
    """Raised when a document type cannot be supplied or inferred."""

    def __init__(self) -> None:
        super().__init__(
            "Document type is required when the source name has no file extension."
        )


class UnsupportedDocumentTypeError(DocumentPipelineError):
    """Raised when no loader exists for a document type."""

    def __init__(self, document_type: str, supported: Iterable[str]) -> None:
        supported_text = ", ".join(sorted(supported)) or "none"
        super().__init__(
            "Unsupported document type {!r}. Registered types: {}.".format(
                document_type, supported_text
            )
        )
        self.document_type = document_type


class DuplicateHandlerError(DocumentPipelineError):
    """Raised when a handler would overwrite an existing registration."""

    def __init__(self, kind: str, document_type: str) -> None:
        super().__init__(
            "A {} is already registered for document type {!r}. "
            "Pass replace=True to replace it.".format(kind, document_type)
        )


class InvalidHandlerResultError(DocumentPipelineError, TypeError):
    """Raised when a loader or cleaner violates its output contract."""
