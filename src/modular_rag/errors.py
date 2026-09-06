"""Exceptions raised when a RAG component violates its contract."""


class RAGError(Exception):
    """Base class for modular RAG errors."""


class ComponentContractError(RAGError, TypeError):
    """Raised when replaceable components return incompatible data."""


class OptionalDependencyError(RAGError, ImportError):
    """Raised when an adapter's optional third-party dependency is absent."""


class VectorDimensionError(RAGError, ValueError):
    """Raised when vector dimensions are inconsistent."""


class RepositoryIngestionError(RAGError, ValueError):
    """Base error for an unsafe or invalid repository ingestion request."""


class UnsafeRepositoryError(RepositoryIngestionError):
    """Raised when a repository root or path violates the containment policy."""


class RepositoryResourceLimitError(RepositoryIngestionError):
    """Raised before indexing work would exceed a configured resource limit."""
