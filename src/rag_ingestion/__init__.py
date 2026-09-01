"""Public API for the extensible RAG document ingestion package."""

from .builtins import (
    create_default_pipeline,
    load_csv,
    load_html,
    load_json,
    make_text_loader,
    normalize_text,
)
from .errors import (
    DocumentPipelineError,
    DuplicateHandlerError,
    InvalidHandlerResultError,
    MissingDocumentTypeError,
    UnsupportedDocumentTypeError,
)
from .models import Document, DocumentSource
from .pipeline import DocumentPipeline
from .registry import Cleaner, CleanerRegistry, Loader, LoaderRegistry

__all__ = [
    "Cleaner",
    "CleanerRegistry",
    "Document",
    "DocumentPipeline",
    "DocumentPipelineError",
    "DocumentSource",
    "DuplicateHandlerError",
    "InvalidHandlerResultError",
    "Loader",
    "LoaderRegistry",
    "MissingDocumentTypeError",
    "UnsupportedDocumentTypeError",
    "create_default_pipeline",
    "load_csv",
    "load_html",
    "load_json",
    "make_text_loader",
    "normalize_text",
]
