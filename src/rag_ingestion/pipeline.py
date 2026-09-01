"""Document ingestion orchestration with injected handler registries."""

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from .errors import InvalidHandlerResultError, MissingDocumentTypeError
from .models import Document, DocumentSource
from .registry import Cleaner, CleanerRegistry, Loader, LoaderRegistry, normalize_document_type

SourceLike = Union[str, Path, bytes, DocumentSource]


class DocumentPipeline:
    """Load and clean documents without coupling handlers to each other."""

    def __init__(
        self,
        loaders: Optional[LoaderRegistry] = None,
        cleaners: Optional[CleanerRegistry] = None,
    ) -> None:
        self.loaders = loaders if loaders is not None else LoaderRegistry()
        self.cleaners = cleaners if cleaners is not None else CleanerRegistry()

    @property
    def supported_document_types(self):
        """Document types currently supported by registered loaders."""

        return self.loaders.supported_types

    def register_loader(
        self,
        document_type: str,
        loader: Optional[Loader] = None,
        *,
        replace: bool = False,
    ):
        return self.loaders.register(document_type, loader, replace=replace)

    def register_cleaner(
        self,
        document_type: str,
        cleaner: Optional[Cleaner] = None,
        *,
        replace: bool = False,
        prepend: bool = False,
    ):
        return self.cleaners.register(
            document_type, cleaner, replace=replace, prepend=prepend
        )

    def load(
        self,
        source: SourceLike,
        *,
        document_type: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Document:
        normalized_source = self._coerce_source(
            source, name=name, document_type=document_type, metadata=metadata
        )
        resolved_type = self._resolve_document_type(normalized_source, document_type)
        typed_source = normalized_source.with_document_type(resolved_type)
        loader = self.loaders.get(resolved_type)
        document = loader(typed_source)

        self._validate_result(document, "Loader", resolved_type)
        actual_type = normalize_document_type(document.document_type)
        if actual_type != resolved_type:
            raise InvalidHandlerResultError(
                "Loader for {!r} returned a document of type {!r}.".format(
                    resolved_type, actual_type
                )
            )

        combined_metadata = typed_source.source_metadata()
        combined_metadata.update(document.metadata)
        return Document(
            text=document.text,
            document_type=resolved_type,
            metadata=combined_metadata,
        )

    def clean(self, document: Document) -> Document:
        document_type = normalize_document_type(document.document_type)
        current = document
        for cleaner in self.cleaners.get(document_type):
            current = cleaner(current)
            self._validate_result(current, "Cleaner", document_type)
            actual_type = normalize_document_type(current.document_type)
            if actual_type != document_type:
                raise InvalidHandlerResultError(
                    "Cleaner for {!r} changed the document type to {!r}.".format(
                        document_type, actual_type
                    )
                )
        return current

    def process(
        self,
        source: SourceLike,
        *,
        document_type: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Document:
        """Run the complete load-then-clean pipeline."""

        document = self.load(
            source,
            document_type=document_type,
            name=name,
            metadata=metadata,
        )
        return self.clean(document)

    @staticmethod
    def _validate_result(result: object, stage: str, document_type: str) -> None:
        if not isinstance(result, Document):
            raise InvalidHandlerResultError(
                "{} for {!r} must return Document, got {}.".format(
                    stage, document_type, type(result).__name__
                )
            )

    @staticmethod
    def _coerce_source(
        source: SourceLike,
        *,
        name: Optional[str],
        document_type: Optional[str],
        metadata: Optional[Mapping[str, Any]],
    ) -> DocumentSource:
        if isinstance(source, DocumentSource):
            if name is not None or metadata is not None:
                raise ValueError(
                    "name and metadata must be set on DocumentSource directly."
                )
            return source
        if isinstance(source, bytes):
            return DocumentSource.from_bytes(
                source,
                name=name,
                document_type=document_type,
                metadata=metadata,
            )
        if isinstance(source, (str, Path)):
            if name is not None:
                raise ValueError("name cannot be used with a path source.")
            return DocumentSource.from_path(
                source, document_type=document_type, metadata=metadata
            )
        raise TypeError(
            "source must be a path, bytes, or DocumentSource; got {}.".format(
                type(source).__name__
            )
        )

    @staticmethod
    def _resolve_document_type(
        source: DocumentSource, explicit_type: Optional[str]
    ) -> str:
        candidate = explicit_type or source.document_type
        if candidate:
            return normalize_document_type(candidate)

        if source.name:
            suffix = Path(source.name).suffix
            if suffix:
                return normalize_document_type(suffix)
        raise MissingDocumentTypeError()
