"""Thread-safe registries for pluggable loader and cleaner functions."""

from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .errors import DuplicateHandlerError, UnsupportedDocumentTypeError
from .models import Document, DocumentSource

Loader = Callable[[DocumentSource], Document]
Cleaner = Callable[[Document], Document]


def _handler_description(handler: Callable[..., Document]) -> Mapping[str, Any]:
    identity = "{}.{}".format(
        getattr(handler, "__module__", type(handler).__module__),
        getattr(handler, "__qualname__", type(handler).__qualname__),
    )
    describe = getattr(handler, "fingerprint_components", None)
    if not callable(describe):
        return {"type": identity, "opaque": True}
    configuration = describe()
    if not isinstance(configuration, Mapping):
        return {"type": identity, "opaque": True}
    return {"type": identity, "configuration": dict(configuration)}


def normalize_document_type(document_type: str) -> str:
    """Normalize values such as ``.PDF`` and `` pdf `` to ``pdf``."""

    if not isinstance(document_type, str):
        raise TypeError("document_type must be a string.")
    normalized = document_type.strip().lower().lstrip(".")
    if not normalized:
        raise ValueError("document_type cannot be empty.")
    return normalized


class LoaderRegistry:
    """Maps one document type to one loader function."""

    def __init__(self) -> None:
        self._handlers: Dict[str, Loader] = {}
        self._lock = RLock()

    @property
    def supported_types(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._handlers))

    def register(
        self,
        document_type: str,
        loader: Optional[Loader] = None,
        *,
        replace: bool = False,
    ):
        """Register directly or use as ``@registry.register("pdf")``."""

        normalized = normalize_document_type(document_type)

        def add(handler: Loader) -> Loader:
            if not callable(handler):
                raise TypeError("loader must be callable.")
            with self._lock:
                if normalized in self._handlers and not replace:
                    raise DuplicateHandlerError("loader", normalized)
                self._handlers[normalized] = handler
            return handler

        if loader is None:
            return add
        return add(loader)

    def unregister(self, document_type: str) -> None:
        normalized = normalize_document_type(document_type)
        with self._lock:
            self._handlers.pop(normalized, None)

    def get(self, document_type: str) -> Loader:
        normalized = normalize_document_type(document_type)
        with self._lock:
            try:
                return self._handlers[normalized]
            except KeyError:
                raise UnsupportedDocumentTypeError(
                    normalized, self._handlers.keys()
                ) from None

    def fingerprint_components(self):
        with self._lock:
            return {
                document_type: _handler_description(handler)
                for document_type, handler in sorted(self._handlers.items())
            }


class CleanerRegistry:
    """Maps a document type to an ordered chain of cleaner functions."""

    ALL_TYPES = "*"

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Cleaner]] = {}
        self._lock = RLock()

    @property
    def supported_types(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(key for key in self._handlers if key != self.ALL_TYPES))

    def register(
        self,
        document_type: str,
        cleaner: Optional[Cleaner] = None,
        *,
        replace: bool = False,
        prepend: bool = False,
    ):
        """Add a cleaner to a type's chain.

        Register against ``"*"`` to run a cleaner before every type-specific
        cleaner. Passing ``replace=True`` replaces the whole chain for the type.
        """

        normalized = (
            self.ALL_TYPES
            if document_type == self.ALL_TYPES
            else normalize_document_type(document_type)
        )

        def add(handler: Cleaner) -> Cleaner:
            if not callable(handler):
                raise TypeError("cleaner must be callable.")
            with self._lock:
                if replace:
                    self._handlers[normalized] = [handler]
                elif prepend:
                    self._handlers.setdefault(normalized, []).insert(0, handler)
                else:
                    self._handlers.setdefault(normalized, []).append(handler)
            return handler

        if cleaner is None:
            return add
        return add(cleaner)

    def unregister(self, document_type: str) -> None:
        normalized = (
            self.ALL_TYPES
            if document_type == self.ALL_TYPES
            else normalize_document_type(document_type)
        )
        with self._lock:
            self._handlers.pop(normalized, None)

    def get(self, document_type: str) -> Tuple[Cleaner, ...]:
        normalized = normalize_document_type(document_type)
        with self._lock:
            return (
                tuple(self._handlers.get(self.ALL_TYPES, ()))
                + tuple(self._handlers.get(normalized, ()))
            )

    def fingerprint_components(self):
        with self._lock:
            return {
                document_type: tuple(
                    _handler_description(handler) for handler in handlers
                )
                for document_type, handlers in sorted(self._handlers.items())
            }
