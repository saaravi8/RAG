"""Small in-process coordination primitive for index state transitions."""

from contextlib import contextmanager
from threading import RLock
from typing import Iterator


class IndexTransactionCoordinator:
    """Serialize coordinated index writes with reads using the same instance.

    The coordinator intentionally uses one re-entrant lock rather than a more
    complicated reader/writer implementation. Index writes are comparatively
    rare in the local in-memory application, and a single lock makes the
    visibility guarantee easy to audit: participating readers observe the
    complete state before or after a transaction, never its intermediate
    component-local commits.
    """

    def __init__(self) -> None:
        self._lock = RLock()

    @contextmanager
    def synchronized(self) -> Iterator[None]:
        with self._lock:
            yield
