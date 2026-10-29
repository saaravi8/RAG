"""Canonical snapshots for exact-match retrieval filters."""

from types import MappingProxyType
from typing import Any, Mapping, Optional

from .models import _immutable_value


def snapshot_filters(
    filters: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Validate and detach one exact-match filter mapping from caller state.

    Filter values use the same recursively immutable built-in domain as model
    metadata. This keeps comparisons deterministic and rejects arbitrary custom
    values whose equality behavior could mutate or otherwise drift during a
    query.
    """

    if filters is None:
        return None
    if not isinstance(filters, Mapping):
        raise TypeError("filters must be a mapping when supplied.")
    try:
        items = tuple(filters.items())
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError("filters must expose stable mapping items.") from exc

    snapshot = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("filters must expose key-value pairs.")
        key, value = item
        if type(key) is not str:
            raise TypeError("filter mapping keys must be built-in strings.")
        if key in snapshot:
            raise ValueError("filter mapping keys must be unique.")
        try:
            snapshot[key] = _immutable_value(value)
        except TypeError as exc:
            raise TypeError(
                "filter values must use immutable built-in scalars or supported "
                "built-in containers."
            ) from exc
    return MappingProxyType(snapshot)
