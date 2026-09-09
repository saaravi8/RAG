"""Deterministic identities for index-producing component configurations."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexFingerprint:
    """Canonical index identity plus whether safe unchanged-file reuse is possible."""

    digest: str
    reusable: bool
    components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", dict(self.components))


def build_index_fingerprint(
    embedder: Any,
    chunker: Any,
    document_indexes: Sequence[Any] = (),
    *,
    processor: Any = None,
    schema_version: int = INDEX_SCHEMA_VERSION,
) -> IndexFingerprint:
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version <= 0
    ):
        raise ValueError("schema_version must be a positive integer.")
    embedder_description, embedder_reusable = describe_component(embedder)
    chunker_description, chunker_reusable = describe_component(chunker)
    processor_description, processor_reusable = (
        describe_component(processor) if processor is not None else (None, True)
    )
    index_descriptions = tuple(
        describe_component(document_index) for document_index in document_indexes
    )
    payload = {
        "schema_version": schema_version,
        "processor": processor_description,
        "embedder": embedder_description,
        "chunker": chunker_description,
        "document_indexes": [item[0] for item in index_descriptions],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return IndexFingerprint(
        hashlib.sha256(canonical).hexdigest(),
        embedder_reusable
        and chunker_reusable
        and processor_reusable
        and all(item[1] for item in index_descriptions),
        payload,
    )


def describe_component(component: Any) -> Tuple[Mapping[str, Any], bool]:
    identity = "{}.{}".format(
        type(component).__module__, type(component).__qualname__
    )
    describe = getattr(component, "fingerprint_components", None)
    if not callable(describe):
        return {"type": identity, "opaque": True}, False
    raw = describe()
    if not isinstance(raw, Mapping):
        return {"type": identity, "opaque": True}, False
    normalized, reusable = _normalize(raw)
    description = {"type": identity, "configuration": normalized}
    return description, reusable


def _normalize(value: Any) -> Tuple[Any, bool]:
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": repr(value), "opaque": True}, False
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, True
    if isinstance(value, Mapping):
        normalized = {}
        reusable = True
        for key in sorted(value, key=str):
            selected, selected_reusable = _normalize(value[key])
            normalized_key = str(key)
            if not isinstance(key, str) or normalized_key in normalized:
                reusable = False
            normalized[normalized_key] = selected
            reusable = reusable and selected_reusable
        if normalized.get("opaque") is True:
            reusable = False
        return normalized, reusable
    if isinstance(value, (list, tuple)):
        normalized_items = []
        reusable = True
        for item in value:
            selected, selected_reusable = _normalize(item)
            normalized_items.append(selected)
            reusable = reusable and selected_reusable
        return normalized_items, reusable
    return {
        "type": "{}.{}".format(type(value).__module__, type(value).__qualname__),
        "opaque": True,
    }, False
