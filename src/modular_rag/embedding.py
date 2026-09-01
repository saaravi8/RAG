"""Dependency-free embeddings for local demos and contract tests."""

import hashlib
import math
import re
from typing import List, Sequence

from .models import Vector


class HashingEmbedder:
    """Deterministic feature hashing; useful for demos, not production quality."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        self.dimensions = dimensions

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        values: List[float] = [0.0] * self.dimensions
        for token in self._TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            values[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude:
            values = [value / magnitude for value in values]
        return tuple(values)
