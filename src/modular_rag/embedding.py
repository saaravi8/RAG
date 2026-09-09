"""Embedding adapters for local demos and semantic retrieval."""

import hashlib
import importlib.metadata
import math
import os
import re
from typing import Any, List, Optional, Sequence

from .errors import ComponentContractError, OptionalDependencyError
from .models import Vector


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("{} must be a positive integer.".format(name))
    if value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return value


class HashingEmbedder:
    """Deterministic feature hashing; useful for demos, not production quality."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = _positive_int(dimensions, "dimensions")

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self._embed(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def fingerprint_components(self):
        return {
            "algorithm": "blake2b-feature-hashing",
            "algorithm_version": 1,
            "dimensions": self.dimensions,
            "normalization": "l2",
        }

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


class SentenceTransformerEmbedder:
    """Dense local embeddings backed by a Sentence Transformers model.

    The defaults are configured for ``intfloat/multilingual-e5-base``. E5 was
    trained with asymmetric retrieval prefixes, so documents and queries must
    be encoded differently. A different model can be selected by overriding
    the model name and prefixes.

    Pass an already loaded ``model`` to share it across components or tests.
    The object only needs a Sentence Transformers-compatible ``encode`` method.
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 32,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        max_seq_length: Optional[int] = None,
        show_progress_bar: bool = False,
        revision: Optional[str] = None,
        cache_folder: Optional[str] = None,
        local_files_only: bool = False,
        model: Optional[Any] = None,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if (
            not isinstance(query_prefix, str)
            or not isinstance(document_prefix, str)
        ):
            raise TypeError("query_prefix and document_prefix must be strings.")
        if revision is not None and not isinstance(revision, str):
            raise TypeError("revision must be a string when supplied.")
        if revision is not None and not revision.strip():
            raise ValueError("revision cannot be empty when supplied.")
        batch_size = _positive_int(batch_size, "batch_size")
        if max_seq_length is not None:
            max_seq_length = _positive_int(max_seq_length, "max_seq_length")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.show_progress_bar = show_progress_bar
        self.revision = revision
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only
        self._injected_model = model is not None
        self._model = model if model is not None else self._load_model()

        if not callable(getattr(self._model, "encode", None)):
            raise TypeError("model must provide a callable encode method.")

        model_limit = getattr(self._model, "max_seq_length", None)
        if (
            max_seq_length is not None
            and isinstance(model_limit, int)
            and model_limit > 0
            and max_seq_length > model_limit
        ):
            raise ValueError(
                "max_seq_length {} exceeds the model limit of {}.".format(
                    max_seq_length, model_limit
                )
            )
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length

    @property
    def dimensions(self) -> Optional[int]:
        """Return the model's output size when the backend exposes it."""

        get_dimensions = getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if not callable(get_dimensions):
            return None
        value = get_dimensions()
        return int(value) if value is not None else None

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        texts = tuple(texts)
        if not texts:
            return ()
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vector:
        vectors = self._encode([self.query_prefix + text])
        return vectors[0]

    def fingerprint_components(self):
        pinned_revision = bool(
            self.revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision)
        )
        local_model_path = os.path.exists(os.path.expanduser(self.model_name))
        try:
            provider_version = importlib.metadata.version("sentence-transformers")
        except importlib.metadata.PackageNotFoundError:
            provider_version = "unknown"
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "provider_version": provider_version,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "dimensions": self.dimensions,
            "max_seq_length": getattr(self._model, "max_seq_length", None),
            "injected_model": self._injected_model,
            "local_model_path": local_model_path,
            "revision_is_immutable": pinned_revision,
            "opaque": (
                self._injected_model or local_model_path or not pinned_revision
            ),
        }

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install it with: pip install 'modular-rag[embeddings]'"
            ) from exc
        return SentenceTransformer(
            self.model_name,
            device=self.device,
            revision=self.revision,
            cache_folder=self.cache_folder,
            local_files_only=self.local_files_only,
        )

    def _encode(self, texts: Sequence[str]) -> Sequence[Vector]:
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        if isinstance(raw_vectors, (str, bytes, bytearray)):
            raise ComponentContractError(
                "Sentence Transformers returned malformed embeddings."
            )
        try:
            rows = tuple(raw_vectors)
        except TypeError as exc:
            raise ComponentContractError(
                "Sentence Transformers returned malformed embeddings."
            ) from exc
        vectors = []
        for row in rows:
            if isinstance(row, (str, bytes, bytearray)):
                raise ComponentContractError(
                    "Sentence Transformers returned malformed embeddings."
                )
            try:
                coordinates = tuple(row)
            except TypeError as exc:
                raise ComponentContractError(
                    "Sentence Transformers returned malformed embeddings."
                ) from exc
            vector = []
            for value in coordinates:
                if isinstance(value, (bool, str, bytes, bytearray)):
                    raise ComponentContractError(
                        "Sentence Transformers returned malformed embeddings."
                    )
                try:
                    vector.append(float(value))
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ComponentContractError(
                        "Sentence Transformers returned malformed embeddings."
                    ) from exc
            vectors.append(tuple(vector))
        normalized_vectors = tuple(vectors)
        if len(normalized_vectors) != len(texts) or any(
            not vector for vector in normalized_vectors
        ):
            raise ComponentContractError(
                "Sentence Transformers returned {} vectors for {} texts.".format(
                    len(normalized_vectors), len(texts)
                )
            )
        dimensions = {len(vector) for vector in normalized_vectors}
        if len(dimensions) != 1:
            raise ComponentContractError(
                "Sentence Transformers returned mixed vector dimensions."
            )
        if any(
            not math.isfinite(value)
            for vector in normalized_vectors
            for value in vector
        ):
            raise ComponentContractError(
                "Sentence Transformers returned non-finite embedding values."
            )
        return normalized_vectors
