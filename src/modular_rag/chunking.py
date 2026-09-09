"""A simple default chunker; replace it for structure-aware chunking."""

import hashlib
import re
from typing import List, Sequence

from rag_ingestion import Document

from .models import Chunk


class WordWindowChunker:
    """Split text into overlapping word windows while preserving source text."""

    _WORD = re.compile(r"\S+")

    def __init__(self, max_words: int = 200, overlap_words: int = 30) -> None:
        if isinstance(max_words, bool) or not isinstance(max_words, int):
            raise TypeError("max_words must be an integer.")
        if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
            raise TypeError("overlap_words must be an integer.")
        if max_words <= 0:
            raise ValueError("max_words must be positive.")
        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must be between 0 and max_words - 1.")
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, document: Document) -> Sequence[Chunk]:
        matches = list(self._WORD.finditer(document.text))
        if not matches:
            return ()

        document_id = self._document_id(document)
        step = self.max_words - self.overlap_words
        chunks: List[Chunk] = []

        for chunk_index, word_start in enumerate(range(0, len(matches), step)):
            word_end = min(word_start + self.max_words, len(matches))
            start_offset = matches[word_start].start()
            end_offset = matches[word_end - 1].end()
            text = document.text[start_offset:end_offset].strip()
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "document_type": document.document_type,
                    "word_start": word_start,
                    "word_end": word_end,
                }
            )
            chunks.append(
                Chunk(
                    id="{}:{}:{}".format(document_id, chunk_index, digest),
                    document_id=document_id,
                    text=text,
                    index=chunk_index,
                    metadata=metadata,
                )
            )
            if word_end == len(matches):
                break

        return tuple(chunks)

    def fingerprint_components(self):
        return {
            "algorithm": "word-window",
            "algorithm_version": 1,
            "max_words": self.max_words,
            "overlap_words": self.overlap_words,
        }

    @staticmethod
    def _document_id(document: Document) -> str:
        explicit = document.metadata.get("document_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit)

        identity = (
            document.metadata.get("source_path")
            or document.metadata.get("source_name")
            or document.text
        )
        digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:20]
        return "doc_{}".format(digest)
