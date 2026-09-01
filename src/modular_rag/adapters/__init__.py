"""Built-in adapters. Add provider-specific implementations beside these."""

from .memory import InMemoryVectorStore
from .spacy_sentences import SpacySentenceSegmenter

__all__ = ["InMemoryVectorStore", "SpacySentenceSegmenter"]
