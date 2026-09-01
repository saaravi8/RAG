"""An offline demo generator; production apps should inject an LLM adapter."""

import re
from typing import List, Sequence, Set, Tuple

from .models import SearchResult


class DemoExtractiveGenerator:
    """Select matching source sentences so the skeleton runs without an API key."""

    _TOKEN = re.compile(r"\w+", re.UNICODE)
    _SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

    def __init__(self, max_sentences: int = 2) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive.")
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[SearchResult]) -> str:
        if not contexts:
            return "I do not have enough indexed context to answer that question."

        query_terms = set(self._TOKEN.findall(question.lower()))
        candidates: List[Tuple[int, float, int, str]] = []
        seen: Set[str] = set()

        for citation_number, result in enumerate(contexts, start=1):
            for sentence in self._SENTENCE.split(result.chunk.text):
                sentence = " ".join(sentence.split())
                if not sentence or sentence in seen:
                    continue
                seen.add(sentence)
                terms = set(self._TOKEN.findall(sentence.lower()))
                overlap = len(query_terms & terms)
                candidates.append(
                    (overlap, result.score, citation_number, sentence)
                )

        if not candidates:
            return "I do not have enough indexed context to answer that question."

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        selected = candidates[: self.max_sentences]
        return " ".join(
            "{} [{}]".format(sentence, citation_number)
            for _, _, citation_number, sentence in selected
        )
