"""Policies for deciding which retrieved results are safe to use as evidence."""

import math
from typing import Mapping, Optional, Protocol, Sequence

from .models import SearchResult


class RelevancePolicy(Protocol):
    """Select retrieved evidence using score semantics appropriate to its backend."""

    def select_relevant(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        method: str,
    ) -> Sequence[SearchResult]:
        ...


class ScoreThresholdRelevancePolicy:
    """Keep finite results whose score is strictly above a configured floor.

    Score ranges vary across retrieval backends. The selected query method is
    therefore available for per-method calibration, while the default zero
    floor matches the cosine-style retrieval scores used by the local demo.
    """

    def __init__(
        self,
        minimum_score: float = 0.0,
        *,
        method_minimum_scores: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.minimum_score = self._validate_score(minimum_score, "minimum_score")
        normalized_scores = {}
        selected_scores = (
            {} if method_minimum_scores is None else method_minimum_scores
        )
        for method, score in selected_scores.items():
            normalized_method = self._normalize_method(method)
            if normalized_method in normalized_scores:
                raise ValueError(
                    "Duplicate relevance threshold for method {!r}.".format(
                        normalized_method
                    )
                )
            normalized_scores[normalized_method] = self._validate_score(
                score,
                "method_minimum_scores[{!r}]".format(method),
            )
        self.method_minimum_scores = normalized_scores

    def select_relevant(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        method: str,
    ) -> Sequence[SearchResult]:
        del question
        threshold = self.method_minimum_scores.get(
            self._normalize_method(method), self.minimum_score
        )
        return tuple(
            result
            for result in results
            if math.isfinite(result.score) and result.score > threshold
        )

    @staticmethod
    def _validate_score(score: float, name: str) -> float:
        if isinstance(score, bool):
            raise TypeError("{} must be a finite number.".format(name))
        try:
            normalized = float(score)
        except (TypeError, ValueError) as exc:
            raise TypeError("{} must be a finite number.".format(name)) from exc
        if not math.isfinite(normalized):
            raise ValueError("{} must be finite.".format(name))
        return normalized

    @staticmethod
    def _normalize_method(method: str) -> str:
        if not isinstance(method, str):
            raise TypeError("Relevance method names must be strings.")
        normalized = method.strip().lower()
        if not normalized:
            raise ValueError("Relevance method names cannot be empty.")
        return normalized
