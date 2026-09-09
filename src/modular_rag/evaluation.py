"""Deterministic retrieval and abstention evaluation primitives."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class EvaluationCase:
    """Expected relevant document IDs for one evaluation question."""

    case_id: str
    relevant_document_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("EvaluationCase.case_id cannot be empty.")
        if isinstance(
            self.relevant_document_ids, (str, bytes, bytearray)
        ) or not isinstance(self.relevant_document_ids, Sequence):
            raise TypeError("Relevant document IDs must be a sequence of strings.")
        relevant = tuple(self.relevant_document_ids)
        if any(not isinstance(value, str) or not value.strip() for value in relevant):
            raise ValueError("Relevant document IDs must be nonempty strings.")
        if len(set(relevant)) != len(relevant):
            raise ValueError("Relevant document IDs cannot contain duplicates.")
        object.__setattr__(self, "relevant_document_ids", relevant)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_document_ids)


@dataclass(frozen=True)
class EvaluationObservation:
    """Ranked retrieval and answer outcome observed for one case."""

    case_id: str
    retrieved_document_ids: Tuple[str, ...]
    abstained: bool
    answer_supported: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("EvaluationObservation.case_id cannot be empty.")
        if isinstance(
            self.retrieved_document_ids, (str, bytes, bytearray)
        ) or not isinstance(self.retrieved_document_ids, Sequence):
            raise TypeError("Retrieved document IDs must be a sequence of strings.")
        retrieved = tuple(self.retrieved_document_ids)
        if any(not isinstance(value, str) or not value.strip() for value in retrieved):
            raise ValueError("Retrieved document IDs must be nonempty strings.")
        if len(set(retrieved)) != len(retrieved):
            raise ValueError("Retrieved document IDs cannot contain duplicates.")
        if not isinstance(self.abstained, bool):
            raise TypeError("EvaluationObservation.abstained must be a boolean.")
        if self.answer_supported is not None and not isinstance(
            self.answer_supported, bool
        ):
            raise TypeError("answer_supported must be a boolean or None.")
        if self.abstained and self.answer_supported is not None:
            raise ValueError("An abstained observation cannot label answer support.")
        if not self.abstained and self.answer_supported is None:
            raise ValueError(
                "A non-abstained observation requires an answer-support label."
            )
        object.__setattr__(self, "retrieved_document_ids", retrieved)


@dataclass(frozen=True)
class EvaluationReport:
    case_count: int
    answerable_count: int
    unanswerable_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    abstention_precision: float
    abstention_recall: float
    unsupported_answer_rate: float
    answered_count: int
    checked_answer_count: int


def evaluate_observations(
    cases: Sequence[EvaluationCase],
    observations: Sequence[EvaluationObservation],
    *,
    top_k: int,
) -> EvaluationReport:
    """Calculate retrieval and answer-safety metrics from fixed observations."""

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    cases = _snapshot_sequence(cases, "cases")
    observations = _snapshot_sequence(observations, "observations")
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    case_by_id = _unique_by_id(cases, "evaluation case")
    observation_by_id = _unique_by_id(observations, "evaluation observation")
    if set(case_by_id) != set(observation_by_id):
        raise ValueError("Cases and observations must have exactly matching IDs.")

    recalls = []
    reciprocal_ranks = []
    ndcgs = []
    true_abstentions = 0
    predicted_abstentions = 0
    unanswerable_count = 0
    unsupported_answers = 0
    checked_answers = 0

    for case_id, case in case_by_id.items():
        observation = observation_by_id[case_id]
        ranked = _deduplicate(observation.retrieved_document_ids)[:top_k]
        relevant = set(case.relevant_document_ids)
        if case.answerable:
            hits = sum(document_id in relevant for document_id in ranked)
            recalls.append(hits / len(relevant))
            first_rank = next(
                (
                    rank
                    for rank, document_id in enumerate(ranked, start=1)
                    if document_id in relevant
                ),
                None,
            )
            reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, document_id in enumerate(ranked, start=1)
                if document_id in relevant
            )
            ideal_hits = min(len(relevant), top_k)
            ideal_dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, ideal_hits + 1)
            )
            ndcgs.append(dcg / ideal_dcg)
        else:
            unanswerable_count += 1
            if observation.abstained:
                true_abstentions += 1

        if observation.abstained:
            predicted_abstentions += 1
        elif observation.answer_supported is not None:
            checked_answers += 1
            if not observation.answer_supported:
                unsupported_answers += 1

    return EvaluationReport(
        case_count=len(case_by_id),
        answerable_count=len(case_by_id) - unanswerable_count,
        unanswerable_count=unanswerable_count,
        recall_at_k=_mean(recalls),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
        ndcg_at_k=_mean(ndcgs),
        abstention_precision=_ratio(true_abstentions, predicted_abstentions),
        abstention_recall=_ratio(true_abstentions, unanswerable_count),
        unsupported_answer_rate=_ratio(unsupported_answers, checked_answers),
        answered_count=sum(
            not observation.abstained for observation in observation_by_id.values()
        ),
        checked_answer_count=checked_answers,
    )


def _snapshot_sequence(items: Sequence[object], label: str) -> Tuple[object, ...]:
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
        raise TypeError("{} must be a non-text sequence.".format(label))
    return tuple(items)


def _unique_by_id(items: Sequence[object], label: str) -> Dict[str, object]:
    selected: Dict[str, object] = {}
    for item in items:
        case_id = getattr(item, "case_id", None)
        if case_id in selected:
            raise ValueError("Duplicate {} ID: {!r}.".format(label, case_id))
        selected[case_id] = item
    return selected


def _deduplicate(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
