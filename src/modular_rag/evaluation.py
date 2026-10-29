"""Deterministic document-retrieval and answer-safety evaluation primitives."""

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
    """Ranked document retrieval and answer outcome observed for one case.

    ``answered`` and ``checked`` are deliberately separate concepts. A returned
    answer may be unchecked when verification is disabled, while a checked
    answer may be suppressed by an enforcement policy. ``verification_status``
    is normalized to ``"not_checked"``, ``"checked"``, or ``"error"``.
    Existing four-argument construction remains valid and infers the status from
    ``answer_supported``.
    """

    case_id: str
    retrieved_document_ids: Tuple[str, ...]
    abstained: bool
    answer_supported: Optional[bool] = None
    abstention_reason: Optional[str] = None
    answer_generated: Optional[bool] = None
    verification_status: Optional[str] = None

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

        reason = self.abstention_reason
        if reason is not None and not isinstance(reason, str):
            raise TypeError("abstention_reason must be a string or None.")
        if isinstance(reason, str) and not reason.strip():
            raise ValueError("abstention_reason cannot be empty.")
        if not self.abstained and reason is not None:
            raise ValueError(
                "A non-abstained observation cannot have an abstention reason."
            )

        generated = self.answer_generated
        if generated is None:
            generated = not self.abstained or reason in {
                "unsupported_answer",
                "verification_error",
            }
        if not isinstance(generated, bool):
            raise TypeError("answer_generated must be a boolean or None.")
        if not self.abstained and not generated:
            raise ValueError("A returned answer must be marked as generated.")
        if self.abstained and generated and reason is None:
            raise ValueError(
                "A post-generation abstention requires an abstention reason."
            )

        verification_status = self.verification_status
        if verification_status is None:
            if reason == "verification_error":
                verification_status = "error"
            elif self.answer_supported is not None:
                verification_status = "checked"
            else:
                verification_status = "not_checked"
        if not isinstance(verification_status, str):
            raise TypeError("verification_status must be a string or None.")
        if verification_status not in {"not_checked", "checked", "error"}:
            raise ValueError(
                "verification_status must be 'not_checked', 'checked', or 'error'."
            )
        if verification_status == "checked" and self.answer_supported is None:
            raise ValueError("A checked answer requires an answer-support label.")
        if verification_status != "checked" and self.answer_supported is not None:
            raise ValueError(
                "Only a checked answer can carry an answer-support label."
            )
        if verification_status == "checked" and not generated:
            raise ValueError("Only a generated answer can be checked.")
        if verification_status == "error":
            if (
                not self.abstained
                or not generated
                or reason != "verification_error"
            ):
                raise ValueError(
                    "A verification error must be a post-generation "
                    "'verification_error' abstention."
                )
        elif reason == "verification_error":
            raise ValueError(
                "A 'verification_error' abstention requires verification_status='error'."
            )
        if self.abstained and verification_status == "checked":
            if self.answer_supported is not False:
                raise ValueError("An abstained checked answer must be unsupported.")
            if reason != "unsupported_answer":
                raise ValueError(
                    "An enforced unsupported answer requires "
                    "abstention_reason='unsupported_answer'."
                )

        object.__setattr__(self, "retrieved_document_ids", retrieved)
        object.__setattr__(self, "answer_generated", generated)
        object.__setattr__(self, "verification_status", verification_status)


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate document-retrieval and answer-state metrics.

    ``answered_count`` counts answers returned to callers. ``generated_answer_count``
    also includes generated answers later suppressed by policy. A checked answer
    has a valid support verdict, including an enforced unsupported verdict;
    verifier failures are counted separately and never become support labels.
    """

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
    generated_answer_count: int = 0
    pre_generation_abstention_count: int = 0
    post_generation_abstention_count: int = 0
    verification_error_count: int = 0


def evaluate_observations(
    cases: Sequence[EvaluationCase],
    observations: Sequence[EvaluationObservation],
    *,
    top_k: int,
) -> EvaluationReport:
    """Calculate document-retrieval and answer-safety metrics from observations."""

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
    generated_answers = 0
    pre_generation_abstentions = 0
    post_generation_abstentions = 0
    verification_errors = 0

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
            if observation.answer_generated:
                post_generation_abstentions += 1
            else:
                pre_generation_abstentions += 1
        if observation.answer_generated:
            generated_answers += 1
        if observation.verification_status == "checked":
            checked_answers += 1
            if not observation.answer_supported:
                unsupported_answers += 1
        elif observation.verification_status == "error":
            verification_errors += 1

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
        generated_answer_count=generated_answers,
        pre_generation_abstention_count=pre_generation_abstentions,
        post_generation_abstention_count=post_generation_abstentions,
        verification_error_count=verification_errors,
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
