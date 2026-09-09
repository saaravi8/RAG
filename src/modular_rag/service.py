"""Query orchestration that depends only on retrieval and generation ports."""

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .errors import ComponentContractError
from .models import Chunk, Citation, RAGResponse, SearchResult, VerificationResult
from .ports import AnswerGenerator, AnswerVerifier, Reranker, Retriever
from .relevance import RelevancePolicy, ScoreThresholdRelevancePolicy
from .verification import AnswerVerificationPolicy


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("{} must be a positive integer.".format(name))
    if value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return value


class RAGService:
    ABSTENTION_ANSWER = (
        "I do not have enough relevant indexed context to answer that question."
    )
    UNSUPPORTED_ANSWER = (
        "I could not verify that the generated answer is supported by the "
        "retrieved evidence."
    )

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        reranker: Optional[Reranker] = None,
        relevance_policy: Optional[RelevancePolicy] = None,
        answer_verifier: Optional[AnswerVerifier] = None,
        verification_policy: Optional[AnswerVerificationPolicy] = None,
        default_top_k: int = 5,
        default_fetch_k: int = 20,
        query_methods: Optional[Mapping[str, Retriever]] = None,
    ) -> None:
        default_top_k = _positive_int(default_top_k, "default_top_k")
        default_fetch_k = _positive_int(default_fetch_k, "default_fetch_k")
        if default_fetch_k < default_top_k:
            raise ValueError("default_fetch_k cannot be smaller than default_top_k.")
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.relevance_policy = (
            relevance_policy
            if relevance_policy is not None
            else ScoreThresholdRelevancePolicy()
        )
        self.answer_verifier = answer_verifier
        self.verification_policy = (
            verification_policy
            if verification_policy is not None
            else AnswerVerificationPolicy()
        )
        if (
            self.verification_policy.mode != "disabled"
            and self.answer_verifier is None
        ):
            raise ValueError(
                "An answer_verifier is required when verification is enabled."
            )
        self.default_top_k = default_top_k
        self.default_fetch_k = default_fetch_k
        self._retrievers: Dict[str, Retriever] = {"standard": retriever}
        for name, method_retriever in (query_methods or {}).items():
            normalized = self._normalize_method(name)
            if normalized == "standard":
                raise ValueError("'standard' is reserved for the default retriever.")
            if normalized in self._retrievers:
                raise ValueError("Duplicate query method: {!r}.".format(normalized))
            self._retrievers[normalized] = method_retriever

    @property
    def available_methods(self) -> Tuple[str, ...]:
        return tuple(self._retrievers)

    def ask(
        self,
        question: str,
        *,
        filters: Optional[Mapping[str, Any]] = None,
        top_k: Optional[int] = None,
        method: str = "standard",
    ) -> RAGResponse:
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise ValueError("question cannot be empty.")
        final_k = (
            _positive_int(top_k, "top_k")
            if top_k is not None
            else self.default_top_k
        )

        selected_method = self._normalize_method(method)
        try:
            retriever = self._retrievers[selected_method]
        except KeyError as exc:
            raise ValueError(
                "Unknown query method {!r}. Available methods: {}.".format(
                    selected_method, ", ".join(self.available_methods)
                )
            ) from exc

        fetch_k = max(
            final_k,
            self.default_fetch_k if self.reranker is not None else final_k,
        )
        retrieved = retriever.retrieve(question, top_k=fetch_k, filters=filters)
        candidates = self._consume_results(retrieved, "Retriever")
        self._validate_search_results(candidates, "Retriever")
        canonical_candidates = self._snapshot_evidence(candidates)
        selected_candidates = self.relevance_policy.select_relevant(
            question,
            self._snapshot_evidence(canonical_candidates),
            method=selected_method,
        )
        policy_results = self._consume_results(
            selected_candidates, "Relevance policy"
        )
        self._validate_search_results(policy_results, "Relevance policy")
        relevant_candidates = self._rebind_selected_results(
            canonical_candidates, policy_results
        )
        if not relevant_candidates:
            return self._abstention(
                question, reason="no_relevant_evidence"
            )

        results: Sequence[SearchResult]
        if self.reranker is not None:
            reranked = self.reranker.rerank(
                question,
                self._snapshot_evidence(relevant_candidates),
                top_k=final_k,
            )
            reranked_results = self._consume_results(reranked, "Reranker")
            self._validate_search_results(reranked_results, "Reranker")
            results = self._rebind_reranked_results(
                relevant_candidates, reranked_results, top_k=final_k
            )
        else:
            results = relevant_candidates[:final_k]

        if not results:
            return self._abstention(
                question, reason="no_reranked_evidence"
            )

        trusted_results = tuple(results)
        citations = tuple(
            self._citation(number, result)
            for number, result in enumerate(trusted_results, start=1)
        )
        answer = self.generator.generate(
            question, self._snapshot_evidence(trusted_results)
        )
        if not isinstance(answer, str):
            raise ComponentContractError("Answer generator must return a string.")
        if not answer.strip():
            raise ComponentContractError(
                "Answer generator must return a nonempty string."
            )
        verification, verification_failed = self._verify_answer(
            question, answer, trusted_results
        )
        if verification_failed:
            return self._abstention(
                question,
                reason="verification_error",
                answer=self.UNSUPPORTED_ANSWER,
                citations=citations,
                results=trusted_results,
                verification=verification,
            )
        if (
            verification is not None
            and not verification.supported
            and self.verification_policy.mode == "enforce"
        ):
            return self._abstention(
                question,
                reason="unsupported_answer",
                answer=self.UNSUPPORTED_ANSWER,
                citations=citations,
                results=trusted_results,
                verification=verification,
            )
        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            results=trusted_results,
            verification=verification,
        )

    def _verify_answer(
        self,
        question: str,
        answer: str,
        results: Sequence[SearchResult],
    ) -> Tuple[Optional[VerificationResult], bool]:
        if self.verification_policy.mode == "disabled":
            return None, False
        try:
            verification = self.answer_verifier.verify(
                question, answer, self._snapshot_evidence(results)
            )
        except Exception:
            if self.verification_policy.on_error == "raise":
                raise
            return VerificationResult(
                supported=False,
                reason="Answer verification failed.",
                metadata={"status": "verification_error"},
            ), True
        if not isinstance(verification, VerificationResult):
            error = ComponentContractError(
                "Answer verifier returned an invalid VerificationResult."
            )
            if self.verification_policy.on_error == "raise":
                raise error
            return VerificationResult(
                supported=False,
                reason="Answer verification returned an invalid result.",
                metadata={"status": "verification_error"},
            ), True
        return verification, False

    @staticmethod
    def _consume_results(value: Any, participant: str) -> Tuple[SearchResult, ...]:
        try:
            iterator = iter(value)
        except TypeError as exc:
            raise ComponentContractError(
                "{} returned a malformed result sequence.".format(participant)
            ) from exc
        return tuple(iterator)

    @staticmethod
    def _snapshot_evidence(
        results: Sequence[SearchResult],
    ) -> Tuple[SearchResult, ...]:
        """Copy evidence before exposing it to replaceable components."""

        return tuple(
            SearchResult(
                chunk=RAGService._snapshot_chunk(result.chunk),
                score=result.score,
            )
            for result in results
        )

    @staticmethod
    def _snapshot_chunk(chunk: Chunk) -> Chunk:
        if not isinstance(chunk, Chunk):
            raise TypeError("Evidence chunk must be a Chunk.")
        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            index=chunk.index,
            metadata=chunk.metadata,
        )

    def _abstention(
        self,
        question: str,
        *,
        reason: str,
        answer: Optional[str] = None,
        citations: Tuple[Citation, ...] = (),
        results: Tuple[SearchResult, ...] = (),
        verification: Optional[VerificationResult] = None,
    ) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=answer if answer is not None else self.ABSTENTION_ANSWER,
            citations=citations,
            results=results,
            abstained=True,
            abstention_reason=reason,
            verification=verification,
        )

    @staticmethod
    def _rebind_selected_results(
        candidates: Sequence[SearchResult],
        selected: Sequence[SearchResult],
    ) -> Tuple[SearchResult, ...]:
        remaining = list(candidates)
        rebound = []
        for result in selected:
            if not isinstance(result, SearchResult):
                raise ComponentContractError(
                    "Relevance policy must return SearchResult instances."
                )
            try:
                selected_chunk = RAGService._snapshot_chunk(result.chunk)
            except (TypeError, ValueError) as exc:
                raise ComponentContractError(
                    "Relevance policy returned a malformed chunk."
                ) from exc
            for index, candidate in enumerate(remaining):
                if (
                    selected_chunk == candidate.chunk
                    and result.score == candidate.score
                ):
                    rebound.append(candidate)
                    del remaining[index]
                    break
            else:
                raise ComponentContractError(
                    "Relevance policy may only filter or reorder its unchanged "
                    "candidate evidence."
                )
        return tuple(rebound)

    @staticmethod
    def _validate_search_results(
        results: Sequence[SearchResult], participant: str
    ) -> None:
        chunk_ids = set()
        for result in results:
            if not isinstance(result, SearchResult):
                raise ComponentContractError(
                    "{} must return SearchResult instances.".format(participant)
                )
            try:
                chunk = RAGService._snapshot_chunk(result.chunk)
            except (TypeError, ValueError) as exc:
                raise ComponentContractError(
                    "{} returned a malformed chunk.".format(participant)
                ) from exc
            if type(result.score) is not float:
                raise ComponentContractError(
                    "{} returned a malformed score.".format(participant)
                )
            try:
                score = float(result.score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ComponentContractError(
                    "{} returned a malformed score.".format(participant)
                ) from exc
            if not math.isfinite(score):
                raise ComponentContractError(
                    "{} returned a non-finite score.".format(participant)
                )
            if chunk.id in chunk_ids:
                raise ComponentContractError(
                    "{} returned duplicate chunk evidence.".format(participant)
                )
            chunk_ids.add(chunk.id)

    @staticmethod
    def _rebind_reranked_results(
        candidates: Sequence[SearchResult],
        selected: Sequence[SearchResult],
        *,
        top_k: int,
    ) -> Tuple[SearchResult, ...]:
        if len(selected) > top_k:
            raise ComponentContractError(
                "Reranker returned more results than the requested top_k."
            )
        remaining = list(candidates)
        rebound = []
        for result in selected:
            try:
                selected_chunk = RAGService._snapshot_chunk(result.chunk)
            except (TypeError, ValueError) as exc:
                raise ComponentContractError(
                    "Reranker returned a malformed chunk."
                ) from exc
            for index, candidate in enumerate(remaining):
                if selected_chunk == candidate.chunk:
                    rebound.append(SearchResult(candidate.chunk, result.score))
                    del remaining[index]
                    break
            else:
                raise ComponentContractError(
                    "Reranker may rescore or reorder candidates but cannot "
                    "alter their chunks."
                )
        return tuple(rebound)

    @staticmethod
    def _normalize_method(method: str) -> str:
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        normalized = method.strip().lower()
        if not normalized:
            raise ValueError("method cannot be empty.")
        return normalized

    @staticmethod
    def _citation(number: int, result: SearchResult) -> Citation:
        metadata = result.chunk.metadata
        source = str(
            metadata.get("source_name")
            or metadata.get("source_path")
            or result.chunk.document_id
        )
        excerpt = " ".join(result.chunk.text.split())[:300]
        return Citation(
            number=number,
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            source=source,
            excerpt=excerpt,
            score=result.score,
            metadata=metadata,
        )
