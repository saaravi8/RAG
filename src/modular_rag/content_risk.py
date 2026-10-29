"""Provider-neutral policies for quarantining risky retrieved content."""

from typing import Sequence

from .models import SearchResult


class PromptInjectionContentRiskPolicy:
    """Quarantine evidence explicitly marked as suspected prompt injection."""

    def select_safe(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        method: str,
    ) -> Sequence[SearchResult]:
        del question, method
        return tuple(
            result
            for result in results
            if result.chunk.metadata.get("prompt_injection_suspected") is not True
        )


class AllowAllContentRiskPolicy:
    """Explicit compatibility policy that performs no content quarantine."""

    def select_safe(
        self,
        question: str,
        results: Sequence[SearchResult],
        *,
        method: str,
    ) -> Sequence[SearchResult]:
        del question, method
        return tuple(results)
