"""Policies controlling optional post-generation answer verification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerVerificationPolicy:
    """Choose whether verification is disabled, reported, or enforced.

    ``on_error`` is deliberately explicit. ``raise`` exposes an unavailable or
    broken verifier to the caller, while ``abstain`` returns a safe structured
    response instead of treating an unverifiable answer as supported.
    """

    mode: str = "disabled"
    on_error: str = "raise"

    def __post_init__(self) -> None:
        mode = self._normalize(self.mode, "mode")
        on_error = self._normalize(self.on_error, "on_error")
        if mode not in {"disabled", "report", "enforce"}:
            raise ValueError(
                "mode must be 'disabled', 'report', or 'enforce'."
            )
        if on_error not in {"raise", "abstain"}:
            raise ValueError("on_error must be 'raise' or 'abstain'.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "on_error", on_error)

    @staticmethod
    def _normalize(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise TypeError("{} must be a string.".format(name))
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("{} cannot be empty.".format(name))
        return normalized
