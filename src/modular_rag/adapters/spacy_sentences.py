"""spaCy adapter for sentence-boundary identification."""

import hashlib
from typing import Any, Optional, Sequence

from rag_ingestion import Document

from ..errors import ComponentContractError, OptionalDependencyError
from ..models import SentenceSpan


class SpacySentenceSegmenter:
    """Identify sentences with spaCy while exposing provider-neutral results.

    By default this creates a blank language pipeline with spaCy's rule-based
    ``sentencizer``. Pass a configured spaCy ``Language`` object through
    ``nlp`` to use a trained parser, statistical sentence recognizer, or
    custom boundary component instead.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        punct_chars: Optional[Sequence[str]] = None,
        nlp: Optional[Any] = None,
    ) -> None:
        if nlp is not None:
            if not callable(nlp):
                raise TypeError("nlp must be a callable spaCy Language pipeline.")
            self._nlp = nlp
            return

        try:
            import spacy
        except ImportError as exc:
            raise OptionalDependencyError(
                "SpacySentenceSegmenter requires spaCy. "
                "Install it with: pip install 'modular-rag[spacy]'"
            ) from exc

        self._nlp = spacy.blank(language)
        config = {}
        if punct_chars is not None:
            config["punct_chars"] = list(punct_chars)
        self._nlp.add_pipe("sentencizer", config=config)

    def segment(self, document: Document) -> Sequence[SentenceSpan]:
        parsed = self._nlp(document.text)
        try:
            identified = tuple(parsed.sents)
        except ValueError as exc:
            raise ComponentContractError(
                "The supplied spaCy pipeline did not set sentence boundaries. "
                "Add a parser, senter, sentencizer, or custom boundary component."
            ) from exc

        document_id = self._document_id(document)
        sentences = []
        for sentence in identified:
            raw_text = sentence.text
            leading_space = len(raw_text) - len(raw_text.lstrip())
            trailing_space = len(raw_text) - len(raw_text.rstrip())
            text = raw_text.strip()
            if not text:
                continue

            start_char = int(sentence.start_char) + leading_space
            end_char = int(sentence.end_char) - trailing_space
            sentence_index = len(sentences)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "char_start": start_char,
                    "char_end": end_char,
                    "document_id": document_id,
                    "document_type": document.document_type,
                    "sentence_index": sentence_index,
                    "sentence_segmenter": "spacy",
                }
            )
            sentences.append(
                SentenceSpan(
                    id="{}:sentence:{}:{}".format(
                        document_id, sentence_index, digest
                    ),
                    document_id=document_id,
                    text=text,
                    index=sentence_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )
        return tuple(sentences)

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
