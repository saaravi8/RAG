"""Small built-in handlers and a ready-to-use default pipeline."""

import csv
import io
import json
import re
import unicodedata
from html.parser import HTMLParser
from typing import Callable

from .models import Document, DocumentSource
from .pipeline import DocumentPipeline


def make_text_loader(encoding: str = "utf-8") -> Callable[[DocumentSource], Document]:
    """Create a plain-text loader with a configurable encoding."""

    def load_text(source: DocumentSource) -> Document:
        assert source.document_type is not None
        return Document(
            text=source.read_text(encoding),
            document_type=source.document_type,
        )

    return load_text


def load_json(source: DocumentSource) -> Document:
    """Parse JSON and produce stable, human-readable text for retrieval."""

    assert source.document_type is not None
    value = json.loads(source.read_text("utf-8"))
    return Document(
        text=json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        document_type=source.document_type,
        metadata={"json_root_type": type(value).__name__},
    )


def load_csv(source: DocumentSource) -> Document:
    """Render a header-based CSV as retrieval-friendly labelled rows."""

    assert source.document_type is not None
    rows = list(csv.reader(io.StringIO(source.read_text("utf-8-sig"))))
    if not rows:
        return Document(
            text="",
            document_type=source.document_type,
            metadata={"row_count": 0, "column_names": ()},
        )

    column_names = tuple(
        name.strip() or "column_{}".format(index + 1)
        for index, name in enumerate(rows[0])
    )
    rendered_rows = []
    for row_number, row in enumerate(rows[1:], start=1):
        cells = []
        width = max(len(column_names), len(row))
        for index in range(width):
            name = (
                column_names[index]
                if index < len(column_names)
                else "column_{}".format(index + 1)
            )
            value = row[index].strip() if index < len(row) else ""
            cells.append("{}: {}".format(name, value))
        rendered_rows.append("Row {}: {}".format(row_number, " | ".join(cells)))

    text = "Columns: {}".format(", ".join(column_names))
    if rendered_rows:
        text += "\n\n" + "\n\n".join(rendered_rows)
    return Document(
        text=text,
        document_type=source.document_type,
        metadata={
            "row_count": max(len(rows) - 1, 0),
            "column_names": column_names,
        },
    )


class _HTMLTextExtractor(HTMLParser):
    """Minimal dependency-free HTML-to-text adapter for ordinary pages."""

    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _SKIPPED_TAGS = {"noscript", "script", "style", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        del attrs
        if tag in self._SKIPPED_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._add_break()

    def handle_startendtag(self, tag, attrs) -> None:
        del attrs
        if not self._skip_depth and tag in self._BLOCK_TAGS:
            self._add_break()

    def handle_endtag(self, tag) -> None:
        if tag in self._SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in self._BLOCK_TAGS:
            self._add_break()

    def handle_data(self, data) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if (
            self._parts
            and not self._parts[-1].endswith((" ", "\n"))
            and text[0] not in ".,;:!?)]}"
        ):
            self._parts.append(" ")
        self._parts.append(text)

    def text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _add_break(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n\n"):
            self._parts.append("\n\n")


def load_html(source: DocumentSource) -> Document:
    """Extract visible text from HTML while ignoring executable/style content."""

    assert source.document_type is not None
    parser = _HTMLTextExtractor()
    parser.feed(source.read_text("utf-8"))
    parser.close()
    return Document(text=parser.text(), document_type=source.document_type)


def normalize_text(document: Document) -> Document:
    """Conservatively normalize Unicode, newlines, NULs, and line endings."""

    text = unicodedata.normalize("NFC", document.text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]

    compacted = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            compacted.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                compacted.append(line)

    while compacted and not compacted[0]:
        compacted.pop(0)
    while compacted and not compacted[-1]:
        compacted.pop()

    return document.with_text("\n".join(compacted))


def create_default_pipeline() -> DocumentPipeline:
    """Create an isolated pipeline with common dependency-free formats."""

    pipeline = DocumentPipeline()
    text_loader = make_text_loader()
    for document_type in ("txt", "md", "markdown"):
        pipeline.register_loader(document_type, text_loader)
    pipeline.register_loader("json", load_json)
    pipeline.register_loader("csv", load_csv)
    for document_type in ("htm", "html"):
        pipeline.register_loader(document_type, load_html)
    pipeline.register_cleaner("*", normalize_text)
    return pipeline
