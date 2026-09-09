"""Non-executing, structure-aware chunking for repository source files."""

import ast
import hashlib
import importlib.metadata
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from rag_ingestion import Document

from .errors import OptionalDependencyError
from .fingerprint import describe_component
from .models import Chunk
from .ports import Chunker


@dataclass(frozen=True)
class SyntaxSpan:
    """A declaration boundary reported by a non-executing syntax parser.

    Line numbers are one-based and inclusive. ``parent`` and ``qualified_name``
    let chunkers retain nested symbol context without depending on a particular
    parser implementation.
    """

    name: str
    kind: str
    start_line: int
    end_line: int
    parent: str = ""
    qualified_name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SyntaxSpan.name cannot be empty.")
        if self.start_line <= 0 or self.end_line < self.start_line:
            raise ValueError("SyntaxSpan line range is invalid.")


@dataclass(frozen=True)
class SyntaxParseResult:
    """Syntax spans plus parser diagnostics safe to place in metadata."""

    spans: Tuple[SyntaxSpan, ...]
    parser_name: str
    parser_version: str
    status: str


class BuiltinSyntaxParser:
    """Safe baseline parser using Python AST and conservative declarations.

    This baseline keeps repository ingestion functional without optional native
    grammars. Install and inject :class:`TreeSitterSyntaxParser` for precise
    multi-language trees. Neither implementation imports or executes source.
    """

    _DECLARATIONS: Mapping[str, re.Pattern] = {
        "javascript": re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
            r"(?:(class|function)\s+([A-Za-z_$][\w$]*)|"
            r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?"
            r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>)"
        ),
        "typescript": re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?"
            r"(?:(class|interface|enum|type|function|namespace)\s+"
            r"([A-Za-z_$][\w$]*)|(?:const|let|var)\s+"
            r"([A-Za-z_$][\w$]*)\s*=)"
        ),
        "java": re.compile(
            r"^\s*(?:(?:public|protected|private|abstract|final|static|sealed)\s+)*"
            r"(class|interface|enum|record)\s+([A-Za-z_$][\w$]*)"
        ),
        "go": re.compile(r"^\s*(func|type)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)"),
        "rust": re.compile(
            r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
            r"(fn|struct|enum|trait|impl|type|mod)\s+([A-Za-z_]\w*)"
        ),
        "c": re.compile(
            r"^\s*(?:(struct|enum|union)\s+([A-Za-z_]\w*)|"
            r"(?:[\w:*&<>]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?)"
        ),
        "cpp": re.compile(
            r"^\s*(?:(class|struct|enum|namespace)\s+([A-Za-z_]\w*)|"
            r"(?:[\w:*&<>~]+\s+)+([A-Za-z_~]\w*)\s*\([^;]*\)\s*\{?)"
        ),
        "c_sharp": re.compile(
            r"^\s*(?:(?:public|private|protected|internal|static|abstract|sealed)\s+)*"
            r"(class|interface|enum|record|struct)\s+([A-Za-z_]\w*)"
        ),
        "ruby": re.compile(r"^\s*(class|module|def)\s+([A-Za-z_][\w!?=.:]*)"),
        "php": re.compile(
            r"^\s*(?:(?:public|private|protected|abstract|final|static)\s+)*"
            r"(class|interface|trait|enum|function)\s+([A-Za-z_]\w*)"
        ),
        "kotlin": re.compile(
            r"^\s*(?:(?:public|private|protected|internal|open|data|sealed)\s+)*"
            r"(class|interface|object|enum\s+class|fun)\s+([A-Za-z_]\w*)"
        ),
        "swift": re.compile(
            r"^\s*(?:(?:public|private|internal|open|final|static)\s+)*"
            r"(class|struct|enum|protocol|extension|func)\s+([A-Za-z_]\w*)"
        ),
        "bash": re.compile(
            r"^\s*(?:(?:function\s+)([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*\(\s*\))\s*\{?"
        ),
        "sql": re.compile(
            r"^\s*create\s+(?:or\s+replace\s+)?"
            r"(table|view|function|procedure|trigger|type|schema)\s+"
            r"(?:if\s+not\s+exists\s+)?([\w.\"`\[\]]+)",
            re.IGNORECASE,
        ),
    }

    def fingerprint_components(self):
        return {
            "algorithm": "builtin-safe-syntax",
            "algorithm_version": 1,
        }

    def parse(self, text: str, language: str) -> SyntaxParseResult:
        """Describe declarations without importing or executing ``text``."""

        if language == "python":
            return self._parse_python(text)
        return self._parse_declarations(text, language)

    def _parse_python(self, text: str) -> SyntaxParseResult:
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return SyntaxParseResult((), "python-ast", "stdlib", "fallback")

        lines = text.splitlines()
        spans: List[SyntaxSpan] = []

        def visit(nodes: Iterable[ast.AST], parent: str = "") -> None:
            for node in nodes:
                if not isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                start = int(getattr(node, "lineno", 1))
                decorators = getattr(node, "decorator_list", ())
                if decorators:
                    start = min(start, *(int(item.lineno) for item in decorators))
                start = self._leading_comment_line(lines, start)
                end = int(getattr(node, "end_lineno", start))
                kind = (
                    "class"
                    if isinstance(node, ast.ClassDef)
                    else "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                qualified = "{}.{}".format(parent, node.name) if parent else node.name
                spans.append(
                    SyntaxSpan(node.name, kind, start, end, parent, qualified)
                )
                visit(getattr(node, "body", ()), qualified)

        visit(tree.body)
        return SyntaxParseResult(
            tuple(sorted(spans, key=lambda span: (span.start_line, -span.end_line))),
            "python-ast",
            "stdlib",
            "parsed",
        )

    @staticmethod
    def _leading_comment_line(lines: Sequence[str], start: int) -> int:
        candidate = start
        cursor = start - 2
        while cursor >= 0 and lines[cursor].lstrip().startswith("#"):
            candidate = cursor + 1
            cursor -= 1
        return candidate

    def _parse_declarations(self, text: str, language: str) -> SyntaxParseResult:
        pattern = self._DECLARATIONS.get(language)
        if pattern is None:
            return SyntaxParseResult((), "bounded-lines", "1", "fallback")

        lines = text.splitlines()
        starts: List[Tuple[int, str, str]] = []
        for number, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            if stripped.startswith(("//", "/*", "*", "#", "--")):
                continue
            match = pattern.match(line)
            if not match:
                continue
            groups = tuple(group for group in match.groups() if group)
            if len(groups) < 2:
                continue
            kind, name = groups[0], groups[-1]
            starts.append((number, name[:128], kind.lower().replace(" ", "_")))

        spans = []
        for index, (start, name, kind) in enumerate(starts):
            next_start = (
                starts[index + 1][0]
                if index + 1 < len(starts)
                else len(lines) + 1
            )
            end = max(start, next_start - 1)
            spans.append(SyntaxSpan(name, kind, start, end, "", name))
        return SyntaxParseResult(
            tuple(spans),
            "bounded-declarations",
            "1",
            "parsed" if spans else "fallback",
        )


class TreeSitterSyntaxParser:
    """Tree-sitter adapter whose grammar provider is selected statically.

    The modern language pack downloads grammars on demand. This adapter refuses
    to request an uncached grammar, keeping ingestion network-free. Applications
    must provision grammars separately or inject a trusted parser factory.
    """

    _NODE_KINDS: Mapping[str, str] = {
        "class_definition": "class",
        "class_declaration": "class",
        "method_definition": "method",
        "method_declaration": "method",
        "function_definition": "function",
        "function_declaration": "function",
        "function_item": "function",
        "interface_declaration": "interface",
        "trait_item": "trait",
        "struct_item": "struct",
        "struct_specifier": "struct",
        "enum_item": "enum",
        "enum_declaration": "enum",
        "type_declaration": "type",
        "impl_item": "implementation",
        "module": "module",
        "namespace_definition": "namespace",
    }

    def __init__(
        self,
        parser_factory: Optional[Callable[[str], Any]] = None,
        *,
        fallback: Optional[BuiltinSyntaxParser] = None,
    ) -> None:
        self._parser_factory = parser_factory
        self._fallback = fallback or BuiltinSyntaxParser()
        self._parsers: Dict[str, Any] = {}

    def parse(self, text: str, language: str) -> SyntaxParseResult:
        """Parse with a provisioned grammar or return a safe baseline result."""

        try:
            parser = self._parser(language)
            tree = parser.parse(text.encode("utf-8"))
            root = tree.root_node
            spans = self._spans(root, text.encode("utf-8"))
            version = self._version()
            return SyntaxParseResult(spans, "tree-sitter", version, "parsed")
        except OptionalDependencyError:
            return self._fallback.parse(text, language)
        except Exception:
            return SyntaxParseResult(
                self._fallback.parse(text, language).spans,
                "tree-sitter",
                self._version(),
                "fallback",
            )

    def fingerprint_components(self):
        fallback, fallback_reusable = describe_component(self._fallback)
        grammar_inventory, inventory_reusable = self._grammar_inventory()
        return {
            "algorithm": "tree-sitter-with-fallback",
            "provider_version": self._version(),
            "grammar_inventory": grammar_inventory,
            "fallback": fallback,
            "custom_parser_factory": self._parser_factory is not None,
            "opaque": (
                self._parser_factory is not None
                or not fallback_reusable
                or not inventory_reusable
            ),
        }

    @staticmethod
    def _grammar_inventory() -> Tuple[Mapping[str, Any], bool]:
        try:
            from tree_sitter_language_pack import downloaded_languages
        except ImportError:
            try:
                import tree_sitter_languages  # noqa: F401
            except ImportError:
                return {"provider": "none", "languages": ()}, True
            return {
                "provider": "tree-sitter-languages",
                "languages": "bundled",
            }, True
        try:
            languages = tuple(sorted(str(item) for item in downloaded_languages()))
        except Exception:
            return {
                "provider": "tree-sitter-language-pack",
                "inventory": "unavailable",
                "opaque": True,
            }, False
        return {
            "provider": "tree-sitter-language-pack",
            "languages": languages,
        }, True

    def _parser(self, language: str) -> Any:
        if language in self._parsers:
            return self._parsers[language]
        factory = self._parser_factory or self._offline_factory()
        parser = factory(language)
        if not callable(getattr(parser, "parse", None)):
            raise TypeError("Tree-sitter parser must provide parse().")
        self._parsers[language] = parser
        return parser

    @staticmethod
    def _offline_factory() -> Callable[[str], Any]:
        try:
            from tree_sitter_language_pack import downloaded_languages, get_parser
        except ImportError:
            try:
                from tree_sitter_languages import get_parser
            except ImportError as exc:
                raise OptionalDependencyError(
                    "Tree-sitter grammars are not installed. Install the code extra "
                    "and provision grammars before repository ingestion."
                ) from exc
            return get_parser

        available = frozenset(downloaded_languages())

        def cached_parser(language: str) -> Any:
            if language not in available:
                raise OptionalDependencyError(
                    "The {!r} Tree-sitter grammar is not provisioned locally; "
                    "ingestion will not download it.".format(language)
                )
            return get_parser(language)

        return cached_parser

    def _spans(self, root: Any, source: bytes) -> Tuple[SyntaxSpan, ...]:
        spans: List[SyntaxSpan] = []

        def visit(node: Any, parent: str = "") -> None:
            node_type = str(getattr(node, "type", ""))
            kind = self._NODE_KINDS.get(node_type)
            active_parent = parent
            if kind:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    raw_name = source[name_node.start_byte : name_node.end_byte]
                    name = raw_name.decode("utf-8", errors="replace")[:128]
                    qualified = "{}.{}".format(parent, name) if parent else name
                    spans.append(
                        SyntaxSpan(
                            name=name,
                            kind=kind,
                            start_line=int(node.start_point[0]) + 1,
                            end_line=max(
                                int(node.start_point[0]) + 1,
                                int(node.end_point[0]) + 1,
                            ),
                            parent=parent,
                            qualified_name=qualified,
                        )
                    )
                    active_parent = qualified
            for child in getattr(node, "children", ()):
                visit(child, active_parent)

        visit(root)
        return tuple(sorted(spans, key=lambda span: (span.start_line, -span.end_line)))

    @staticmethod
    def _version() -> str:
        for distribution in ("tree-sitter-language-pack", "tree-sitter-languages"):
            try:
                return importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        return "unknown"


@dataclass(frozen=True)
class _ChunkRange:
    start_line: int
    end_line: int
    symbol_name: str = ""
    symbol_kind: str = "module"
    parent_symbol: str = ""
    qualified_symbol: str = ""


class CodeChunker:
    """Split repository source along safe syntax boundaries and line windows."""

    def __init__(
        self,
        *,
        max_lines: int = 200,
        overlap_lines: int = 20,
        parser: Optional[Any] = None,
    ) -> None:
        if isinstance(max_lines, bool) or not isinstance(max_lines, int):
            raise TypeError("max_lines must be an integer.")
        if isinstance(overlap_lines, bool) or not isinstance(overlap_lines, int):
            raise TypeError("overlap_lines must be an integer.")
        if max_lines <= 0:
            raise ValueError("max_lines must be positive.")
        if overlap_lines < 0 or overlap_lines >= max_lines:
            raise ValueError("overlap_lines must be between 0 and max_lines - 1.")
        self.max_lines = max_lines
        self.overlap_lines = overlap_lines
        self.parser = parser or TreeSitterSyntaxParser()

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Return bounded chunks with line, symbol, and parser metadata.

        Empty documents return no chunks. Invalid or unavailable syntax parsers
        fall back to bounded line windows through ``TreeSitterSyntaxParser``.
        """

        lines = document.text.splitlines(keepends=True)
        if not lines or not document.text.strip():
            return ()

        language = str(document.metadata.get("language") or document.document_type)
        parsed = self.parser.parse(document.text, language)
        ranges = self._ranges(len(lines), parsed.spans)
        document_id = self._document_id(document)
        chunks: List[Chunk] = []

        for item in ranges:
            text = "".join(lines[item.start_line - 1 : item.end_line]).rstrip()
            if not text.strip():
                continue
            chunk_index = len(chunks)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "document_id": document_id,
                    "document_type": document.document_type,
                    "line_start": item.start_line,
                    "line_end": item.end_line,
                    "symbol_name": item.symbol_name,
                    "symbol_kind": item.symbol_kind,
                    "parent_symbol": item.parent_symbol,
                    "qualified_symbol": item.qualified_symbol,
                    "parser_name": parsed.parser_name,
                    "parser_version": parsed.parser_version,
                    "parse_status": parsed.status,
                    "breadcrumb": self._breadcrumb(document, item),
                }
            )
            chunks.append(
                Chunk(
                    id="{}:{}:{}".format(document_id, chunk_index, digest),
                    document_id=document_id,
                    text=text,
                    index=chunk_index,
                    metadata=metadata,
                )
            )
        return tuple(chunks)

    def fingerprint_components(self):
        parser, parser_reusable = describe_component(self.parser)
        return {
            "algorithm": "syntax-aware-line-window",
            "algorithm_version": 1,
            "max_lines": self.max_lines,
            "overlap_lines": self.overlap_lines,
            "parser": parser,
            "opaque": not parser_reusable,
        }

    def _ranges(
        self, line_count: int, spans: Sequence[SyntaxSpan]
    ) -> Tuple[_ChunkRange, ...]:
        valid = tuple(
            span
            for span in spans
            if 1 <= span.start_line <= span.end_line <= line_count
        )
        ordered = sorted(valid, key=lambda span: (span.start_line, -span.end_line))
        outer: List[SyntaxSpan] = []
        covered_until = 0
        for span in ordered:
            if span.start_line <= covered_until:
                continue
            outer.append(span)
            covered_until = span.end_line

        children: Dict[str, List[SyntaxSpan]] = {}
        for span in valid:
            if span.parent:
                children.setdefault(span.parent, []).append(span)
        for nested in children.values():
            nested.sort(key=lambda span: (span.start_line, span.end_line))

        result: List[_ChunkRange] = []
        cursor = 1
        for span in outer:
            if span.start_line > cursor:
                result.extend(self._line_windows(cursor, span.start_line - 1))
            result.extend(self._span_ranges(span, children))
            cursor = max(cursor, span.end_line + 1)
        if cursor <= line_count:
            result.extend(self._line_windows(cursor, line_count))
        if not result:
            result.extend(self._line_windows(1, line_count))
        return tuple(result)

    def _span_ranges(
        self,
        span: SyntaxSpan,
        children: Mapping[str, Sequence[SyntaxSpan]],
    ) -> Sequence[_ChunkRange]:
        if span.end_line - span.start_line + 1 <= self.max_lines:
            return (self._from_span(span),)

        nested = tuple(
            child
            for child in children.get(span.qualified_name, ())
            if span.start_line <= child.start_line <= child.end_line <= span.end_line
        )
        if not nested:
            return self._line_windows(
                span.start_line,
                span.end_line,
                symbol_name=span.name,
                symbol_kind=span.kind,
                parent_symbol=span.parent,
                qualified_symbol=span.qualified_name,
            )

        result: List[_ChunkRange] = []
        cursor = span.start_line
        for child in nested:
            if child.start_line > cursor:
                result.extend(
                    self._line_windows(
                        cursor,
                        child.start_line - 1,
                        symbol_name=span.name,
                        symbol_kind=span.kind,
                        parent_symbol=span.parent,
                        qualified_symbol=span.qualified_name,
                    )
                )
            result.extend(self._span_ranges(child, children))
            cursor = max(cursor, child.end_line + 1)
        if cursor <= span.end_line:
            result.extend(
                self._line_windows(
                    cursor,
                    span.end_line,
                    symbol_name=span.name,
                    symbol_kind=span.kind,
                    parent_symbol=span.parent,
                    qualified_symbol=span.qualified_name,
                )
            )
        return tuple(result)

    @staticmethod
    def _from_span(span: SyntaxSpan) -> _ChunkRange:
        return _ChunkRange(
            span.start_line,
            span.end_line,
            span.name,
            span.kind,
            span.parent,
            span.qualified_name,
        )

    def _line_windows(
        self,
        start: int,
        end: int,
        *,
        symbol_name: str = "",
        symbol_kind: str = "module",
        parent_symbol: str = "",
        qualified_symbol: str = "",
    ) -> Sequence[_ChunkRange]:
        if end < start:
            return ()
        step = self.max_lines - self.overlap_lines
        ranges = []
        cursor = start
        while cursor <= end:
            window_end = min(cursor + self.max_lines - 1, end)
            ranges.append(
                _ChunkRange(
                    cursor,
                    window_end,
                    symbol_name,
                    symbol_kind,
                    parent_symbol,
                    qualified_symbol,
                )
            )
            if window_end == end:
                break
            cursor += step
        return tuple(ranges)

    @staticmethod
    def _document_id(document: Document) -> str:
        explicit = document.metadata.get("document_id")
        if explicit is not None and str(explicit).strip():
            return str(explicit)
        digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()[:20]
        return "code_{}".format(digest)

    @staticmethod
    def _breadcrumb(document: Document, item: _ChunkRange) -> str:
        path = str(document.metadata.get("relative_path") or "source")
        return " > ".join(part for part in (path, item.qualified_symbol) if part)


class RoutingChunker:
    """Route repository code/configuration without changing normal documents."""

    def __init__(self, default: Chunker, code: Optional[Chunker] = None) -> None:
        self.default = default
        self.code = code or CodeChunker()

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Route repository code/configuration and preserve normal prose behavior."""

        if document.metadata.get("content_kind") in {"code", "config"}:
            return self.code.chunk(document)
        return self.default.chunk(document)

    def fingerprint_components(self):
        default, default_reusable = describe_component(self.default)
        code, code_reusable = describe_component(self.code)
        return {
            "algorithm": "content-kind-routing",
            "algorithm_version": 1,
            "default": default,
            "code": code,
            "opaque": not default_reusable or not code_reusable,
        }
