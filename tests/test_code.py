import importlib.util
import sys
import unittest

from rag_ingestion import Document

from modular_rag import (
    BuiltinSyntaxParser,
    CodeChunker,
    OptionalDependencyError,
    RoutingChunker,
    SyntaxParseResult,
    SyntaxSpan,
    TreeSitterSyntaxParser,
    WordWindowChunker,
)


class BuiltinSyntaxParserTests(unittest.TestCase):
    def test_python_parser_reports_nested_symbols_without_executing_source(self):
        """AST parsing records structure even when runtime names do not exist."""

        source = """@missing_decorator
class Service:
    def run(self):
        return missing_runtime_name()

async def load():
    return 1
"""

        result = BuiltinSyntaxParser().parse(source, "python")

        self.assertEqual(result.status, "parsed")
        self.assertEqual(
            [
                (
                    span.name,
                    span.kind,
                    span.start_line,
                    span.end_line,
                    span.parent,
                    span.qualified_name,
                )
                for span in result.spans
            ],
            [
                ("Service", "class", 1, 4, "", "Service"),
                ("run", "function", 3, 4, "Service", "Service.run"),
                ("load", "async_function", 6, 7, "", "load"),
            ],
        )

    def test_invalid_python_and_unknown_languages_use_bounded_fallbacks(self):
        """Parser failures retain a safe line-window route instead of executing code."""

        invalid = BuiltinSyntaxParser().parse("def broken(:", "python")
        unknown = BuiltinSyntaxParser().parse("some content", "unknown")

        self.assertEqual((invalid.spans, invalid.status), ((), "fallback"))
        self.assertEqual((unknown.spans, unknown.status), ((), "fallback"))


class CodeChunkerTests(unittest.TestCase):
    class StaticParser:
        def __init__(self, spans):
            self.spans = tuple(spans)

        def parse(self, text, language):
            del text, language
            return SyntaxParseResult(self.spans, "test-parser", "1", "parsed")

    def test_symbol_chunks_have_traceable_metadata_and_contiguous_indexes(self):
        """Skipped blank preambles cannot leave gaps in public chunk indexes."""

        document = Document(
            "\n\nclass Service:\n    pass\n",
            "py",
            {
                "document_id": "repo-file",
                "relative_path": "src/service.py",
                "language": "python",
                "content_kind": "code",
            },
        )
        parser = self.StaticParser(
            (SyntaxSpan("Service", "class", 3, 4, "", "Service"),)
        )

        chunks = CodeChunker(max_lines=10, overlap_lines=0, parser=parser).chunk(
            document
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].index, chunks[0].metadata["chunk_index"]), (0, 0))
        self.assertEqual(chunks[0].text, "class Service:\n    pass")
        self.assertEqual(chunks[0].metadata["line_start"], 3)
        self.assertEqual(chunks[0].metadata["line_end"], 4)
        self.assertEqual(chunks[0].metadata["qualified_symbol"], "Service")
        self.assertEqual(chunks[0].metadata["breadcrumb"], "src/service.py > Service")

    def test_oversized_parent_splits_at_children_and_bounded_line_windows(self):
        """Large declarations remain bounded while preserving child symbol identity."""

        lines = ["line {}\n".format(number) for number in range(1, 13)]
        spans = (
            SyntaxSpan("Service", "class", 1, 12, "", "Service"),
            SyntaxSpan("first", "function", 2, 3, "Service", "Service.first"),
            SyntaxSpan("second", "function", 8, 9, "Service", "Service.second"),
        )
        document = Document(
            "".join(lines),
            "py",
            {
                "document_id": "large-file",
                "relative_path": "service.py",
                "language": "python",
                "content_kind": "code",
            },
        )

        chunks = CodeChunker(
            max_lines=4,
            overlap_lines=1,
            parser=self.StaticParser(spans),
        ).chunk(document)

        self.assertTrue(chunks)
        self.assertTrue(
            all(
                chunk.metadata["line_end"] - chunk.metadata["line_start"] + 1 <= 4
                for chunk in chunks
            )
        )
        symbols = {chunk.metadata["qualified_symbol"] for chunk in chunks}
        self.assertIn("Service.first", symbols)
        self.assertIn("Service.second", symbols)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))

    def test_large_span_sets_use_declared_parent_relationships(self):
        """Many declarations are handled without repeated all-span containment scans."""

        child_count = 500
        spans = [SyntaxSpan("Root", "class", 1, child_count + 1, "", "Root")]
        spans.extend(
            SyntaxSpan(
                "method_{}".format(number),
                "function",
                number + 1,
                number + 1,
                "Root",
                "Root.method_{}".format(number),
            )
            for number in range(1, child_count + 1)
        )
        document = Document(
            "".join("line {}\n".format(number) for number in range(child_count + 1)),
            "py",
            {
                "document_id": "many-symbols",
                "relative_path": "many.py",
                "language": "python",
                "content_kind": "code",
            },
        )

        chunks = CodeChunker(
            max_lines=10,
            overlap_lines=0,
            parser=self.StaticParser(spans),
        ).chunk(document)

        self.assertEqual(len(chunks), child_count + 1)
        self.assertEqual(chunks[-1].metadata["qualified_symbol"], "Root.method_500")

    def test_routing_chunker_only_sends_code_and_config_to_code_chunker(self):
        """Normal documents preserve the existing default chunking behavior."""

        class RecordingChunker:
            def __init__(self, label):
                self.label = label
                self.documents = []

            def chunk(self, document):
                self.documents.append(document)
                return (self.label,)

        default = RecordingChunker("default")
        code = RecordingChunker("code")
        router = RoutingChunker(default, code)
        prose = Document("prose", "txt", {"content_kind": "document"})
        source = Document("source", "py", {"content_kind": "code"})
        config = Document("setting", "toml", {"content_kind": "config"})

        self.assertEqual(router.chunk(prose), ("default",))
        self.assertEqual(router.chunk(source), ("code",))
        self.assertEqual(router.chunk(config), ("code",))
        self.assertEqual(default.documents, [prose])
        self.assertEqual(code.documents, [source, config])

    def test_configuration_rejects_windows_without_forward_progress(self):
        """Every line-window configuration must be positive and advance."""

        for arguments in (
            {"max_lines": 0},
            {"max_lines": 4, "overlap_lines": -1},
            {"max_lines": 4, "overlap_lines": 4},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CodeChunker(**arguments)

    def test_word_and_code_chunkers_require_real_integer_window_limits(self):
        """Booleans and fractional sizes cannot silently become window geometry."""

        constructors = (
            (WordWindowChunker, "max_words", "overlap_words"),
            (CodeChunker, "max_lines", "overlap_lines"),
        )
        for constructor, maximum_name, overlap_name in constructors:
            invalid_types = (
                {maximum_name: True},
                {maximum_name: 4.0},
                {maximum_name: "4"},
                {overlap_name: False},
                {overlap_name: 1.0},
                {overlap_name: "1"},
            )
            for arguments in invalid_types:
                with self.subTest(
                    constructor=constructor.__name__, arguments=arguments
                ):
                    with self.assertRaises(TypeError):
                        constructor(**arguments)

    def test_word_and_code_chunkers_keep_range_failures_as_value_errors(self):
        """Correctly typed limits still require positive size and forward progress."""

        cases = (
            (WordWindowChunker, {"max_words": 0}),
            (WordWindowChunker, {"max_words": -1}),
            (WordWindowChunker, {"max_words": 4, "overlap_words": -1}),
            (WordWindowChunker, {"max_words": 4, "overlap_words": 4}),
            (WordWindowChunker, {"max_words": 4, "overlap_words": 5}),
            (CodeChunker, {"max_lines": 0}),
            (CodeChunker, {"max_lines": -1}),
            (CodeChunker, {"max_lines": 4, "overlap_lines": -1}),
            (CodeChunker, {"max_lines": 4, "overlap_lines": 4}),
            (CodeChunker, {"max_lines": 4, "overlap_lines": 5}),
        )
        for constructor, arguments in cases:
            with self.subTest(
                constructor=constructor.__name__, arguments=arguments
            ):
                with self.assertRaises(ValueError):
                    constructor(**arguments)


class TreeSitterSyntaxParserTests(unittest.TestCase):
    class Node:
        def __init__(self, node_type, start, end, *, name=None, children=()):
            self.type = node_type
            self.start_point = (start - 1, 0)
            self.end_point = (end - 1, 1)
            self.start_byte = 0
            self.end_byte = 0
            self.children = tuple(children)
            self._name = name

        def child_by_field_name(self, field):
            return self._name if field == "name" else None

    class NameNode:
        def __init__(self, source, name):
            encoded = source.encode("utf-8")
            target = name.encode("utf-8")
            self.start_byte = encoded.index(target)
            self.end_byte = self.start_byte + len(target)

    def test_injected_parser_extracts_nested_qualified_symbols(self):
        """The optional adapter maps provider nodes into stable neutral spans."""

        source = "class Service:\n    def run(self):\n        pass\n"
        method = self.Node(
            "function_definition",
            2,
            3,
            name=self.NameNode(source, "run"),
        )
        root = self.Node(
            "class_definition",
            1,
            3,
            name=self.NameNode(source, "Service"),
            children=(method,),
        )

        class Parser:
            def parse(self, source_bytes):
                del source_bytes
                return type("Tree", (), {"root_node": root})()

        result = TreeSitterSyntaxParser(parser_factory=lambda language: Parser()).parse(
            source, "python"
        )

        self.assertEqual(
            [(span.qualified_name, span.parent) for span in result.spans],
            [("Service", ""), ("Service.run", "Service")],
        )
        self.assertEqual((result.parser_name, result.status), ("tree-sitter", "parsed"))

    def test_missing_injected_grammar_uses_builtin_fallback(self):
        """Unavailable native grammars degrade to the non-executing baseline parser."""

        def unavailable(language):
            del language
            raise OptionalDependencyError("not installed")

        result = TreeSitterSyntaxParser(parser_factory=unavailable).parse(
            "def fallback():\n    pass\n", "python"
        )

        self.assertEqual(result.parser_name, "python-ast")
        self.assertEqual(result.spans[0].name, "fallback")

    def test_declared_tree_sitter_provider_imports_when_extra_is_installed(self):
        """Both conditional dependency branches expose their expected parser API."""

        if sys.version_info < (3, 10):
            if importlib.util.find_spec("tree_sitter_languages") is None:
                self.skipTest("code extra is optional")
            from tree_sitter_languages import get_parser

            self.assertTrue(callable(get_parser))
        else:
            if importlib.util.find_spec("tree_sitter_language_pack") is None:
                self.skipTest("code extra is optional")
            from tree_sitter_language_pack import downloaded_languages, get_parser

            self.assertTrue(callable(downloaded_languages))
            self.assertTrue(callable(get_parser))

        result = TreeSitterSyntaxParser().parse(
            "def provider_check():\n    pass\n", "python"
        )
        self.assertEqual(result.spans[0].name, "provider_check")


if __name__ == "__main__":
    unittest.main()
