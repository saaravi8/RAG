import tempfile
import unittest
from pathlib import Path

from rag_ingestion import (
    Document,
    DocumentPipeline,
    DocumentSource,
    DuplicateHandlerError,
    InvalidHandlerResultError,
    MissingDocumentTypeError,
    UnsupportedDocumentTypeError,
)


class DocumentPipelineTests(unittest.TestCase):
    def test_custom_handlers_can_be_registered_as_functions(self):
        pipeline = DocumentPipeline()

        @pipeline.register_loader("upper")
        def load_upper(source):
            return Document(source.read_text(), source.document_type)

        @pipeline.register_cleaner("upper")
        def uppercase(document):
            return document.with_text(document.text.upper())

        source = DocumentSource.from_text(
            "hello", name="sample.upper", metadata={"tenant_id": "acme"}
        )
        result = pipeline.process(source)

        self.assertEqual(result.text, "HELLO")
        self.assertEqual(result.document_type, "upper")
        self.assertEqual(result.metadata["tenant_id"], "acme")
        self.assertEqual(result.metadata["source_name"], "sample.upper")
        self.assertEqual(pipeline.supported_document_types, ("upper",))

    def test_cleaners_run_in_global_then_registration_order(self):
        pipeline = DocumentPipeline()
        pipeline.register_loader(
            "x", lambda source: Document(source.read_text(), source.document_type)
        )
        pipeline.register_cleaner("*", lambda document: document.with_text(document.text + "1"))
        pipeline.register_cleaner("x", lambda document: document.with_text(document.text + "2"))
        pipeline.register_cleaner("x", lambda document: document.with_text(document.text + "3"))

        result = pipeline.process(b"0", document_type="x")

        self.assertEqual(result.text, "0123")

    def test_type_is_inferred_from_a_path(self):
        pipeline = DocumentPipeline()
        pipeline.register_loader(
            ".TXT",
            lambda source: Document(source.read_text(), source.document_type),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.TXT"
            path.write_text("hello", encoding="utf-8")
            result = pipeline.load(path)

        self.assertEqual(result.document_type, "txt")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.metadata["source_name"], "example.TXT")
        self.assertEqual(result.metadata["source_path"], str(path))

    def test_explicit_type_supports_bytes_without_a_name(self):
        pipeline = DocumentPipeline()
        pipeline.register_loader(
            "custom", lambda source: Document(source.read_text(), source.document_type)
        )

        result = pipeline.process(b"content", document_type="CUSTOM")

        self.assertEqual(result.document_type, "custom")

    def test_missing_type_has_a_clear_error(self):
        with self.assertRaises(MissingDocumentTypeError):
            DocumentPipeline().load(b"content")

    def test_unsupported_type_lists_registered_types(self):
        pipeline = DocumentPipeline()
        pipeline.register_loader(
            "txt", lambda source: Document(source.read_text(), source.document_type)
        )

        with self.assertRaisesRegex(
            UnsupportedDocumentTypeError, "Registered types: txt"
        ):
            pipeline.load(b"content", document_type="pdf")

    def test_loader_registration_requires_explicit_replace(self):
        pipeline = DocumentPipeline()
        first = lambda source: Document("first", source.document_type)
        second = lambda source: Document("second", source.document_type)
        pipeline.register_loader("txt", first)

        with self.assertRaises(DuplicateHandlerError):
            pipeline.register_loader("txt", second)

        pipeline.register_loader("txt", second, replace=True)
        self.assertEqual(pipeline.load(b"x", document_type="txt").text, "second")

    def test_handler_must_return_a_document(self):
        pipeline = DocumentPipeline()
        pipeline.register_loader("bad", lambda source: "not a document")

        with self.assertRaises(InvalidHandlerResultError):
            pipeline.load(b"x", document_type="bad")


if __name__ == "__main__":
    unittest.main()
