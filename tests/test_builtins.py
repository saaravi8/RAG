import unittest

from rag_ingestion import DocumentSource, create_default_pipeline


class BuiltinHandlerTests(unittest.TestCase):
    def test_default_pipeline_advertises_every_builtin_format(self):
        pipeline = create_default_pipeline()

        self.assertEqual(
            pipeline.supported_document_types,
            ("csv", "htm", "html", "json", "markdown", "md", "txt"),
        )

    def test_default_text_pipeline_normalizes_conservatively(self):
        pipeline = create_default_pipeline()
        source = DocumentSource.from_text(
            "  title  \r\n\r\n\r\n\r\nbody\x00  ", name="notes.md"
        )

        result = pipeline.process(source)

        self.assertEqual(result.text, "  title\n\n\nbody")
        self.assertEqual(result.document_type, "md")

    def test_json_loader_produces_stable_text_and_metadata(self):
        pipeline = create_default_pipeline()

        result = pipeline.process(
            '{"z": 1, "a": [true, false]}'.encode("utf-8"),
            name="data.json",
        )

        self.assertLess(result.text.index('"a"'), result.text.index('"z"'))
        self.assertEqual(result.metadata["json_root_type"], "dict")
        self.assertEqual(result.metadata["source_name"], "data.json")

    def test_csv_loader_labels_values_with_column_names(self):
        pipeline = create_default_pipeline()

        result = pipeline.process(
            "name,role\nAda,Engineer\nLin,Designer".encode("utf-8"),
            name="people.csv",
        )

        self.assertIn("Row 1: name: Ada | role: Engineer", result.text)
        self.assertEqual(result.metadata["row_count"], 2)
        self.assertEqual(result.metadata["column_names"], ("name", "role"))

    def test_html_loader_extracts_visible_text(self):
        pipeline = create_default_pipeline()

        result = pipeline.process(
            (
                "<html><head><style>hidden css</style></head>"
                "<body><h1>Guide</h1><p>Use <strong>PostgreSQL</strong>.</p>"
                "<script>hidden code</script></body></html>"
            ).encode("utf-8"),
            name="guide.html",
        )

        self.assertIn("Guide", result.text)
        self.assertIn("Use PostgreSQL.", result.text)
        self.assertNotIn("hidden", result.text)

    def test_default_pipeline_instances_do_not_share_registrations(self):
        first = create_default_pipeline()
        second = create_default_pipeline()
        first.register_loader(
            "private", lambda source: result_document(source.document_type)
        )

        self.assertIn("private", first.supported_document_types)
        self.assertNotIn("private", second.supported_document_types)


def result_document(document_type):
    from rag_ingestion import Document

    return Document("private", document_type)


if __name__ == "__main__":
    unittest.main()
