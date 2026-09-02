import io
import json
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from rag_ingestion import DocumentSource, create_default_pipeline


class BuiltinHandlerTests(unittest.TestCase):
    def test_default_pipeline_advertises_every_builtin_format(self):
        pipeline = create_default_pipeline()

        self.assertEqual(
            pipeline.supported_document_types,
            ("csv", "htm", "html", "json", "markdown", "md", "pdf", "txt"),
        )

    def test_pdf_loader_reports_pages_and_empty_pages(self):
        stream = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_metadata({"/Title": "Reference", "/Author": "RAG team"})
        writer.write(stream)

        result = create_default_pipeline().process(
            stream.getvalue(), name="reference.pdf"
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.document_type, "pdf")
        self.assertEqual(result.metadata["page_count"], 1)
        self.assertEqual(result.metadata["pages_with_text"], 0)
        self.assertEqual(result.metadata["empty_pages"], (1,))
        self.assertEqual(result.metadata["title"], "Reference")
        self.assertEqual(result.metadata["author"], "RAG team")
        self.assertEqual(result.metadata["source_name"], "reference.pdf")

    def test_pdf_loader_extracts_text_with_a_page_label(self):
        result = create_default_pipeline().process(
            make_text_pdf("Hello PDF ingestion"), name="hello.pdf"
        )

        self.assertEqual(result.text, "Page 1\nHello PDF ingestion")
        self.assertEqual(result.metadata["pages_with_text"], 1)
        self.assertEqual(result.metadata["empty_pages"], ())

    def test_default_pipeline_ingests_supported_file_paths(self):
        samples = {
            "notes.txt": (b"Plain text file", "Plain text file"),
            "people.csv": (b"name,role\nAda,Engineer", "name: Ada"),
            "guide.html": (b"<h1>HTML guide</h1>", "HTML guide"),
            "manual.pdf": (make_text_pdf("PDF manual"), "PDF manual"),
        }
        pipeline = create_default_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            for filename, (data, expected_text) in samples.items():
                path = Path(directory) / filename
                path.write_bytes(data)

                result = pipeline.process(path)

                self.assertIn(expected_text, result.text)
                self.assertEqual(result.metadata["source_path"], str(path))

    def test_default_text_pipeline_normalizes_conservatively(self):
        pipeline = create_default_pipeline()
        source = DocumentSource.from_text(
            "  title  \r\n\r\n\r\n\r\nbody\x00  ", name="notes.md"
        )

        result = pipeline.process(source)

        self.assertEqual(result.text, "  title\n\n\nbody")
        self.assertEqual(result.document_type, "md")

    def test_text_normalization_handles_unicode_cr_and_outer_blank_lines(self):
        """Normalize representation without collapsing meaningful content."""

        pipeline = create_default_pipeline()
        source = DocumentSource.from_text(
            "\rCaf\u0065\u0301\rline  \r\r", name="unicode.txt"
        )

        result = pipeline.process(source)

        self.assertEqual(result.text, "Caf\u00e9\nline")

    def test_json_loader_produces_stable_text_and_metadata(self):
        pipeline = create_default_pipeline()

        result = pipeline.process(
            '{"z": 1, "a": [true, false]}'.encode("utf-8"),
            name="data.json",
        )

        self.assertLess(result.text.index('"a"'), result.text.index('"z"'))
        self.assertEqual(result.metadata["json_root_type"], "dict")
        self.assertEqual(result.metadata["source_name"], "data.json")

    def test_json_loader_preserves_unicode_and_reports_non_object_roots(self):
        """Keep JSON arrays and non-ASCII content readable and classified."""

        pipeline = create_default_pipeline()

        result = pipeline.process(
            '["caf\u00e9", {"answer": 42}]'.encode("utf-8"), name="data.json"
        )

        self.assertIn("caf\u00e9", result.text)
        self.assertNotIn("\\u00e9", result.text)
        self.assertEqual(result.metadata["json_root_type"], "list")

    def test_json_loader_rejects_malformed_input(self):
        """Invalid JSON must fail instead of being indexed as misleading raw text."""

        with self.assertRaises(json.JSONDecodeError):
            create_default_pipeline().process(b'{"broken":', name="broken.json")

    def test_csv_loader_labels_values_with_column_names(self):
        pipeline = create_default_pipeline()

        result = pipeline.process(
            "name,role\nAda,Engineer\nLin,Designer".encode("utf-8"),
            name="people.csv",
        )

        self.assertIn("Row 1: name: Ada | role: Engineer", result.text)
        self.assertEqual(result.metadata["row_count"], 2)
        self.assertEqual(result.metadata["column_names"], ("name", "role"))

    def test_csv_loader_handles_empty_input(self):
        """An empty CSV has a stable empty representation rather than a phantom row."""

        result = create_default_pipeline().process(b"", name="empty.csv")

        self.assertEqual(result.text, "")
        self.assertEqual(result.metadata["row_count"], 0)
        self.assertEqual(result.metadata["column_names"], ())

    def test_csv_loader_handles_bom_blank_headers_and_uneven_rows(self):
        """CSV width irregularities retain every cell under a stable label."""

        result = create_default_pipeline().process(
            "\ufeffname,,role\nAda,,Engineer,extra\nLin,Designer".encode("utf-8"),
            name="uneven.csv",
        )

        self.assertEqual(result.metadata["row_count"], 2)
        self.assertEqual(
            result.metadata["column_names"], ("name", "column_2", "role")
        )
        self.assertIn("role: Engineer | column_4: extra", result.text)
        self.assertIn("Row 2: name: Lin | column_2: Designer | role:", result.text)

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

    def test_html_loader_preserves_entities_and_block_boundaries(self):
        """Entity decoding and block separation keep visible prose readable."""

        result = create_default_pipeline().process(
            (
                "<p>A&amp;B<br/>Next</p><template>hidden</template>"
                "<svg>vector</svg><p>Done</p>"
            ).encode("utf-8"),
            name="blocks.html",
        )

        self.assertEqual(result.text, "A&B\n\nNext\n\nDone")

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


def make_text_pdf(text):
    """Build a tiny standards-compliant PDF without another test dependency."""

    content = "BT /F1 12 Tf 72 720 Td ({}) Tj ET".format(text).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(content)
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend("{} 0 obj\n".format(number).encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend("xref\n0 {}\n".format(len(objects) + 1).encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend("{:010d} 00000 n \n".format(offset).encode("ascii"))
    pdf.extend(
        (
            "trailer\n<< /Size {} /Root 1 0 R >>\n"
            "startxref\n{}\n%%EOF\n"
        )
        .format(len(objects) + 1, xref_offset)
        .encode("ascii")
    )
    return bytes(pdf)


if __name__ == "__main__":
    unittest.main()
