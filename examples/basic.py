"""Run with: PYTHONPATH=src python3 examples/basic.py"""

from rag_ingestion import DocumentSource

from modular_rag import build_demo_rag


app = build_demo_rag(max_words=60, overlap_words=10)

app.index(
    DocumentSource.from_text(
        """
        Project Atlas stores application data in PostgreSQL. Vector similarity
        search is provided by the pgvector extension.

        Documents are parsed and cleaned before they are split into chunks.
        Every answer should retain citations to the original source.
        """,
        name="architecture.md",
        metadata={"document_id": "architecture", "tenant_id": "demo"},
    )
)

response = app.ask(
    "Where is application data stored?",
    filters={"tenant_id": "demo"},
)

print(response.answer)
for citation in response.citations:
    print("[{}] {}: {}".format(citation.number, citation.source, citation.excerpt))
