import unittest
from threading import Event, Thread

from rag_ingestion import Document

from modular_rag import (
    Chunk,
    ComponentContractError,
    CoordinatedPreparedDocumentReplacement,
    Indexer,
    IndexTransactionCoordinator,
    InMemoryVectorStore,
    QASCConfig,
    QASCRetriever,
    SearchResult,
    SentenceSpan,
    TransactionalDocumentIndex,
    TransactionalVectorStore,
    VectorRecord,
)


class _SingleChunker:
    def chunk(self, document):
        document_id = document.metadata["document_id"]
        return (
            Chunk(
                "{}:0".format(document_id),
                document_id,
                document.text,
                0,
                document.metadata,
            ),
        )


class _UnitEmbedder:
    def embed_documents(self, texts):
        return ((1.0,),) * len(texts)

    def embed_query(self, text):
        del text
        return (1.0,)


class _FailingSentenceSegmenter:
    def __init__(self):
        self.fail = False

    def segment(self, document):
        if self.fail:
            raise RuntimeError("injected QASC preparation failure")
        document_id = document.metadata["document_id"]
        return (
            SentenceSpan(
                "{}:0".format(document_id),
                document_id,
                document.text,
                0,
                0,
                len(document.text),
                document.metadata,
            ),
        )


class _PreparedReplacement:
    def __init__(
        self,
        owner,
        candidate,
        *,
        fail_commit=False,
        interrupt_commit=False,
        commit_error=None,
        deleted_count=0,
    ):
        self.owner = owner
        self.previous = owner._visible
        self.candidate = candidate
        self.fail_commit = fail_commit
        self.interrupt_commit = interrupt_commit
        self.commit_error = commit_error
        self.deleted_count = deleted_count
        self.committed = False
        self.rolled_back = False

    def commit(self):
        self.owner.events.append("commit:{}".format(self.owner.name))
        self.owner._visible = self.candidate
        self.committed = True
        if self.fail_commit:
            raise RuntimeError("injected commit failure: {}".format(self.owner.name))
        if self.interrupt_commit:
            raise KeyboardInterrupt("injected commit interruption")
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self):
        if self.rolled_back:
            return
        self.owner.events.append("rollback:{}".format(self.owner.name))
        if self.committed:
            self.owner._visible = self.previous
        self.rolled_back = True

    @property
    def transaction_coordinator(self):
        return self.owner.transaction_coordinator


class _TransactionalStore:
    def __init__(self, events, transaction_coordinator=None):
        self.name = "primary"
        self.events = events
        self._visible = {}
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else IndexTransactionCoordinator()
        )
        self.replace_prepare_error = None
        self.replace_commit_error = None
        self.delete_prepare_error = None
        self.delete_commit_error = None

    def prepare_replace_document(self, document_id, records):
        self.events.append("prepare:{}".format(self.name))
        if self.replace_prepare_error is not None:
            raise self.replace_prepare_error
        candidate = dict(self._visible)
        candidate[document_id] = tuple(records)
        return _PreparedReplacement(
            self, candidate, commit_error=self.replace_commit_error
        )

    def replace_document(self, document_id, records):
        replacement = self.prepare_replace_document(document_id, records)
        replacement.commit()

    def prepare_delete_document(self, document_id):
        self.events.append("prepare-delete:{}".format(self.name))
        if self.delete_prepare_error is not None:
            raise self.delete_prepare_error
        candidate = dict(self._visible)
        deleted_count = len(candidate.pop(document_id, ()))
        return _PreparedReplacement(
            self,
            candidate,
            commit_error=self.delete_commit_error,
            deleted_count=deleted_count,
        )

    def delete_document(self, document_id):
        deletion = self.prepare_delete_document(document_id)
        deletion.commit()
        return deletion.deleted_count

    def search(self, query_vector, *, top_k, filters=None):
        del query_vector
        selected = filters or {}
        records = tuple(
            record
            for document_id, document_records in self._visible.items()
            for record in document_records
            if selected.get("document_id", document_id) == document_id
        )
        return tuple(
            SearchResult(record.chunk, 1.0) for record in records[:top_k]
        )


class _TransactionalDocumentIndex:
    def __init__(self, name, events, transaction_coordinator=None):
        self.name = name
        self.events = events
        self._visible = {}
        self.transaction_coordinator = (
            transaction_coordinator
            if transaction_coordinator is not None
            else IndexTransactionCoordinator()
        )
        self.fail_prepare = False
        self.fail_commit = False
        self.interrupt_commit = False
        self.fail_delete_prepare = False
        self.fail_delete_commit = False
        self.replace_prepare_error = None
        self.replace_commit_error = None
        self.delete_prepare_error = None
        self.delete_commit_error = None

    def prepare_replace_document(self, document):
        self.events.append("prepare:{}".format(self.name))
        if self.fail_prepare:
            raise RuntimeError("injected prepare failure: {}".format(self.name))
        if self.replace_prepare_error is not None:
            raise self.replace_prepare_error
        document_id = document.metadata["document_id"]
        candidate = dict(self._visible)
        candidate[document_id] = document
        return _PreparedReplacement(
            self,
            candidate,
            fail_commit=self.fail_commit,
            interrupt_commit=self.interrupt_commit,
            commit_error=self.replace_commit_error,
        )

    def replace_document(self, document):
        replacement = self.prepare_replace_document(document)
        replacement.commit()

    def prepare_delete_document(self, document_id):
        self.events.append("prepare-delete:{}".format(self.name))
        if self.fail_delete_prepare:
            raise RuntimeError(
                "injected delete prepare failure: {}".format(self.name)
            )
        if self.delete_prepare_error is not None:
            raise self.delete_prepare_error
        candidate = dict(self._visible)
        deleted_count = int(candidate.pop(document_id, None) is not None)
        return _PreparedReplacement(
            self,
            candidate,
            fail_commit=self.fail_delete_commit,
            commit_error=self.delete_commit_error,
            deleted_count=deleted_count,
        )

    def delete_document(self, document_id):
        deletion = self.prepare_delete_document(document_id)
        deletion.commit()
        return deletion.deleted_count

    def retrieve_text(self, document_id):
        return self._visible[document_id].text


class AtomicIndexingTests(unittest.TestCase):
    @staticmethod
    def _build_indexer(auxiliary_count=3):
        events = []
        coordinator = IndexTransactionCoordinator()
        store = _TransactionalStore(events, coordinator)
        indexes = tuple(
            _TransactionalDocumentIndex(
                "aux{}".format(index), events, coordinator
            )
            for index in range(auxiliary_count)
        )
        indexer = Indexer(
            object(),
            _SingleChunker(),
            _UnitEmbedder(),
            store,
            document_indexes=indexes,
            transaction_coordinator=coordinator,
        )
        return indexer, store, indexes, events

    @staticmethod
    def _document(text):
        return Document(text, "txt", {"document_id": "guide"})

    @staticmethod
    def _primary_text(store):
        results = store.search(
            (1.0,), top_k=1, filters={"document_id": "guide"}
        )
        return results[0].chunk.text

    def test_all_replacements_are_prepared_before_the_first_commit(self):
        """No participant publishes a new version while another is still staging."""

        indexer, store, indexes, events = self._build_indexer(auxiliary_count=2)

        indexer.index_document(self._document("new version"))

        self.assertEqual(
            events,
            [
                "prepare:primary",
                "prepare:aux0",
                "prepare:aux1",
                "commit:primary",
                "commit:aux0",
                "commit:aux1",
            ],
        )
        self.assertEqual(self._primary_text(store), "new version")
        self.assertEqual(
            [index.retrieve_text("guide") for index in indexes],
            ["new version", "new version"],
        )

    def test_permanent_failure_matrix_restores_every_participant(self):
        """Every prepare/commit position handles ordinary and control failures."""

        failure_types = (RuntimeError, KeyboardInterrupt, SystemExit)
        operations = ("replace", "delete")
        stages = ("prepare", "commit")
        participant_names = (
            "primary",
            "aux0",
            "aux1",
            "aux2",
            "additional",
        )

        for operation in operations:
            for stage in stages:
                for failing_position, failing_name in enumerate(participant_names):
                    for failure_type in failure_types:
                        with self.subTest(
                            operation=operation,
                            stage=stage,
                            participant=failing_name,
                            failure_type=failure_type.__name__,
                        ):
                            indexer, store, indexes, events = self._build_indexer()
                            indexer.index_document(self._document("old version"))
                            events.clear()
                            additional = _TransactionalDocumentIndex(
                                "additional",
                                events,
                                indexer.transaction_coordinator,
                            )
                            participants = (store,) + indexes + (additional,)
                            error = failure_type(
                                "injected permanent {} {} failure: {}".format(
                                    operation, stage, failing_name
                                )
                            )
                            setattr(
                                participants[failing_position],
                                "{}_{}_error".format(operation, stage),
                                error,
                            )

                            with self.assertRaises(failure_type):
                                if operation == "replace":
                                    indexer.index_document(
                                        self._document("new version"),
                                        additional_preparations=(
                                            lambda: additional.prepare_replace_document(
                                                self._document("new version")
                                            ),
                                        ),
                                    )
                                else:
                                    indexer.delete(
                                        "guide",
                                        additional_preparations=(
                                            lambda: additional.prepare_delete_document(
                                                "guide"
                                            ),
                                        ),
                                    )

                            self.assertEqual(self._primary_text(store), "old version")
                            self.assertEqual(
                                [
                                    index.retrieve_text("guide")
                                    for index in indexes
                                ],
                                ["old version", "old version", "old version"],
                            )
                            self.assertEqual(additional._visible, {})

                            prepare_prefix = (
                                "prepare" if operation == "replace" else "prepare-delete"
                            )
                            if stage == "prepare":
                                expected = [
                                    "{}:{}".format(prepare_prefix, name)
                                    for name in participant_names[
                                        : failing_position + 1
                                    ]
                                ]
                                expected.extend(
                                    "rollback:{}".format(name)
                                    for name in reversed(
                                        participant_names[:failing_position]
                                    )
                                )
                            else:
                                expected = [
                                    "{}:{}".format(prepare_prefix, name)
                                    for name in participant_names
                                ]
                                expected.extend(
                                    "commit:{}".format(name)
                                    for name in participant_names[
                                        : failing_position + 1
                                    ]
                                )
                                expected.extend(
                                    "rollback:{}".format(name)
                                    for name in reversed(participant_names)
                                )
                            self.assertEqual(events, expected)

    def test_auxiliary_prepare_failure_preserves_every_previous_version(self):
        """Failure at any auxiliary prepare cannot publish a mixed replacement."""

        for failing_index in range(3):
            with self.subTest(failing_index=failing_index):
                indexer, store, indexes, events = self._build_indexer()
                indexer.index_document(self._document("old version"))
                events.clear()
                indexes[failing_index].fail_prepare = True

                with self.assertRaisesRegex(RuntimeError, "injected prepare failure"):
                    indexer.index_document(self._document("new version"))

                self.assertEqual(self._primary_text(store), "old version")
                self.assertEqual(
                    [index.retrieve_text("guide") for index in indexes],
                    ["old version", "old version", "old version"],
                )
                self.assertNotIn("commit:primary", events)

    def test_auxiliary_commit_failure_rolls_back_every_previous_version(self):
        """Even a broken commit that mutates then raises cannot leave mixed versions."""

        for failing_index in range(3):
            with self.subTest(failing_index=failing_index):
                indexer, store, indexes, events = self._build_indexer()
                indexer.index_document(self._document("old version"))
                events.clear()
                indexes[failing_index].fail_commit = True

                with self.assertRaisesRegex(RuntimeError, "injected commit failure"):
                    indexer.index_document(self._document("new version"))

                self.assertEqual(self._primary_text(store), "old version")
                self.assertEqual(
                    [index.retrieve_text("guide") for index in indexes],
                    ["old version", "old version", "old version"],
                )

    def test_commit_interruption_rolls_back_before_propagating(self):
        """Process-control exceptions cannot leave a half-published generation."""

        indexer, store, indexes, events = self._build_indexer()
        indexer.index_document(self._document("old version"))
        events.clear()
        indexes[1].interrupt_commit = True

        with self.assertRaisesRegex(KeyboardInterrupt, "interruption"):
            indexer.index_document(self._document("new version"))

        self.assertEqual(self._primary_text(store), "old version")
        self.assertEqual(
            [index.retrieve_text("guide") for index in indexes],
            ["old version", "old version", "old version"],
        )
        self.assertIn("rollback:primary", events)

    def test_multi_index_composition_rejects_each_legacy_participant(self):
        """Every write participant must support safe staged replacement."""

        class LegacyIndex:
            def replace_document(self, document):
                del document

            def delete_document(self, document_id):
                del document_id
                return 0

        events = []
        store = _TransactionalStore(events)
        transactional_index = _TransactionalDocumentIndex("aux0", events)

        unsafe_compositions = (
            (LegacyIndex(), (transactional_index,), "primary vector store"),
            (store, (LegacyIndex(),), "auxiliary document index 0"),
        )
        for selected_store, indexes, missing_participant in unsafe_compositions:
            with self.subTest(missing_participant=missing_participant):
                with self.assertRaisesRegex(
                    ComponentContractError, missing_participant
                ):
                    Indexer(
                        object(),
                        _SingleChunker(),
                        _UnitEmbedder(),
                        selected_store,
                        document_indexes=indexes,
                    )

        self.assertEqual(store._visible, {})
        self.assertEqual(events, [])

    def test_atomic_composition_requires_exact_coordinator_on_every_participant(self):
        """Missing, null, and foreign coordinator ownership all fail closed."""

        participant_names = (
            "primary vector store",
            "auxiliary document index 0",
            "auxiliary document index 1",
        )
        for failing_position, participant_name in enumerate(participant_names):
            for binding in ("missing", "none", "mismatched"):
                with self.subTest(
                    participant=participant_name,
                    binding=binding,
                ):
                    events = []
                    coordinator = IndexTransactionCoordinator()
                    store = _TransactionalStore(events, coordinator)
                    indexes = (
                        _TransactionalDocumentIndex("aux0", events, coordinator),
                        _TransactionalDocumentIndex("aux1", events, coordinator),
                    )
                    participants = (store,) + indexes
                    if binding == "missing":
                        del participants[failing_position].transaction_coordinator
                    elif binding == "none":
                        participants[failing_position].transaction_coordinator = None
                    else:
                        participants[failing_position].transaction_coordinator = (
                            IndexTransactionCoordinator()
                        )

                    with self.assertRaisesRegex(
                        ComponentContractError, participant_name
                    ):
                        Indexer(
                            object(),
                            _SingleChunker(),
                            _UnitEmbedder(),
                            store,
                            document_indexes=indexes,
                            transaction_coordinator=coordinator,
                        )

                    self.assertEqual(events, [])

    def test_additional_participant_requires_primary_coordinator_ownership(self):
        """A legacy single store remains valid until a coordinated write is asked."""

        class LegacyTransactionalStore(_TransactionalStore):
            transaction_coordinator = None

        events = []
        store = LegacyTransactionalStore(events)
        del store.transaction_coordinator
        indexer = Indexer(object(), _SingleChunker(), _UnitEmbedder(), store)
        preparation_called = False

        def prepare_additional():
            nonlocal preparation_called
            preparation_called = True
            raise AssertionError("must not prepare without shared ownership")

        with self.assertRaisesRegex(ComponentContractError, "primary vector store"):
            indexer.index_document(
                self._document("new version"),
                additional_preparations=(prepare_additional,),
            )

        self.assertFalse(preparation_called)
        self.assertEqual(events, [])

    def test_dynamic_handles_reject_invalid_unbound_and_mismatched_results(self):
        """Extra preparations cannot join a transaction without explicit binding."""

        class DynamicHandle:
            def __init__(self, coordinator_marker):
                self.commit_count = 0
                self.rollback_count = 0
                if coordinator_marker != "missing":
                    self.transaction_coordinator = coordinator_marker

            def commit(self):
                self.commit_count += 1

            def rollback(self):
                self.rollback_count += 1

        class InvalidHandle:
            def __init__(self, coordinator):
                self.transaction_coordinator = coordinator

            def commit(self):
                pass

        for operation in ("replace", "delete"):
            for binding in ("invalid", "missing", "none", "mismatched"):
                with self.subTest(operation=operation, binding=binding):
                    indexer, store, indexes, events = self._build_indexer(
                        auxiliary_count=1
                    )
                    indexer.index_document(self._document("old version"))
                    events.clear()
                    if binding == "invalid":
                        handle = InvalidHandle(indexer.transaction_coordinator)
                        expected_message = "invalid prepared document replacement"
                    else:
                        coordinator_marker = {
                            "missing": "missing",
                            "none": None,
                            "mismatched": IndexTransactionCoordinator(),
                        }[binding]
                        handle = DynamicHandle(coordinator_marker)
                        expected_message = "not bound"

                    with self.assertRaisesRegex(
                        ComponentContractError, expected_message
                    ):
                        if operation == "replace":
                            indexer.index_document(
                                self._document("new version"),
                                additional_preparations=(lambda: handle,),
                            )
                        else:
                            indexer.delete(
                                "guide",
                                additional_preparations=(lambda: handle,),
                            )

                    self.assertEqual(self._primary_text(store), "old version")
                    self.assertEqual(indexes[0].retrieve_text("guide"), "old version")
                    self.assertNotIn("commit:primary", events)
                    if binding != "invalid":
                        self.assertEqual(handle.rollback_count, 1)

    def test_single_store_indexing_keeps_the_original_write_contract(self):
        """A primary-only composition does not require the new optional protocol."""

        class LegacyStore:
            def __init__(self):
                self.records = ()

            def replace_document(self, document_id, records):
                del document_id
                self.records = tuple(records)

            def delete_document(self, document_id):
                del document_id
                deleted = len(self.records)
                self.records = ()
                return deleted

            def search(self, query_vector, *, top_k, filters=None):
                del query_vector, filters
                return tuple(
                    SearchResult(record.chunk, 1.0)
                    for record in self.records[:top_k]
                )

        store = LegacyStore()
        indexer = Indexer(object(), _SingleChunker(), _UnitEmbedder(), store)

        indexer.index_document(self._document("compatible version"))

        self.assertEqual(store.records[0].chunk.text, "compatible version")

    def test_primary_only_legacy_store_delete_keeps_original_contract(self):
        """A lone store still deletes directly without staged/coordinator APIs."""

        class LegacyStore:
            def __init__(self):
                self.deleted_ids = []

            def replace_document(self, document_id, records):
                del document_id, records

            def delete_document(self, document_id):
                self.deleted_ids.append(document_id)
                return 3

            def search(self, query_vector, *, top_k, filters=None):
                del query_vector, top_k, filters
                return ()

        store = LegacyStore()
        indexer = Indexer(object(), _SingleChunker(), _UnitEmbedder(), store)

        self.assertEqual(indexer.delete("guide"), 3)
        self.assertEqual(store.deleted_ids, ["guide"])

    def test_transaction_protocols_are_structural_and_coordinator_aware(self):
        """Adapters satisfy the public protocols without inheriting project classes."""

        indexer, store, indexes, _ = self._build_indexer(auxiliary_count=1)
        handle = _PreparedReplacement(store, {})

        self.assertIsInstance(store, TransactionalVectorStore)
        self.assertIsInstance(indexes[0], TransactionalDocumentIndex)
        self.assertIsInstance(handle, CoordinatedPreparedDocumentReplacement)
        self.assertIs(
            store.transaction_coordinator,
            indexer.transaction_coordinator,
        )
        self.assertIs(
            indexes[0].transaction_coordinator,
            indexer.transaction_coordinator,
        )

    def test_indexer_rejects_forged_canonical_document_id_before_embedding(self):
        """A chunker cannot redirect a canonical document write to another owner."""

        class ForgingChunker:
            def chunk(self, document):
                document.metadata["document_id"] = "forged"
                return (
                    Chunk(
                        "forged:0",
                        "forged",
                        document.text,
                        0,
                        document.metadata,
                    ),
                )

        class RecordingEmbedder(_UnitEmbedder):
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                return super().embed_documents(texts)

        events = []
        store = _TransactionalStore(events)
        embedder = RecordingEmbedder()
        indexer = Indexer(object(), ForgingChunker(), embedder, store)

        with self.assertRaisesRegex(
            ComponentContractError, "canonical document metadata"
        ):
            indexer.index_document(self._document("canonical"))

        self.assertEqual(embedder.calls, 0)
        self.assertEqual(store._visible, {})
        self.assertEqual(events, [])

    def test_indexer_rejects_missing_or_changed_protected_chunk_metadata(self):
        """Protected repository provenance must survive chunking byte-for-byte."""

        canonical_metadata = {
            "document_id": "guide",
            "repository_id": "example",
            "relative_path": "src/app.py",
            "document_state_token": "trusted-token",
        }

        class MetadataChunker:
            def __init__(self, mutation):
                self.mutation = mutation

            def chunk(self, document):
                metadata = dict(document.metadata)
                self.mutation(metadata)
                return (
                    Chunk(
                        "guide:0",
                        "guide",
                        document.text,
                        0,
                        metadata,
                    ),
                )

        class RecordingEmbedder(_UnitEmbedder):
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                return super().embed_documents(texts)

        cases = (
            (
                lambda metadata: metadata.__setitem__(
                    "relative_path", "src/forged.py"
                ),
                "relative_path",
            ),
            (
                lambda metadata: metadata.pop("document_state_token"),
                "document_state_token",
            ),
        )
        document = Document("protected", "txt", canonical_metadata)
        for mutation, failed_key in cases:
            with self.subTest(failed_key=failed_key):
                events = []
                store = _TransactionalStore(events)
                embedder = RecordingEmbedder()
                indexer = Indexer(
                    object(), MetadataChunker(mutation), embedder, store
                )

                with self.assertRaisesRegex(
                    ComponentContractError, repr(failed_key)
                ):
                    indexer.index_document(
                        document,
                        required_chunk_metadata=canonical_metadata,
                    )

                self.assertEqual(embedder.calls, 0)
                self.assertEqual(store._visible, {})
                self.assertEqual(events, [])

        store = _TransactionalStore([])
        indexer = Indexer(
            object(), MetadataChunker(lambda metadata: None), _UnitEmbedder(), store
        )
        indexer.index_document(
            document,
            required_chunk_metadata=canonical_metadata,
        )
        self.assertEqual(self._primary_text(store), "protected")

    def test_max_chunks_requires_a_positive_non_boolean_integer(self):
        """Strict typing prevents booleans and lossy numeric limits at the boundary."""

        cases = (
            (True, TypeError),
            (False, TypeError),
            (1.0, TypeError),
            ("1", TypeError),
            (0, ValueError),
            (-1, ValueError),
        )
        for value, expected_error in cases:
            with self.subTest(value=value):
                indexer, store, indexes, events = self._build_indexer(
                    auxiliary_count=0
                )

                with self.assertRaises(expected_error):
                    indexer.index_document(
                        self._document("bounded"), max_chunks=value
                    )

                self.assertEqual(store._visible, {})
                self.assertEqual(indexes, ())
                self.assertEqual(events, [])

        indexer, store, _, _ = self._build_indexer(auxiliary_count=0)
        indexer.index_document(self._document("bounded"), max_chunks=1)
        self.assertEqual(self._primary_text(store), "bounded")

    def test_chunker_result_boundary_rejects_non_iterables_and_non_chunks(self):
        """Malformed chunk adapters fail clearly before any item is dereferenced."""

        class ReturningChunker:
            def __init__(self, result):
                self.result = result

            def chunk(self, document):
                del document
                return self.result

        cases = (
            (None, "non-iterable"),
            (42, "non-iterable"),
            (("not a chunk",), "non-Chunk item at position 0"),
            ((_SingleChunker(),), "non-Chunk item at position 0"),
        )
        for result, expected_message in cases:
            with self.subTest(result=repr(result)):
                events = []
                store = _TransactionalStore(events)
                indexer = Indexer(
                    object(), ReturningChunker(result), _UnitEmbedder(), store
                )

                with self.assertRaisesRegex(
                    ComponentContractError, expected_message
                ):
                    indexer.index_document(self._document("malformed"))

                self.assertEqual(store._visible, {})
                self.assertEqual(events, [])

    def test_lazy_chunker_type_error_propagates_from_adapter_iteration(self):
        """A generator's own TypeError remains distinguishable from non-iterability."""

        class LazyFailingChunker:
            def chunk(self, document):
                yield _SingleChunker().chunk(document)[0]
                raise TypeError("chunk adapter failed during iteration")

        events = []
        store = _TransactionalStore(events)
        indexer = Indexer(
            object(), LazyFailingChunker(), _UnitEmbedder(), store
        )

        with self.assertRaisesRegex(
            TypeError, "chunk adapter failed during iteration"
        ):
            indexer.index_document(self._document("lazy chunks"))

        self.assertEqual(store._visible, {})
        self.assertEqual(events, [])

    def test_embedder_result_boundary_distinguishes_non_iterable_and_lazy_failure(self):
        """Only failure to obtain an iterator is translated into a contract error."""

        class ReturningEmbedder:
            def __init__(self, result):
                self.result = result

            def embed_documents(self, texts):
                del texts
                return self.result

        for result in (None, 42):
            with self.subTest(non_iterable=repr(result)):
                events = []
                store = _TransactionalStore(events)
                indexer = Indexer(
                    object(), _SingleChunker(), ReturningEmbedder(result), store
                )

                with self.assertRaisesRegex(
                    ComponentContractError, "Embedder returned a non-iterable"
                ):
                    indexer.index_document(self._document("malformed vectors"))

                self.assertEqual(store._visible, {})
                self.assertEqual(events, [])

        def lazy_vectors():
            yield (1.0,)
            raise TypeError("embedder adapter failed during iteration")

        events = []
        store = _TransactionalStore(events)
        indexer = Indexer(
            object(), _SingleChunker(), ReturningEmbedder(lazy_vectors()), store
        )
        with self.assertRaisesRegex(
            TypeError, "embedder adapter failed during iteration"
        ):
            indexer.index_document(self._document("lazy vectors"))
        self.assertEqual(store._visible, {})
        self.assertEqual(events, [])

    def test_indexer_constructs_vector_records_to_validate_each_vector(self):
        """Iterable output still crosses the VectorRecord numeric validation seam."""

        class ReturningEmbedder:
            def __init__(self, vector):
                self.vector = vector

            def embed_documents(self, texts):
                del texts
                return (self.vector,)

        cases = (
            (1, TypeError, "numeric sequence"),
            ("bad", TypeError, "numeric sequence"),
            ((), ValueError, "cannot be empty"),
            ((True,), TypeError, "not a boolean"),
            ((float("nan"),), ValueError, "must be finite"),
        )
        for vector, expected_error, expected_message in cases:
            with self.subTest(vector=repr(vector)):
                store = _TransactionalStore([])
                indexer = Indexer(
                    object(), _SingleChunker(), ReturningEmbedder(vector), store
                )

                with self.assertRaisesRegex(expected_error, expected_message):
                    indexer.index_document(self._document("invalid vector"))

                self.assertEqual(store._visible, {})

    def test_delete_requires_a_nonempty_string_document_id(self):
        """The public delete boundary rejects ambiguous IDs before touching storage."""

        cases = (
            (None, TypeError),
            (b"guide", TypeError),
            (1, TypeError),
            (True, TypeError),
            ("", ValueError),
            (" \t", ValueError),
        )
        for document_id, expected_error in cases:
            with self.subTest(document_id=repr(document_id)):
                events = []
                store = _TransactionalStore(events)
                indexer = Indexer(
                    object(), _SingleChunker(), _UnitEmbedder(), store
                )

                with self.assertRaises(expected_error):
                    indexer.delete(document_id)

                self.assertEqual(store._visible, {})
                self.assertEqual(events, [])

    def test_real_store_and_qasc_preserve_old_version_on_qasc_prepare_failure(self):
        """Concrete primary and QASC adapters remain aligned after failed staging."""

        embedder = _UnitEmbedder()
        segmenter = _FailingSentenceSegmenter()
        coordinator = IndexTransactionCoordinator()
        store = InMemoryVectorStore(transaction_coordinator=coordinator)
        qasc = QASCRetriever(
            segmenter,
            embedder,
            config=QASCConfig(window_radius=0),
            transaction_coordinator=coordinator,
        )
        indexer = Indexer(
            object(),
            _SingleChunker(),
            embedder,
            store,
            document_indexes=(qasc,),
            transaction_coordinator=coordinator,
        )
        indexer.index_document(self._document("old version"))
        segmenter.fail = True

        with self.assertRaisesRegex(
            RuntimeError, "injected QASC preparation failure"
        ):
            indexer.index_document(self._document("new version"))

        self.assertEqual(self._primary_text(store), "old version")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "old version",
        )

    def test_qasc_cannot_publish_under_a_forged_document_owner(self):
        """A segmenter cannot split primary and auxiliary lifecycle ownership."""

        class ForgingSentenceSegmenter:
            forge = True

            def segment(self, document):
                canonical = document.metadata["document_id"]
                if self.forge:
                    document.metadata["document_id"] = "forged"
                document_id = document.metadata["document_id"]
                return (
                    SentenceSpan(
                        "{}:0".format(document_id),
                        document_id,
                        document.text,
                        0,
                        0,
                        len(document.text),
                        document.metadata,
                    ),
                )

        coordinator = IndexTransactionCoordinator()
        embedder = _UnitEmbedder()
        segmenter = ForgingSentenceSegmenter()
        store = InMemoryVectorStore(transaction_coordinator=coordinator)
        qasc = QASCRetriever(
            segmenter,
            embedder,
            config=QASCConfig(window_radius=0),
            transaction_coordinator=coordinator,
        )
        indexer = Indexer(
            object(),
            _SingleChunker(),
            embedder,
            store,
            document_indexes=(qasc,),
            transaction_coordinator=coordinator,
        )

        with self.assertRaisesRegex(
            ComponentContractError, "canonical document metadata"
        ):
            indexer.index_document(self._document("rejected"))
        self.assertEqual(store.count, 0)
        self.assertEqual(qasc.document_count, 0)

        segmenter.forge = False
        indexer.index_document(self._document("accepted"))
        self.assertEqual(store.count, 1)
        self.assertEqual(qasc.document_count, 1)
        self.assertEqual(indexer.delete("guide"), 1)
        self.assertEqual(store.count, 0)
        self.assertEqual(qasc.document_count, 0)

    def test_real_store_and_qasc_handles_are_idempotent(self):
        """Repeated commit/rollback calls preserve each real handle's contract."""

        coordinator = IndexTransactionCoordinator()
        embedder = _UnitEmbedder()
        store = InMemoryVectorStore(transaction_coordinator=coordinator)
        qasc = QASCRetriever(
            _FailingSentenceSegmenter(),
            embedder,
            config=QASCConfig(window_radius=0),
            transaction_coordinator=coordinator,
        )
        indexer = Indexer(
            object(),
            _SingleChunker(),
            embedder,
            store,
            document_indexes=(qasc,),
            transaction_coordinator=coordinator,
        )
        indexer.index_document(self._document("old version"))

        new_document = self._document("new version")
        new_record = VectorRecord(
            Chunk(
                "guide:0",
                "guide",
                "new version",
                0,
                new_document.metadata,
            ),
            (1.0,),
        )
        replacements = (
            store.prepare_replace_document("guide", (new_record,)),
            qasc.prepare_replace_document(new_document),
        )
        for replacement in replacements:
            replacement.commit()
            replacement.commit()
        self.assertEqual(self._primary_text(store), "new version")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "new version",
        )
        for replacement in reversed(replacements):
            replacement.rollback()
            replacement.rollback()
        self.assertEqual(self._primary_text(store), "old version")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "old version",
        )

        deletions = (
            store.prepare_delete_document("guide"),
            qasc.prepare_delete_document("guide"),
        )
        for deletion in deletions:
            deletion.commit()
            deletion.commit()
        self.assertEqual(store.count, 0)
        self.assertEqual(qasc.document_count, 0)
        for deletion in reversed(deletions):
            deletion.rollback()
            deletion.rollback()
        self.assertEqual(self._primary_text(store), "old version")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "old version",
        )

    def test_real_store_and_qasc_stale_handles_cannot_clobber_a_winner(self):
        """Stale replacement and deletion handles leave the newer generation intact."""

        coordinator = IndexTransactionCoordinator()
        embedder = _UnitEmbedder()
        store = InMemoryVectorStore(transaction_coordinator=coordinator)
        qasc = QASCRetriever(
            _FailingSentenceSegmenter(),
            embedder,
            config=QASCConfig(window_radius=0),
            transaction_coordinator=coordinator,
        )
        indexer = Indexer(
            object(),
            _SingleChunker(),
            embedder,
            store,
            document_indexes=(qasc,),
            transaction_coordinator=coordinator,
        )
        indexer.index_document(self._document("old version"))

        def record(text):
            document = self._document(text)
            return VectorRecord(
                Chunk("guide:0", "guide", text, 0, document.metadata),
                (1.0,),
            )

        stale_replacements = (
            store.prepare_replace_document("guide", (record("stale"),)),
            qasc.prepare_replace_document(self._document("stale")),
        )
        winners = (
            store.prepare_replace_document("guide", (record("winner"),)),
            qasc.prepare_replace_document(self._document("winner")),
        )
        for winner in winners:
            winner.commit()
        for stale in stale_replacements:
            with self.assertRaisesRegex(ComponentContractError, "stale"):
                stale.commit()
            stale.rollback()
            stale.rollback()
        self.assertEqual(self._primary_text(store), "winner")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "winner",
        )

        stale_deletions = (
            store.prepare_delete_document("guide"),
            qasc.prepare_delete_document("guide"),
        )
        newer_winners = (
            store.prepare_replace_document("guide", (record("newer winner"),)),
            qasc.prepare_replace_document(self._document("newer winner")),
        )
        for winner in newer_winners:
            winner.commit()
        for stale in stale_deletions:
            with self.assertRaisesRegex(ComponentContractError, "stale"):
                stale.commit()
            stale.rollback()
            stale.rollback()
        self.assertEqual(self._primary_text(store), "newer winner")
        self.assertEqual(
            qasc.retrieve("question", top_k=1)[0].chunk.text,
            "newer winner",
        )

    def test_auxiliary_delete_failure_restores_every_participant(self):
        """A failed staged delete cannot leave primary and auxiliaries divided."""

        for failure_kind in ("prepare", "commit"):
            for failing_index in range(3):
                with self.subTest(
                    failure_kind=failure_kind,
                    failing_index=failing_index,
                ):
                    indexer, store, indexes, events = self._build_indexer()
                    indexer.index_document(self._document("old version"))
                    events.clear()
                    if failure_kind == "prepare":
                        indexes[failing_index].fail_delete_prepare = True
                    else:
                        indexes[failing_index].fail_delete_commit = True

                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        indexer.delete("guide")

                    self.assertEqual(self._primary_text(store), "old version")
                    self.assertEqual(
                        [index.retrieve_text("guide") for index in indexes],
                        ["old version", "old version", "old version"],
                    )

    def test_shared_coordinator_hides_intermediate_component_commits(self):
        """Readers wait until standard and QASC publish one complete generation."""

        class BlockingIndex:
            def __init__(self, inner):
                self.inner = inner
                self.transaction_coordinator = inner.transaction_coordinator
                self.block = False
                self.commit_entered = Event()
                self.release_commit = Event()

            def prepare_replace_document(self, document):
                prepared = self.inner.prepare_replace_document(document)
                owner = self

                class BlockingReplacement:
                    def commit(self):
                        if owner.block:
                            owner.commit_entered.set()
                            if not owner.release_commit.wait(1.0):
                                raise RuntimeError("timed out waiting to commit")
                        prepared.commit()

                    def rollback(self):
                        prepared.rollback()

                return BlockingReplacement()

            def replace_document(self, document):
                prepared = self.prepare_replace_document(document)
                prepared.commit()

            def prepare_delete_document(self, document_id):
                return self.inner.prepare_delete_document(document_id)

            def delete_document(self, document_id):
                return self.inner.delete_document(document_id)

        coordinator = IndexTransactionCoordinator()
        embedder = _UnitEmbedder()
        store = InMemoryVectorStore(transaction_coordinator=coordinator)
        qasc = QASCRetriever(
            _FailingSentenceSegmenter(),
            embedder,
            config=QASCConfig(window_radius=0),
            transaction_coordinator=coordinator,
        )
        blocking = BlockingIndex(qasc)
        indexer = Indexer(
            object(),
            _SingleChunker(),
            embedder,
            store,
            document_indexes=(blocking,),
            transaction_coordinator=coordinator,
        )
        indexer.index_document(self._document("old version"))
        blocking.block = True

        writer_errors = []
        writer = Thread(
            target=lambda: self._capture_error(
                writer_errors,
                lambda: indexer.index_document(self._document("new version")),
            )
        )
        writer.start()
        self.assertTrue(blocking.commit_entered.wait(1.0))

        reader_started = Event()
        reader_done = Event()
        observed = []

        def read_both():
            reader_started.set()
            primary = self._primary_text(store)
            auxiliary = qasc.retrieve("question", top_k=1)[0].chunk.text
            observed.append((primary, auxiliary))
            reader_done.set()

        reader = Thread(target=read_both)
        reader.start()
        self.assertTrue(reader_started.wait(1.0))
        self.assertFalse(reader_done.wait(0.05))

        blocking.release_commit.set()
        writer.join(1.0)
        reader.join(1.0)

        self.assertFalse(writer.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(writer_errors, [])
        self.assertEqual(observed, [("new version", "new version")])

    @staticmethod
    def _capture_error(errors, operation):
        try:
            operation()
        except Exception as error:
            errors.append(error)


if __name__ == "__main__":
    unittest.main()
