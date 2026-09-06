# Test rationale

The suite uses parameters that isolate a contract boundary or make ordering
deterministic. They are not production retrieval-quality targets. This document
records the rationale for tests added or materially refined during the contract
test expansion.

## Built-in ingestion handlers

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_pdf_loader_reports_pages_and_empty_pages` | PDF routing, page accounting, document metadata, and source provenance | A one-page blank PDF is generated in memory with `pypdf`; it deterministically exercises the empty-page path without checking parser-specific text layout, while title and author verify useful metadata extraction. |
| `test_pdf_loader_extracts_text_with_a_page_label` | Embedded PDF text is extracted and retains its human-readable page reference | A minimal one-page Helvetica PDF contains a single unique phrase, avoiding layout ambiguity while exercising the real `pypdf` parser. |
| `test_default_pipeline_ingests_supported_file_paths` | PDF, TXT, CSV, and HTML all work through the filesystem entry point | One minimal file per requested format verifies suffix routing, extracted text, and `source_path` provenance without involving retrieval ranking. |
| `test_text_normalization_handles_unicode_cr_and_outer_blank_lines` | NFC normalization, classic-Mac newlines, trailing spaces, and outer blank removal | A decomposed `e` + acute accent proves NFC composition; `\r` isolates CR conversion from the already-covered CRLF case; two outer CRs prove blank trimming without changing internal prose. |
| `test_json_loader_preserves_unicode_and_reports_non_object_roots` | Readable Unicode and correct root metadata for JSON arrays | A two-item array is the smallest non-object root containing both Unicode text and a nested object; `42` is a stable scalar with no formatting ambiguity. |
| `test_json_loader_rejects_malformed_input` | Invalid structured data is not silently indexed | `{"broken":` is valid enough to enter JSON parsing but necessarily incomplete, producing the standard `JSONDecodeError`. |
| `test_csv_loader_handles_empty_input` | Empty input has zero rows and columns | `b""` is the actual empty-file boundary; no fabricated header or row should appear. |
| `test_csv_loader_handles_bom_blank_headers_and_uneven_rows` | BOM decoding, fallback headers, and short/long rows | One blank header triggers `column_2`; a four-cell row against three headers triggers `column_4`; a two-cell row proves missing-cell padding. The final expectation ends at `role:` because global normalization intentionally removes trailing whitespace. |
| `test_html_loader_preserves_entities_and_block_boundaries` | Visible entities and paragraph/break structure while skipped elements stay hidden | `&amp;` exercises entity decoding, `<br/>` the self-closing block path, and `template`/`svg` two separate skipped-content categories. |

## Document pipeline

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_cleaner_prepend_and_replace_control_the_type_specific_chain` | Exact semantics of `prepend=True` and `replace=True` | Appending `A`, prepending `B`, then replacing with `C` makes order observable as `0BA` and replacement observable as `0C`; single characters avoid text-normalization effects. |
| `test_loader_must_preserve_the_resolved_document_type` | A dispatched loader cannot relabel its result | Registered `txt` returning `pdf` is the smallest unambiguous mismatch and verifies both sides of the error message. |
| `test_cleaner_must_return_a_document` | Cleaner output obeys the canonical model contract | A plain string is callable output but not a `Document`, isolating result validation from invocation failure. |
| `test_cleaner_must_preserve_the_document_type` | Cleaners cannot change routing identity | `txt` to `pdf` is a clear type mutation while leaving the text valid. |
| `test_source_arguments_reject_ambiguous_or_unsupported_inputs` | Provenance cannot be supplied twice, and invalid source classes fail early | A `DocumentSource` plus metadata tests duplicate provenance; a path plus `name` tests conflicting filesystem identity; integer `42` is neither bytes, path, nor `DocumentSource`. |

## Sentence segmentation

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_spacy_adapter_trims_spans_skips_blanks_and_generates_a_stable_id` | Correct offsets after trimming, blank-span removal, and deterministic fallback IDs | Two leading and one trailing spaces make the expected span exactly `[2, 8)`; a second all-space span must disappear; repeating the same `source_name` must reproduce the ID. |
| `test_spacy_adapter_rejects_a_non_callable_pipeline` | Invalid dependency injection fails at construction | `object()` is deliberately non-callable and has no spaCy-like accidental behavior. |
| `test_spacy_adapter_reports_missing_sentence_boundaries` | Boundaryless spaCy pipelines return the project-specific actionable error | The fake `sents` property raises the same `ValueError` shape as spaCy without requiring the optional package. |

## Existing modular RAG tests refined

| Test | Refinement | Parameter rationale |
| --- | --- | --- |
| `test_demo_factory_accepts_a_pipeline_with_a_new_format` | Moved from `ChunkerTests` to `FactoryIntegrationTests` and documented | No runtime parameter changed; the move fixes ownership because the test exercises composition and injection, not chunk boundaries. |
| `test_replacement_removes_old_chunks_without_touching_other_documents` | Replaced the unsupported “atomic” claim with assertions for stale-chunk removal and document isolation | Existing 2D vectors and `top_k=5` were retained: 2D makes cosine order obvious, and five is safely above the three possible records so truncation cannot hide stale data. |
| `test_demo_pipeline_indexes_retrieves_answers_and_cites` | Added chunk-count, result-count, citation-count, marker, and excerpt checks | Existing `max_words=20` and overlap `3` were retained so both short sentences form one evidence chunk; this isolates wiring and citation fidelity from ranking across chunks. |
| `test_reindex_and_delete_remove_stale_evidence_end_to_end` | Added application-level replacement and deletion coverage | `max_words=10` fits each five-word version in exactly one chunk; overlap `0` removes overlap as a confounder; the same `document_id` is required to exercise replacement. |
| `test_hashing_embedder_is_deterministic_normalized_and_input_sensitive` | Strengthened same-input equality with dimension, norm, different-input, and empty-input assertions | Dimension `32` was retained because it is small and fast while giving the two chosen token sets ample room to avoid a contrived collision; unit norm is the cosine-search contract, while empty text must remain the zero vector. |

## Core contract tests

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_chunker_rejects_invalid_window_configuration` | Constructor boundaries and guaranteed forward progress | `max_words=0` is the nonpositive boundary; overlap `-1` is the first negative value; overlap `4` equals a four-word window and would make the step zero; overlap `3` proves the largest valid value. |
| `test_chunker_generates_stable_source_based_document_ids` | Fallback IDs are deterministic and provenance-sensitive | Four words with window `3` and overlap `1` must start at words `0` and `2`; identical text under two source names proves provenance participates in identity. |
| `test_rejected_replacements_preserve_existing_records` | Store validation occurs before mutation | A valid 2D record is challenged by a wrong owner, a mixed 2D/3D batch, and an all-3D incompatible replacement; querying 2D afterward proves the original survived each failure. |
| `test_deleting_the_last_document_resets_vector_dimension` | Empty stores can migrate embedding dimensions | The change from 2D to 3D is the smallest visible dimension migration and occurs only after deletion of the sole document. |
| `test_search_supports_reserved_filters_and_deterministic_ties` | Reserved ID filters and stable tie-breaking | Two equal unit vectors force a score tie; reverse insertion order proves sorting uses chunk IDs; `top_k=2` includes both, and combined document/chunk filters select exactly one. |
| `test_search_rejects_nonpositive_top_k` | Invalid result limits fail explicitly | `0` tests the empty boundary and `-1` the negative domain. |
| `test_indexer_rejects_an_empty_chunk_sequence` | A processed document cannot report success with no evidence | An eight-dimensional embedder is a cheap valid collaborator but is intentionally never called; the empty tuple isolates the chunker contract. |
| `test_indexer_rejects_chunks_with_multiple_document_ids` | One replacement cannot mix document owners | Two chunks are the minimum sequence that can disagree; IDs `a` and `b` make the mismatch direct. |
| `test_indexer_updates_and_deletes_auxiliary_document_indexes` | Optional indexes follow primary replacement/deletion lifecycle | Window `10` with overlap `0` forces the three-word document into one chunk; dimension `8` keeps hashing cheap while remaining a valid nonempty vector. |
| `test_hashing_embedder_rejects_nonpositive_dimensions` | Embeddings always have at least one coordinate | `0` is the empty-vector boundary and `-1` covers the negative domain. |
| `test_structural_verifier_receives_the_answer_and_exact_contexts` | Third-party answer verifiers satisfy the protocol structurally and receive the generated answer plus its evidence unchanged | One result is the smallest nonempty evidence set; distinct `question` and `answer` strings make argument order observable, while score `0.9` is data forwarded to the adapter rather than a verification threshold. |
| `test_verification_result_validates_fields_and_snapshots_metadata` | Verification verdicts remain unambiguous and their top-level diagnostics cannot change through the caller's original mapping | `False` exercises an unsupported verdict; integer `1` must not masquerade as a boolean, `None` is the smallest invalid reason, and a single metadata key is sufficient to prove defensive copying. |
| `test_service_rejects_invalid_default_limits` | Service defaults are positive and fetch covers final selection | `(0,1)` invalidates final count, `(1,0)` invalidates fetch count, and `(3,2)` uses positive values but violates `fetch >= final`. |
| `test_service_rejects_invalid_requests_and_method_registrations` | Questions, per-call limits, method types, reserved names, and normalized duplicates are validated | Whitespace isolates empty text; `top_k=0` is the boundary; integer `7` is non-string; padded/case-varied method names prove validation happens after normalization. |
| `test_reranking_fetches_more_candidates_than_it_returns` | Candidate breadth and final response size remain separate | Fetch `4` and final `2` are deliberately different and small; reversing four candidates makes it observable that reranking saw all four before two were returned. |
| `test_citations_normalize_excerpts_and_apply_source_fallbacks` | Citation numbering, readable excerpts, and provenance fallback | Two results are the minimum for numbering; scores `0.9` and `0.8` keep their order deterministic but are not acceptance thresholds; missing source metadata on the second result forces document-ID fallback. |
| `test_qasc_configuration_requires_explicit_enablement` | Optional indexing cost cannot be activated accidentally | Segmenter-only and config-only cases cover both optional arguments independently while leaving `enable_qasc` at its explicit default `False`. |

## Cross-encoder reranking

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_scores_query_chunk_pairs_and_returns_the_best_results` | The adapter sends the correct pairs, honors inference settings, ranks by model relevance, and exposes those final scores | Three candidates make the reordering visible; `top_k=2` proves truncation happens after all candidate scores are considered; batch size `8` is distinct from both counts so argument forwarding is observable. |
| `test_equal_scores_preserve_retrieval_order` | Model ties remain deterministic without inventing a secondary relevance signal | Two equal `0.5` scores are the minimum tie; reverse-alphabetical IDs prove the original retrieval order, rather than chunk identity, resolves it. |
| `test_empty_candidates_skip_model_inference` | Empty retrieval results do not trigger unnecessary model calls | An empty tuple is the exact no-evidence boundary while a positive limit keeps validation separate. |
| `test_rejects_invalid_configuration_and_limits` | Model injection and all positive-size boundaries fail early | Zero is the lower boundary for batch, sequence, and result sizes; `object()` isolates the required `predict` capability. |
| `test_rejects_malformed_model_scores` | Provider output has one finite scalar per candidate | One candidate paired with zero scores, a nested score, and NaN independently exercises count, shape, and finiteness validation. |
| `test_factory_accepts_an_injected_reranker` | The composition root keeps the reranker replaceable | Identity comparison proves the supplied adapter reaches `RAGService` unchanged without requiring model inference. |

## QASC contracts

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_config_rejects_values_outside_each_supported_domain` | Every QASC parameter respects its mathematical domain | Percentiles `-0.1` and `100.1` sit just outside both inclusive limits; `-1` is the first invalid integer for radii/gaps; `-0.1` is a small invalid negative float for decay/factor; `0` and `100` prove valid percentile boundaries. |
| `test_replacement_rejects_empty_mixed_or_noncontiguous_sentence_spans` | Sentence batches are nonempty, single-owner, zero-based, and contiguous | Empty, two-owner, and index sequence `(0,2)` isolate the three distinct contract failures with the smallest inputs. |
| `test_replacement_rejects_invalid_sentence_embedding_batches` | Embedding batches match sentence count and have one nonzero dimension | Two sentences are paired with one vector, two empty vectors, and mixed 2D/3D vectors to isolate count, emptiness, and rectangularity. |
| `test_rejected_replacement_preserves_the_previous_sentence_index` | Invalid re-indexing preserves valid QASC evidence | Radius `0` makes one sentence produce exactly one window; changing the embedder batch to empty triggers validation without changing stored query vectors. |
| `test_query_rejects_empty_text_nonpositive_limits_and_bad_vectors` | Query validation precedes scoring | Whitespace tests semantic emptiness, `top_k=0` the lower boundary, `()` an empty embedding, and 3D against stored 2D the smallest clear mismatch. |
| `test_deleting_the_last_document_allows_a_new_embedding_dimension` | A fully empty QASC index supports model migration | The same one-sentence document moves from 2D to 3D only after deletion, separating dimension reset from replacement behavior. |

## Code-aware chunking

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_python_parser_reports_nested_symbols_without_executing_source` | Python structure is extracted with qualified parent relationships without importing the file | An undefined decorator and runtime name would fail if executed; one class, one method, and one async function cover nesting, decorators, symbol kinds, and inclusive line ranges. |
| `test_invalid_python_and_unknown_languages_use_bounded_fallbacks` | Syntax errors and unsupported languages remain indexable through safe line windows | One necessarily invalid function header forces Python fallback, while an unknown language isolates the no-parser path. |
| `test_symbol_chunks_have_traceable_metadata_and_contiguous_indexes` | Blank preambles cannot create gaps in chunk indexes, and symbol provenance reaches metadata | Two blank lines create an empty leading range; one four-line class keeps the expected output small while proving one-based source lines, parser identity, symbol identity, and breadcrumb composition. |
| `test_oversized_parent_splits_at_children_and_bounded_line_windows` | Oversized declarations respect the configured line limit without losing child identity | Twelve lines exceed `max_lines=4`; overlap `1` exercises advancing windows, and two separated children force prefix, gap, child, and tail ranges. |
| `test_large_span_sets_use_declared_parent_relationships` | Range construction stays practical for files containing many declarations | Five hundred children are large enough to expose repeated all-span scans while keeping the unit test fast; ten-line chunks separate the performance contract from single-line window configuration. |
| `test_routing_chunker_only_sends_code_and_config_to_code_chunker` | Adding repository support does not change ordinary document chunking | One prose, one code, and one configuration document cover all routing branches; unique return labels make the selected collaborator observable. |
| `test_configuration_rejects_windows_without_forward_progress` | Code line windows are positive and always advance | Zero is the invalid size boundary, `-1` is the first invalid overlap, and overlap equal to a four-line window would produce a zero step. |
| `test_injected_parser_extracts_nested_qualified_symbols` | Tree-sitter provider nodes are converted into provider-neutral spans | A three-line class with one method is the smallest tree that proves nested qualification, one-based line conversion, and injected-parser use without a native grammar. |
| `test_missing_injected_grammar_uses_builtin_fallback` | Missing optional grammars do not disable repository ingestion | An injected factory raises the project-specific optional-dependency error, while a one-function Python source proves the built-in AST result is returned. |
| `test_declared_tree_sitter_provider_imports_when_extra_is_installed` | Both Python-conditional code extras expose the API expected by the adapter | Python 3.9 checks the prebuilt `tree_sitter_languages` parser factory; newer Python checks the language-pack factory and downloaded-grammar inventory, then parses one function without requiring a network download. |

## Repository ingestion

| Test | What it protects | Parameter rationale |
| --- | --- | --- |
| `test_sensitive_paths_are_rejected_at_any_repository_depth` | Credential files and sensitive directories are rejected at repository root and below it | Root/nested `.env`, `.ssh`, `secrets`, private-key, and certificate paths cover name, directory, and suffix policies; `.env.example` and `secrets_manager.py` guard against overbroad matching. |
| `test_high_confidence_secret_shapes_are_redacted_with_categories` | Recognized credentials are removed and reported without exposing their values | One assignment, AWS-style access key, credential URL, and private-key block exercise every built-in redaction category exactly once. |
| `test_every_resource_limit_must_be_a_positive_non_boolean_integer` | Resource controls cannot be disabled or confused with booleans | `0`, `-1`, `True`, and `1.5` cover zero, negative, Python's boolean-as-integer edge case, and non-integral input. |
| `test_indexes_safe_files_and_reports_ignored_unsafe_and_binary_entries` | End-to-end discovery, classification, `.gitignore`, binary rejection, secret handling, trust flags, prompt-injection flags, code symbols, and excluded directories work together | A deliberately small repository contains one representative of each path; `code_max_lines=20` keeps the class together, and `top_k=50` is above the possible chunk count so assertions cannot be hidden by ranking truncation. |
| `test_complete_rescan_deletes_files_removed_from_the_repository` | A successful refresh removes stale vectors and manifest entries | Two files are the minimum snapshot that can delete one while proving the other remains; line limit `20` prevents chunking details from affecting lifecycle behavior. |
| `test_incomplete_discovery_preserves_files_outside_the_bounded_snapshot` | A truncated scan cannot misclassify unseen files as deletions | Two files with `max_files=1` force an incomplete snapshot while leaving one prior entry outside the observed set. |
| `test_per_file_chunk_limit_rejects_before_vectors_are_replaced` | Oversized files fail before partial vector replacement | Three nonempty lines with `code_max_lines=1` create three chunks against a limit of two; total limit ten ensures only the per-file boundary is tested. |
| `test_total_byte_limit_marks_the_snapshot_incomplete` | Cumulative repository bytes stop further processing without claiming a complete snapshot | Two five-byte files against a seven-byte total allow exactly one file before the boundary and keep per-file limits irrelevant. |
| `test_total_byte_limit_rechecks_the_bytes_actually_read` | A file that grows after discovery cannot bypass the cumulative byte budget | The discovered file is one byte but the safe reader returns eight against a seven-byte limit, isolating the second size check from the normal pre-read check. |
| `test_symlinked_repository_root_and_remote_urls_are_rejected` | Repository ingestion stays local and starts from an unambiguous real directory | An HTTPS URL covers remote input, and a directory symlink covers root aliasing without reading repository contents. |
| `test_repository_ids_are_validated_before_files_are_read` | Stable repository identity has an explicit safe format | Empty text, whitespace-containing text, and integer `7` independently exercise missing, unsafe-character, and wrong-type validation. |
| `test_failed_reindex_preserves_the_last_successful_manifest_entry` | A transient indexing failure cannot erase the last known-good manifest identity | One file is successfully indexed, changed, then rejected by an injected failure so the before/after manifest comparison is unambiguous. |
| `test_failed_stale_delete_keeps_the_manifest_entry_and_reports_failure` | A deletion failure cannot make the manifest deny evidence that still exists | One previously indexed file is removed from disk while its exact document ID is configured to fail deletion, isolating stale cleanup from discovery. |
| `test_falsey_manifest_is_still_injected` | Optional dependency injection uses `is not None` rather than collaborator truthiness | A manifest whose `__bool__` returns `False` is valid but exposes accidental use of `or` in factory or indexer construction. |
| `test_existing_three_argument_application_construction_remains_valid` | The new repository feature does not break direct construction of the pre-existing application container | Three `None` placeholders reproduce the old positional shape; calling the new method must then fail explicitly rather than during construction. |
| `test_custom_secret_scanner_must_return_the_declared_result_type` | Malformed security adapters cannot pass unsanitized text to indexing | One safe text file and a scanner returning a plain string prove validation happens before the indexer; the outcome is a content-free per-file failure report. |

## CI parameter change

The `spacy-adapter` job installs `.[spacy]` and runs only
`test_sentence_segmentation.py`. Python 3.9 is used because it is the project's
oldest supported interpreter and `pyproject.toml` already declares a dedicated
compatible spaCy range for it. The ten-minute timeout matches the core test job
and provides installation headroom without permitting a hung optional job.
