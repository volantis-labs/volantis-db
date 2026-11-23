#![allow(clippy::expect_used)]

use std::sync::Arc;
use tempfile::tempdir;
use volantis_db::ingest::*;
use volantis_db::lifecycle::embedded::*;
use volantis_db::lifecycle::*;
use volantis_db::meta::*;

#[path = "test_support/namespace_model.rs"]
mod namespace_model;
use namespace_model::Model;

#[test]
fn fresh_embedded_namespace_reads_expose_caller_metadata() {
    let dir = tempdir().expect("namespace");
    let store =
        Store::open_embedded(dir.path(), OpenOptions::new(), Arc::new(Model)).expect("open");
    let id = DocId::new(1);
    store
        .ingest(IngestBatch::new(vec![
            IngestDocument::new(DocumentVersion::new(id, Revision::new(1)), vec![])
                .with_text("target|other")
                .with_metadata(b"source".to_vec()),
        ]))
        .expect("ingest active row");

    let fields = DocumentFields::TEXT | DocumentFields::METADATA | DocumentFields::ATTRIBUTES;
    let check = |store: &Store| {
        let get = store
            .get_documents(&[id], fields)
            .expect("get")
            .pop()
            .flatten()
            .expect("live document");
        let scan = store
            .scan_documents(DocumentScanRequest::new(1, fields, running()))
            .expect("scan");
        let scanned = scan.documents.first().expect("scanned document");
        assert_eq!(
            (get.metadata.as_deref(), scanned.metadata.as_deref()),
            (Some(b"source".as_slice()), Some(b"source".as_slice()))
        );
    };
    check(&store);
    let view = store.open_snapshot().expect("snapshot of active row");
    check(&view);
    drop(view);
    store.seal().expect("seal");
    check(&store);
}

#[test]
fn filtered_query_returns_k_matching_chunks_before_top_k() {
    let dir = tempdir().expect("directory");
    let column = ColumnId::new(1);
    let sequence = ColumnId::new(2);
    let schema = Schema::new(vec![
        ColumnDefinition::new(column, "meeting", ColumnType::Id128, false),
        ColumnDefinition::new(sequence, "sequence", ColumnType::U64, false),
    ])
    .expect("schema");
    let store = Store::open_embedded(
        dir.path().join("notes"),
        OpenOptions::new().with_schema(schema),
        Arc::new(Model),
    )
    .expect("open");
    let wanted = DocId::new(u128::MAX);
    let value = PredicateValue::Id128(wanted);
    let mut docs = (1..=100)
        .map(|n| {
            IngestDocument::new(
                DocumentVersion::new(DocId::new(n), Revision::new(1)),
                vec![],
            )
            .with_text("target")
            .with_columns(vec![
                (column, PredicateValue::Id128(DocId::new(1))),
                (sequence, PredicateValue::U64(0)),
            ])
        })
        .collect::<Vec<_>>();
    docs.push(
        IngestDocument::new(DocumentVersion::new(wanted, Revision::new(7)), vec![])
            .with_text("café target|target résumé")
            .with_timestamp(20)
            .with_columns(vec![
                (column, value.clone()),
                (sequence, PredicateValue::U64(9)),
            ]),
    );
    store.ingest(IngestBatch::new(docs)).expect("ingest");
    store.seal().expect("seal");
    let snapshot = store.open_snapshot().expect("snapshot");
    let accept = |p: Predicate| {
        QueryFilter::new(store.schema(), Some(&p), None)
            .expect("valid filter")
            .expect("predicate")
    };
    let filters = [
        accept(Predicate::Eq {
            column,
            value: value.clone(),
        }),
        accept(Predicate::In {
            column,
            values: vec![value.clone()],
        }),
        accept(Predicate::Not(Box::new(Predicate::Eq {
            column,
            value: PredicateValue::Id128(DocId::new(1)),
        }))),
        accept(Predicate::Range(RangePredicate {
            column: sequence,
            lower: Some(RangeBound::inclusive(PredicateValue::U64(9))),
            upper: None,
        })),
        QueryFilter::new(store.schema(), None, Some((20, 21)))
            .expect("timestamp range")
            .expect("predicate"),
        QueryFilter::eligible(store.schema(), &[wanted]),
    ];
    for filter in filters {
        for leg in [
            EmbeddedLeg::Dense,
            EmbeddedLeg::Lexical,
            EmbeddedLeg::Hybrid,
        ] {
            let hits = snapshot
                .query_embedded(
                    "target",
                    leg,
                    2,
                    Some(&filter),
                    QueryControl::Cancel(CancelToken::new()),
                )
                .expect("query");
            assert_eq!(hits.len(), 2);
            let mut spans: Vec<_> = hits
                .into_iter()
                .map(|h| {
                    assert_eq!(h.version, DocumentVersion::new(wanted, Revision::new(7)));
                    assert_eq!(
                        h.attributes,
                        vec![(column, value.clone()), (sequence, PredicateValue::U64(9))]
                    );
                    (h.chunk, h.bytes.start, h.bytes.end, h.text)
                })
                .collect();
            spans.sort();
            assert_eq!(
                spans,
                vec![
                    (0, 0, 12, "café target".into()),
                    (1, 13, 28, "target résumé".into())
                ]
            );
        }
    }
}

#[test]
fn namespace_batch_backup_revision_and_cascade_keep_chunk_ownership() {
    let root = tempdir().expect("namespace root");
    let owner = ColumnId::new(1);
    let parent_id = DocId::new(u128::MAX - 1);
    let child_id = DocId::new(u128::MAX);
    let schema = Schema::new(vec![ColumnDefinition::new(
        owner,
        "owner",
        ColumnType::Id128,
        false,
    )])
    .expect("schema");
    let embedded = OpenOptions::new()
        .with_schema(schema)
        .with_embedder(Arc::new(Model))
        .expect("model");
    let participants = || {
        vec![
            NamespaceMutation {
                name: "parents".into(),
                options: OpenOptions::new(),
                upserts: vec![],
                deletes: vec![],
                delete_where: None,
            },
            NamespaceMutation {
                name: "notes".into(),
                options: embedded.clone(),
                upserts: vec![],
                deletes: vec![],
                delete_where: None,
            },
        ]
    };
    let parent_path = root.path().join("parents");
    let child_path = root.path().join("notes");
    drop(Store::open(parent_path, OpenOptions::new()).expect("parents"));
    let notes = Store::open(&child_path, embedded.clone()).expect("notes");
    #[cfg(feature = "graph-cypher")]
    notes.enable_graph().expect("unified graph");
    drop(notes);
    namespace_declare_cascade(
        root.path(),
        participants(),
        CascadeRule {
            parent: "parents".into(),
            child: "notes".into(),
            attribute: owner,
        },
    )
    .expect("cascade");
    let document = |r, t: &str| {
        IngestDocument::new(DocumentVersion::new(child_id, Revision::new(r)), vec![])
            .with_text(t)
            .with_metadata(vec![42])
            .with_columns(vec![(owner, PredicateValue::Id128(parent_id))])
    };
    let mut batch = participants();
    batch
        .first_mut()
        .expect("parent")
        .upserts
        .push(IngestDocument::new(
            DocumentVersion::new(parent_id, Revision::new(1)),
            vec![1., 0.],
        ));
    batch
        .last_mut()
        .expect("child")
        .upserts
        .push(document(1, "first|second"));
    assert_eq!(
        namespace_batch(root.path(), batch)
            .expect("atomic batch")
            .len(),
        2
    );
    let store = Store::open(&child_path, embedded.clone()).expect("writer");
    let backup = root.path().join("backup");
    store.write_snapshot(&backup).expect("backup");
    let verification = volantis_db::verify::verify_store(&backup).expect("verify");
    assert!(verification.is_clean(), "{verification:?}");
    let inspected = Store::open_for_inspection(&backup).expect("inspection");
    let record = inspected
        .get_documents(&[child_id], DocumentFields::ALL)
        .expect("inspect rows")
        .pop()
        .flatten()
        .expect("child row");
    assert_eq!(record.metadata, Some(vec![42]));
    drop(inspected);
    let restored = Store::open_embedded(&backup, OpenOptions::read_only(), Arc::new(Model))
        .expect("read-only backup");
    let search = |s: &Store| {
        s.query_embedded(
            "second",
            EmbeddedLeg::Dense,
            8,
            None,
            QueryControl::Cancel(CancelToken::new()),
        )
        .expect("query")
    };
    assert_eq!(search(&restored).len(), 2);
    let view = store.open_snapshot().expect("read view");
    store
        .ingest(IngestBatch::new(vec![document(2, "new")]))
        .expect("revision");
    assert_eq!(search(&store).len(), 1);
    assert_eq!(search(&view).len(), 2);
    let row = store
        .get_documents(&[child_id], DocumentFields::ALL)
        .expect("get")
        .pop()
        .flatten()
        .expect("document");
    assert_eq!(row.metadata, Some(vec![42]));
    assert_eq!(row.text.as_deref(), Some("new"));
    drop(view);
    store.seal().expect("seal before delete-only batch");
    drop(store);
    let mut cascade = participants();
    cascade.first_mut().expect("parent").deletes.push(parent_id);
    namespace_delete_cascade(root.path(), cascade).expect("cascade delete");
    let clean = Store::open(&child_path, embedded).expect("reopen");
    assert!(search(&clean).is_empty());
    let erased = root.path().join("erased-backup");
    clean.write_snapshot(&erased).expect("backup after cascade");
    assert!(
        volantis_db::verify::verify_store(&erased)
            .expect("verify deletion")
            .is_clean()
    );
    assert_eq!(search(&restored).len(), 2);
}

#[cfg(feature = "graph-cypher")]
#[test]
fn mixed_graph_batch_applies_the_embedded_document_contract() {
    use volantis_db::property_graph::staging::{StructuredOperation, StructuredWrite, WriteImage};
    use volantis_db::property_graph::{
        ApplicationKey, CanonicalContents, EntityKind, GraphBatch, GraphRevision,
        GraphStoreErrorKind, GraphWriteOutcome,
    };

    let root = tempdir().expect("namespace");
    let store =
        Store::open_embedded(root.path(), OpenOptions::new(), Arc::new(Model)).expect("open");
    store.enable_graph().expect("graph");
    let doc = |id, vector, text: &str| {
        IngestDocument::new(
            DocumentVersion::new(DocId::new(id), Revision::new(1)),
            vector,
        )
        .with_text(text)
    };
    store
        .ingest(IngestBatch::new(vec![doc(1, vec![], "target")]))
        .expect("seed");
    let control = QueryControl::Cancel(CancelToken::new());
    let image = CanonicalContents::node(&mut [], &mut [], None, None).expect("node");
    let writes = [StructuredWrite {
        key: ApplicationKey::new(EntityKind::Node, "mixed", "node").expect("key"),
        revision: GraphRevision::new(1).expect("revision"),
        operation: StructuredOperation::Create,
        image: Some(WriteImage::Node(&image)),
    }];
    let apply = |documents: &IngestBatch| {
        store.graph_apply(
            GraphBatch {
                documents: Some(documents),
                writes: &writes,
                node_documents: &[],
            },
            &control,
        )
    };
    let rejected = IngestBatch::new(vec![doc(2, vec![1., 0.], "caller vector")])
        .with_epoch(store.epoch_identity().expect("namespace epoch"));
    let err = apply(&rejected).expect_err("caller vector must abort the mixed batch");
    assert_eq!(err.kind(), GraphStoreErrorKind::InvalidRequest);
    assert!(err.to_string().contains("without a caller vector"));
    let query = |leg| {
        store
            .query_embedded("target", leg, 8, None, control.clone())
            .expect("embedded query")
    };
    assert_eq!(query(EmbeddedLeg::Dense).len(), 1);
    let valid = IngestBatch::new(vec![doc(2, vec![], "target|target")]);
    assert!(matches!(
        apply(&valid)
            .expect("same graph create with text")
            .outcome(),
        GraphWriteOutcome::Committed { .. }
    ));
    let check = || {
        for leg in [
            EmbeddedLeg::Dense,
            EmbeddedLeg::Lexical,
            EmbeddedLeg::Hybrid,
        ] {
            let hits = query(leg);
            assert_eq!(hits.len(), 3);
            assert_eq!(
                hits.iter()
                    .filter(|h| h.version.doc_id() == DocId::new(2))
                    .count(),
                2
            );
        }
    };
    check();
    store.seal().expect("seal");
    check();
    assert!(
        volantis_db::verify::verify_store(root.path())
            .expect("verify")
            .is_clean()
    );
}

fn neighbour_store() -> (tempfile::TempDir, Store) {
    let dir = tempdir().expect("namespace directory");
    let column = ColumnId::new(1);
    let schema = Schema::new(vec![ColumnDefinition::new(
        column,
        "meeting",
        ColumnType::U64,
        false,
    )])
    .expect("schema");
    let store = Store::open_embedded(
        dir.path(),
        OpenOptions::new().with_schema(schema),
        Arc::new(Model),
    )
    .expect("open");
    let docs = [
        (u128::MAX, "target|other", 1),
        (2, "target", 1),
        (3, "other|other", 2),
    ]
    .into_iter()
    .map(|(id, text, meeting)| {
        IngestDocument::new(
            DocumentVersion::new(DocId::new(id), Revision::new(1)),
            vec![],
        )
        .with_text(text)
        .with_columns(vec![(column, PredicateValue::U64(meeting))])
    })
    .collect();
    store.ingest(IngestBatch::new(docs)).expect("ingest");
    store.seal().expect("seal");
    // Read persisted vectors without registering either model tower.
    drop(store);
    let reader = Store::open_for_inspection(dir.path()).expect("read without model");
    (dir, reader)
}

fn running() -> QueryControl {
    QueryControl::Cancel(CancelToken::new())
}

#[test]
fn nearest_by_id_excludes_source_by_default_and_includes_it_when_requested() {
    let (_dir, store) = neighbour_store();
    let id = DocId::new(u128::MAX);
    let options = NeighbourOptions {
        k: Some(8),
        ..Default::default()
    };
    let hits = store
        .nearest_chunks(id, None, &options, running())
        .expect("whole document");
    assert_eq!(
        hits.iter()
            .map(|h| (h.id, h.chunk, h.score))
            .collect::<Vec<_>>(),
        [
            (DocId::new(2), 0, 0.),
            (DocId::new(3), 0, 0.),
            (DocId::new(3), 1, 0.)
        ]
    );
    let options = NeighbourOptions {
        include_source: true,
        ..options
    };
    let hits = store
        .nearest_chunks(id, Some(1), &options, running())
        .expect("second source chunk");
    assert_eq!(
        hits.iter()
            .map(|h| (h.id, h.chunk, h.score))
            .collect::<Vec<_>>(),
        [
            (DocId::new(3), 0, 0.),
            (DocId::new(3), 1, 0.),
            (id, 1, 0.),
            (DocId::new(2), 0, -2.),
            (id, 0, -2.)
        ]
    );
}

#[test]
fn nearest_batch_matches_repeated_single_calls_in_input_order() {
    let (_dir, store) = neighbour_store();
    let sources = [
        (DocId::new(3), None),
        (DocId::new(u128::MAX), Some(0)),
        (DocId::new(3), None),
    ];
    let opts = NeighbourOptions {
        k: Some(2),
        ..Default::default()
    };
    let batch = store
        .nearest_chunks_batch(&sources, &opts, running())
        .expect("batch");
    assert_eq!(batch.len(), sources.len());
    for ((id, chunk), group) in sources.into_iter().zip(batch) {
        assert_eq!(
            group,
            store
                .nearest_chunks(id, chunk, &opts, running())
                .expect("single")
        );
    }
}

#[test]
fn nearest_filter_excludes_source_meeting_before_top_k() {
    let (_dir, store) = neighbour_store();
    let filter = QueryFilter::new(
        store.schema(),
        Some(&Predicate::Not(Box::new(Predicate::Eq {
            column: ColumnId::new(1),
            value: PredicateValue::U64(1),
        }))),
        None,
    )
    .expect("valid filter");
    let opts = NeighbourOptions {
        k: Some(2),
        filter: filter.as_ref(),
        ..Default::default()
    };
    let hits = store
        .nearest_chunks(DocId::new(u128::MAX), Some(0), &opts, running())
        .expect("filtered");
    assert_eq!(
        hits.iter()
            .map(|h| (h.id, h.chunk, h.score))
            .collect::<Vec<_>>(),
        [(DocId::new(3), 0, -2.), (DocId::new(3), 1, -2.)]
    );
}

#[test]
fn nearest_threshold_returns_only_scores_strictly_above_it() {
    let dir = tempdir().expect("namespace directory");
    let store =
        Store::open_embedded(dir.path(), OpenOptions::new(), Arc::new(Model)).expect("open");
    let id = DocId::new(1);
    let text = format!("{}other", "target|".repeat(12));
    store
        .ingest(IngestBatch::new(vec![
            IngestDocument::new(DocumentVersion::new(id, Revision::new(1)), vec![]).with_text(text),
        ]))
        .expect("ingest");
    for (threshold, k, count) in [
        (-3., None, 13),
        (-2., None, 12),
        (-2., Some(2), 2),
        (0., None, 0),
    ] {
        let options = NeighbourOptions {
            k,
            threshold: Some(threshold),
            include_source: true,
            filter: None,
        };
        let hits = store
            .nearest_chunks(id, Some(0), &options, running())
            .expect("threshold");
        assert_eq!(hits.len(), count);
        assert!(hits.iter().all(|h| h.score > threshold));
    }
}

#[test]
fn nearest_batch_deadline_and_cancellation_return_no_partial_results() {
    let (_dir, store) = neighbour_store();
    let sources = [(DocId::new(2), None), (DocId::new(3), None)];
    let token = CancelToken::new();
    token.cancel();
    let run = |control| store.nearest_chunks_batch(&sources, &NeighbourOptions::default(), control);
    assert!(matches!(
        run(QueryControl::Cancel(token)),
        Err(QueryError::Cancelled { partial: false })
    ));
    let deadline = Deadline::after(std::time::Duration::ZERO).expect("deadline");
    assert!(matches!(
        run(QueryControl::Deadline(deadline)),
        Err(QueryError::Timeout { partial: false })
    ));
}

#[test]
fn nearest_unknown_document_or_chunk_is_a_typed_error() {
    let (_dir, store) = neighbour_store();
    for (id, chunk) in [(DocId::new(42), None), (DocId::new(2), Some(1))] {
        let result = store.nearest_chunks_batch(
            &[(DocId::new(3), None), (id, chunk)],
            &NeighbourOptions::default(),
            running(),
        );
        assert!(
            matches!(result, Err(QueryError::Store(StoreError::NeighbourSourceNotFound { id: missing, chunk: ordinal })) if missing == id && ordinal == chunk)
        );
    }
}

#[test]
fn same_text_reuse_updates_the_whole_revision() {
    use std::sync::atomic::Ordering;
    let dir = tempdir().expect("namespace");
    let model = Arc::new(namespace_model::CountingModel::default());
    let column = ColumnId::new(1);
    let schema = Schema::new(vec![ColumnDefinition::new(
        column,
        "status",
        ColumnType::RawString,
        false,
    )])
    .expect("schema");
    let store = Store::open_embedded(
        dir.path(),
        OpenOptions::new().with_schema(schema),
        model.clone(),
    )
    .expect("open");
    let id = DocId::new(59601);
    let doc = |id, rev, time, label: &str, meta: &[u8]| {
        IngestDocument::new(DocumentVersion::new(id, Revision::new(rev)), vec![])
            .with_text("target|faké")
            .with_timestamp(time)
            .with_columns(vec![(column, PredicateValue::String(label.into()))])
            .with_metadata(meta.to_vec())
    };
    store
        .ingest(IngestBatch::new(vec![doc(
            id,
            1,
            101,
            "fake-before",
            &[0, 255],
        )]))
        .expect("insert");
    let snapshot = store.open_snapshot().expect("snapshot");
    store
        .ingest(IngestBatch::new(vec![doc(
            id,
            2,
            202,
            "fake-after",
            &[7, 0, 9],
        )]))
        .expect("replace");
    assert_eq!(model.documents.load(Ordering::SeqCst), 1);
    for (reader, rev, time, label, metadata) in [
        (&store, 2, 202, "fake-after", &[7, 0, 9][..]),
        (&snapshot, 1, 101, "fake-before", &[0, 255][..]),
    ] {
        let row = reader
            .get_documents(&[id], DocumentFields::ALL)
            .expect("get")
            .pop()
            .flatten()
            .expect("row");
        let scan = reader
            .scan_documents(DocumentScanRequest::new(1, DocumentFields::ALL, running()))
            .expect("scan");
        for record in [&row, scan.documents.first().expect("row")] {
            assert_eq!(record.revision, Revision::new(rev));
            assert_eq!(record.timestamp, time);
            assert_eq!(record.metadata.as_deref(), Some(metadata));
            assert_eq!(
                record.attributes,
                Some(vec![(column, PredicateValue::String(label.into()))])
            );
            assert_eq!(record.vector.as_deref(), Some(&[1., 0.][..]));
        }
        let hits = reader
            .query_embedded("target", EmbeddedLeg::Dense, 2, None, running())
            .expect("chunks");
        assert_eq!(
            hits.iter()
                .map(|h| (h.chunk, h.bytes.clone(), h.score))
                .collect::<Vec<_>>(),
            vec![(0, 0..6, 0.), (1, 7..12, -2.)]
        );
    }
    for rev in [1, 2] {
        store
            .ingest(IngestBatch::new(
                (59700..59732)
                    .map(|n| doc(DocId::new(n), rev, 202, "fake-after", &[]))
                    .collect(),
            ))
            .expect("32 documents");
        assert_eq!(model.documents.load(Ordering::SeqCst), 33);
    }
}

fn reuse_doc(id: u128, rev: u64, text: &str) -> IngestDocument {
    IngestDocument::new(
        DocumentVersion::new(DocId::new(id), Revision::new(rev)),
        vec![],
    )
    .with_text(text)
    .with_timestamp(rev as i64)
}
fn reuse_row(store: &Store, id: u128) -> Option<StoredDocument> {
    store
        .get_documents(&[DocId::new(id)], DocumentFields::ALL)
        .expect("point lookup")
        .pop()
        .flatten()
}
fn calls(model: &namespace_model::CountingModel) -> usize {
    model.documents.load(std::sync::atomic::Ordering::SeqCst)
}
#[test]
fn same_text_reuse_after_seal_merge_and_reopen() {
    let dir = tempdir().expect("directory");
    let model = Arc::new(namespace_model::CountingModel::default());
    let store = Store::open_embedded(dir.path(), OpenOptions::new(), model.clone()).expect("open");
    let mut last = 0;
    for (id, text) in [(59601, "target|faké"), (59602, "target|虚構")] {
        store
            .ingest(IngestBatch::new(vec![reuse_doc(id, 1, text)]))
            .expect("seed");
        last = store.seal().expect("seal");
    }
    assert_eq!(store.snapshot().expect("snapshot").segments().len(), 2);
    assert!(store.merge_sealed().expect("merge") > last);
    assert_eq!(store.snapshot().expect("snapshot").segments().len(), 1);
    assert_eq!(calls(&model), 2);
    drop(store);
    let model = Arc::new(namespace_model::CountingModel::default());
    let store =
        Store::open_embedded(dir.path(), OpenOptions::new(), model.clone()).expect("reopen");
    store
        .ingest(IngestBatch::new(vec![reuse_doc(59601, 2, "target|faké")]))
        .expect("replace sealed");
    assert_eq!(calls(&model), 0);
    let hits = store
        .query_embedded("target", EmbeddedLeg::Dense, 4, None, running())
        .expect("chunks");
    assert_eq!(
        hits.iter()
            .filter(|h| h.version == DocumentVersion::new(DocId::new(59601), Revision::new(2)))
            .map(|h| (h.bytes.clone(), h.score))
            .collect::<Vec<_>>(),
        vec![(0..6, 0.), (7..12, -2.)]
    );
}

#[test]
fn same_text_reuse_requires_live_id_exact_bytes_and_epoch() {
    let dir = tempdir().expect("directory");
    let model = Arc::new(namespace_model::CountingModel::default());
    let store = Store::open_embedded(dir.path(), OpenOptions::new(), model.clone()).expect("open");
    let write = |id, rev, text| {
        store
            .ingest(IngestBatch::new(vec![reuse_doc(id, rev, text)]))
            .expect("write")
    };
    write(59601, 1, "target|faké");
    write(59602, 1, "target|faké");
    assert_eq!(calls(&model), 2);
    write(59601, 2, "target|fake\u{301}");
    assert_eq!(calls(&model), 3);
    write(59601, 3, "target");
    assert_eq!(calls(&model), 4);
    let hits = store
        .query_embedded("target", EmbeddedLeg::Dense, 4, None, running())
        .expect("chunks");
    assert_eq!(
        hits.iter()
            .filter(|h| h.version.doc_id() == DocId::new(59601))
            .count(),
        1
    );
    store
        .delete(DeleteBatch::new(vec![DocId::new(59601)]))
        .expect("delete");
    write(59601, 4, "target");
    assert_eq!(calls(&model), 5);
    store
        .await_physical_purge(store.purge(&[DocId::new(59601)]).expect("schedule"))
        .expect("purge");
    write(59601, 5, "target");
    assert_eq!(calls(&model), 6);
    let changed = Arc::new(namespace_model::CountingModel {
        different_epoch: true,
        ..Default::default()
    });
    let options = OpenOptions::new()
        .with_embedder(changed.clone())
        .expect("options");
    let incompatible = options.declared_epoch().expect("epoch").identity();
    assert!(matches!(
        store
            .ingest(IngestBatch::new(vec![reuse_doc(59601, 6, "target")]).with_epoch(incompatible)),
        Err(IngestError::EpochMismatch(_))
    ));
    assert_eq!(
        reuse_row(&store, 59601).expect("live").revision,
        Revision::new(5)
    );
    drop(store);
    assert!(matches!(
        Store::open(dir.path(), options),
        Err(StoreError::EpochMismatch(_))
    ));
}

#[test]
fn same_text_reuse_preserves_revision_order_and_atomic_failure() {
    let dir = tempdir().expect("directory");
    let model = Arc::new(namespace_model::CountingModel::default());
    let store = Store::open_embedded(dir.path(), OpenOptions::new(), model.clone()).expect("open");
    let a = |rev, text| reuse_doc(59601, rev, text);
    store
        .ingest(IngestBatch::new(vec![a(1, "old")]))
        .expect("seed");
    store
        .ingest(IngestBatch::new(vec![
            a(2, "target"),
            a(2, "ignored"),
            a(3, "target").with_metadata(vec![7, 0, 9]),
        ]))
        .expect("ordered");
    assert_eq!(calls(&model), 3);
    let expected = reuse_row(&store, 59601).expect("live");
    assert_eq!(expected.revision, Revision::new(3));
    assert_eq!(expected.metadata, Some(vec![7, 0, 9]));
    for bad in [
        a(1, "target"),
        a(4, "target").with_expected_revision(ExpectedRevision::Absent),
        reuse_doc(59602, 2, "!fail"),
        IngestDocument::new(a(4, "target").version(), vec![1., 0.]).with_text("target"),
        IngestDocument::new(a(4, "target").version(), vec![]),
    ] {
        assert!(
            store
                .ingest(IngestBatch::new(vec![a(4, "target"), bad]))
                .is_err()
        );
        assert_eq!(reuse_row(&store, 59601), Some(expected.clone()));
        assert!(reuse_row(&store, 59602).is_none());
    }
}

fn mutation(name: &str, options: OpenOptions, upserts: Vec<IngestDocument>) -> NamespaceMutation {
    NamespaceMutation {
        name: name.into(),
        options,
        upserts,
        deletes: vec![],
        delete_where: None,
    }
}

#[test]
fn same_text_reuse_in_namespace_batch() {
    let root = tempdir().expect("root");
    let model_a = Arc::new(namespace_model::CountingModel::default());
    let model_b = Arc::new(namespace_model::CountingModel::default());
    let options_a = OpenOptions::new()
        .with_embedder(model_a.clone())
        .expect("model");
    let options_b = OpenOptions::new()
        .with_embedder(model_b.clone())
        .expect("model");
    let a = Store::open(root.path().join("a"), options_a.clone()).expect("open");
    let b = Store::open(root.path().join("b"), options_b.clone()).expect("open");
    let doc = |rev, text| reuse_doc(59601, rev, text);
    for store in [&a, &b] {
        store
            .ingest(IngestBatch::new(vec![doc(1, "target")]))
            .expect("seed");
    }
    let apply = |rev, options| {
        namespace_batch_live(
            root.path(),
            vec![
                LiveNamespaceMutation {
                    store: &a,
                    mutation: mutation("a", options_a.clone(), vec![doc(rev, "target")]),
                },
                LiveNamespaceMutation {
                    store: &b,
                    mutation: mutation("b", options, vec![doc(rev, "target|虚構")]),
                },
            ],
        )
    };
    apply(2, options_b.clone()).expect("commit");
    assert_eq!((calls(&model_a), calls(&model_b)), (1, 2));
    let wrong = OpenOptions::new()
        .with_embedder(Arc::new(namespace_model::CountingModel {
            different_epoch: true,
            ..Default::default()
        }))
        .expect("model");
    assert!(apply(3, wrong).is_err());
    for store in [&a, &b] {
        assert_eq!(
            reuse_row(store, 59601).expect("live").revision,
            Revision::new(2)
        );
    }
    drop((a, b));
    let before = (calls(&model_a), calls(&model_b));
    namespace_batch(
        root.path(),
        vec![
            mutation("a", options_a.clone(), vec![doc(3, "target")]),
            mutation("b", options_b.clone(), vec![doc(3, "target|虚構")]),
        ],
    )
    .expect("path batch");
    assert_eq!((calls(&model_a), calls(&model_b)), before);
    for (name, options) in [("a", options_a), ("b", options_b)] {
        let store = Store::open(root.path().join(name), options).expect("reopen");
        assert_eq!(
            reuse_row(&store, 59601).expect("live").revision,
            Revision::new(3)
        );
    }
}

#[cfg(feature = "graph-cypher")]
#[test]
fn same_text_reuse_in_graph_writes() {
    use volantis_db::property_graph::{staging::*, *};
    let root = tempdir().expect("namespace");
    let model = Arc::new(namespace_model::CountingModel::default());
    let store = Store::open_embedded(root.path(), OpenOptions::new(), model.clone()).expect("open");
    store.enable_graph().expect("graph");
    let image = CanonicalContents::node(&mut [], &mut [], None, None).expect("image");
    let key = ApplicationKey::new(EntityKind::Node, "reuse", "linked").expect("key");
    let node = NodeId::new(59601).expect("node");
    for rev in [1, 2] {
        let writes = [StructuredWrite {
            key,
            revision: GraphRevision::new(rev).expect("revision"),
            operation: if rev == 1 {
                StructuredOperation::Create
            } else {
                StructuredOperation::Put(EntityId::Node(node))
            },
            image: Some(WriteImage::Node(&image)),
        }];
        let docs = [GraphNodeDocument {
            item_index: 0,
            document: reuse_doc(59601, rev, "target|faké"),
        }];
        let result = store
            .graph_apply(GraphBatch::with_node_documents(&writes, &docs), &running())
            .expect("linked write");
        assert!(matches!(
            result.outcome(),
            GraphWriteOutcome::Committed { .. }
        ));
        assert_eq!(calls(&model), 1);
        assert_eq!(
            reuse_row(&store, 59601).expect("linked").timestamp,
            rev as i64
        );
    }
    store
        .ingest(IngestBatch::new(vec![reuse_doc(59602, 1, "target|虚構")]))
        .expect("seed");
    let writes = [StructuredWrite {
        key: ApplicationKey::new(EntityKind::Node, "reuse", "other").expect("key"),
        revision: GraphRevision::new(1).expect("revision"),
        operation: StructuredOperation::Create,
        image: Some(WriteImage::Node(&image)),
    }];
    let documents = IngestBatch::new(vec![reuse_doc(59602, 2, "target|虚構")]);
    let result = store
        .graph_apply(
            GraphBatch {
                writes: &writes,
                documents: Some(&documents),
                node_documents: &[],
            },
            &running(),
        )
        .expect("mixed write");
    assert!(matches!(
        result.outcome(),
        GraphWriteOutcome::Committed { .. }
    ));
    assert_eq!(calls(&model), 2);
    assert_eq!(reuse_row(&store, 59602).expect("live").timestamp, 2);
}

#[test]
fn same_text_reuse_race_rechecks_publication() {
    #[derive(Clone, Copy)]
    enum Writer {
        Ordinary,
        Namespace,
        #[cfg(feature = "graph-cypher")]
        Graph,
    }
    for writer in [
        Writer::Ordinary,
        Writer::Namespace,
        #[cfg(feature = "graph-cypher")]
        Writer::Graph,
    ] {
        for conditional in [false, true] {
            let root = tempdir().expect("root");
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let model = Arc::new(namespace_model::CountingModel {
                pause: Some(barrier.clone()),
                ..Default::default()
            });
            let options = OpenOptions::new()
                .with_embedder(model.clone())
                .expect("options");
            let store = Store::open(root.path().join("a"), options.clone()).expect("open");
            let other = Store::open(root.path().join("b"), OpenOptions::new()).expect("other");
            #[cfg(feature = "graph-cypher")]
            if matches!(writer, Writer::Graph) {
                store.enable_graph().expect("graph");
            }
            store
                .ingest(IngestBatch::new(vec![reuse_doc(59601, 1, "target")]))
                .expect("seed");
            std::thread::scope(|scope| {
                let handle = scope.spawn(|| {
                    let mut hit = reuse_doc(59601, 2, "target");
                    if conditional {
                        hit =
                            hit.with_expected_revision(ExpectedRevision::Exactly(Revision::new(1)));
                    }
                    let docs = vec![hit, reuse_doc(59602, 1, "pause")];
                    match writer {
                        Writer::Ordinary => store
                            .ingest(IngestBatch::new(docs))
                            .map(|_| ())
                            .map_err(|e| e.to_string()),
                        Writer::Namespace => namespace_batch_live(
                            root.path(),
                            vec![
                                LiveNamespaceMutation {
                                    store: &store,
                                    mutation: mutation("a", options, docs),
                                },
                                LiveNamespaceMutation {
                                    store: &other,
                                    mutation: mutation("b", OpenOptions::new(), vec![]),
                                },
                            ],
                        )
                        .map(|_| ())
                        .map_err(|e| e.to_string()),
                        #[cfg(feature = "graph-cypher")]
                        Writer::Graph => {
                            use volantis_db::property_graph::{staging::*, *};
                            let image = CanonicalContents::node(&mut [], &mut [], None, None)
                                .expect("image");
                            let writes = [StructuredWrite {
                                key: ApplicationKey::new(EntityKind::Node, "reuse", "race")
                                    .expect("key"),
                                revision: GraphRevision::new(1).expect("revision"),
                                operation: StructuredOperation::Create,
                                image: Some(WriteImage::Node(&image)),
                            }];
                            store
                                .graph_apply(
                                    GraphBatch {
                                        documents: Some(&IngestBatch::new(docs)),
                                        writes: &writes,
                                        node_documents: &[],
                                    },
                                    &running(),
                                )
                                .map(|_| ())
                                .map_err(|e| e.to_string())
                        }
                    }
                });
                barrier.wait();
                if conditional {
                    store
                        .delete(DeleteBatch::new(vec![DocId::new(59601)]))
                        .expect("delete");
                } else {
                    let token = store.purge(&[DocId::new(59601)]).expect("schedule");
                    store.await_physical_purge(token).expect("purge");
                }
                barrier.wait();
                let result = handle.join().expect("writer");
                if conditional {
                    let error = result.expect_err("stale expected revision");
                    assert!(error.contains("revision condition failed"), "{error}");
                } else {
                    result.expect("unconditional replacement");
                }
                assert_eq!(
                    reuse_row(&store, 59601).map(|row| row.revision),
                    (!conditional).then(|| Revision::new(2))
                );
                assert_eq!(reuse_row(&store, 59602).is_some(), !conditional);
                assert_eq!(calls(&model), 4);
            });
        }
    }
}

#[derive(Default)]
struct RecordingModel(std::sync::Mutex<Vec<Vec<String>>>);
impl NamespaceEmbedder for RecordingModel {
    fn epoch(&self) -> volantis_db::epoch::EmbeddingEpoch {
        Model.epoch()
    }
    fn query(&self, text: &str) -> Result<Vec<f32>, volantis_db::epoch::EmbedderError> {
        Model.query(text)
    }
    fn document(
        &self,
        text: &str,
    ) -> Result<Vec<EmbeddedChunk>, volantis_db::epoch::EmbedderError> {
        self.0.lock().expect("calls").push(vec![text.into()]);
        Model.document(text)
    }
    fn documents(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<EmbeddedChunk>>, volantis_db::epoch::EmbedderError> {
        self.0
            .lock()
            .expect("calls")
            .push(texts.iter().map(|s| s.to_string()).collect());
        let mut chunks = Model.documents(texts)?;
        match texts.last().copied() {
            Some("too few") => {
                chunks.pop();
            }
            Some("too many") => chunks.push(Model.document("extra")?),
            Some("empty") => chunks.last_mut().expect("last").clear(),
            Some("bad span") => {
                chunks
                    .last_mut()
                    .and_then(|c| c.first_mut())
                    .expect("chunk")
                    .bytes = 0..100
            }
            Some("nonfinite") => {
                chunks
                    .last_mut()
                    .and_then(|c| c.first_mut())
                    .expect("chunk")
                    .vector = vec![f32::NAN, 0.]
            }
            _ => {}
        }
        Ok(chunks)
    }
}

#[test]
fn ze597_dispatches_only_rows_that_need_inference() {
    let root = tempdir().expect("root");
    let model = Arc::new(RecordingModel::default());
    let store =
        Store::open_embedded(root.path(), OpenOptions::new(), model.clone()).expect("store");
    store
        .ingest(IngestBatch::new(vec![reuse_doc(1, 1, "target")]))
        .expect("seed");
    model.0.lock().expect("calls").clear();
    for rev in [2, 3] {
        store
            .ingest(IngestBatch::new(
                [(2, "bronze"), (1, "target"), (3, "zeppelin")]
                    .map(|(id, text)| {
                        reuse_doc(id, rev, text).with_metadata(b"FAKE-597-v2".to_vec())
                    })
                    .to_vec(),
            ))
            .expect("write");
        for id in 1..=3 {
            let row = reuse_row(&store, id).expect("live");
            assert_eq!(row.revision, Revision::new(rev));
            assert_eq!(row.metadata.as_deref(), Some(b"FAKE-597-v2".as_slice()));
            assert_eq!(
                row.vector,
                Some(if id == 1 { vec![1., 0.] } else { vec![0., 1.] })
            );
        }
    }
    assert!(matches!(
        store.ingest(IngestBatch::new(vec![])),
        Err(IngestError::EmptyBatch)
    ));
    assert_eq!(
        *model.0.lock().expect("calls"),
        vec![vec!["bronze", "zeppelin"]]
    );
}

#[test]
fn ze597_batch_failure_publishes_nothing() {
    let root = tempdir().expect("root");
    let options = OpenOptions::new()
        .with_embedder(Arc::new(RecordingModel::default()))
        .expect("model");
    let open = |name: &str| Store::open(root.path().join(name), options.clone()).expect("open");
    let a = open("a");
    let b = open("b");
    for store in [&a, &b] {
        store
            .ingest(IngestBatch::new(vec![reuse_doc(1, 1, "target|old")]))
            .expect("seed");
    }
    let original = reuse_row(&a, 1).expect("old revision");
    let generation = a.snapshot().expect("snapshot").generation();
    let check = |store: &Store| {
        assert_eq!(store.snapshot().expect("snapshot").generation(), generation);
        assert_eq!(reuse_row(store, 1), Some(original.clone()));
        assert!(reuse_row(store, 2).is_none());
        let chunks = store
            .query_embedded("target", EmbeddedLeg::Dense, 2, None, running())
            .expect("chunks");
        assert_eq!(
            chunks
                .iter()
                .map(|hit| (hit.chunk, hit.bytes.clone(), hit.score))
                .collect::<Vec<_>>(),
            [(0, 0..6, 0.), (1, 7..10, -2.)]
        );
    };
    let rows = |last| vec![reuse_doc(1, 2, "replacement"), reuse_doc(2, 1, last)];
    for bad in [
        "!fail",
        "too few",
        "too many",
        "empty",
        "bad span",
        "nonfinite",
    ] {
        assert!(matches!(
            a.ingest(IngestBatch::new(rows(bad))),
            Err(IngestError::Store(StoreError::Embedding(_)))
        ));
        check(&a);
    }
    assert!(matches!(
        namespace_batch_live(
            root.path(),
            vec![
                LiveNamespaceMutation {
                    store: &a,
                    mutation: mutation("a", options.clone(), rows("good"))
                },
                LiveNamespaceMutation {
                    store: &b,
                    mutation: mutation("b", options.clone(), rows("!fail"))
                },
            ]
        ),
        Err(StoreError::Embedding(_))
    ));
    check(&a);
    check(&b);
    drop((a, b));
    check(&open("a"));
    check(&open("b"));
}

// WF-985: commit text now, embed it later in the background.
const WF985_META: &[u8] = b"FAKE600";
const WF985_MAUVE: &str = "fictitious mauve sprocket";
const WF985_CYAN: &str = "fictitious cyan thimble";
const WF985_WAIT: std::time::Duration = std::time::Duration::from_secs(30);

fn wf985_store(
    path: &std::path::Path,
    model: Arc<namespace_model::GateModel>,
    schedule: DocumentSchedule,
) -> Arc<Store> {
    let options = OpenOptions::new()
        .with_embedder(model)
        .expect("model")
        .with_document_schedule(schedule);
    Arc::new(Store::open(path, options).expect("open"))
}

fn wf985_doc(id: u128, rev: u64, text: &str) -> IngestDocument {
    IngestDocument::new(
        DocumentVersion::new(DocId::new(id), Revision::new(rev)),
        vec![],
    )
    .with_text(text)
    .with_timestamp(rev as i64)
    .with_metadata(WF985_META.to_vec())
}

fn wf985_report(store: &Store, text: &str, leg: EmbeddedLeg, k: usize) -> EmbeddedReport {
    store
        .query_embedded_report(text, leg, k, None, running())
        .expect("embedded report")
}

fn wf985_hits(report: &EmbeddedReport) -> Vec<(u128, EmbeddedRowState)> {
    report
        .hits
        .iter()
        .map(|hit| (hit.version.doc_id().get(), hit.state))
        .collect()
}

/// Raw dense search over stored vectors, exact and graph-free.
fn wf985_dense_ids(store: &Store, k: usize) -> Vec<u128> {
    let mut ids = store
        .search(
            SearchRequest::new(&[1.0, 0.0]),
            k,
            volantis_db::scan::ScanOptions { thread_budget: 1 },
            running(),
        )
        .expect("dense search")
        .candidates
        .iter()
        .filter_map(|candidate| candidate.document())
        .map(|version| version.doc_id().get())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}

fn wf985_vector(store: &Store, id: u128) -> Option<Vec<f32>> {
    reuse_row(store, id).expect("live row").vector
}

#[test]
fn wf985_ack_precedes_inference() {
    let root = tempdir().expect("root");
    let model = Arc::new(namespace_model::GateModel::default());
    model.close_gate();
    let store = wf985_store(
        &root.path().join("bg"),
        model.clone(),
        DocumentSchedule::Background,
    );
    store.attach_embedding_worker().expect("attach");
    store
        .ingest(IngestBatch::new(vec![wf985_doc(60001, 7, WF985_MAUVE)]))
        .expect("first write commits before inference");
    assert!(
        model.wait_entered(1, WF985_WAIT),
        "worker never called the model"
    );
    // The model is held at the gate: writes, reads and lexical search go on.
    let ack = store
        .ingest(IngestBatch::new(vec![wf985_doc(60002, 1, WF985_CYAN)]))
        .expect("second write while the model is blocked");
    let row = reuse_row(&store, 60001).expect("pending row");
    assert_eq!(row.revision, Revision::new(7));
    assert_eq!(row.text.as_deref(), Some(WF985_MAUVE));
    assert_eq!(row.metadata.as_deref(), Some(WF985_META));
    assert_eq!(row.vector, None);
    let lexical = wf985_report(&store, "mauve", EmbeddedLeg::Lexical, 4);
    assert_eq!(lexical.generation, ack.generation());
    assert_eq!(lexical.pending_rows, 2);
    assert_eq!(
        wf985_hits(&lexical),
        vec![(60001, EmbeddedRowState::Pending)]
    );
    let hit = lexical.hits.first().expect("pending hit");
    assert_eq!((hit.chunk, hit.bytes.clone()), (0, 0..WF985_MAUVE.len()));
    let dense = wf985_report(&store, "mauve", EmbeddedLeg::Dense, 4);
    assert!(dense.hits.is_empty());
    assert_eq!(dense.pending_rows, 2);
    let status = store.embedding_status().expect("status");
    assert!(status.in_flight && status.running && !status.paused);
    assert_eq!(status.pending, 2);
    model.open_gate();
    let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
    assert_eq!(
        (status.pending, status.last_error),
        (0, EmbeddingErrorCode::None)
    );
    assert_eq!(wf985_vector(&store, 60001), Some(vec![1., 0.]));
    assert_eq!(wf985_vector(&store, 60002), Some(vec![0., 1.]));
    let dense = wf985_report(&store, "mauve", EmbeddedLeg::Dense, 4);
    assert_eq!(
        wf985_hits(&dense),
        vec![
            (60001, EmbeddedRowState::Ready),
            (60002, EmbeddedRowState::Ready)
        ]
    );
    assert_eq!(dense.pending_rows, 0);
    assert_eq!(
        reuse_row(&store, 60001).expect("ready").revision,
        Revision::new(7)
    );
    store.close().expect("close");

    // A paused open makes no document call.
    let paused = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(
        &root.path().join("paused"),
        paused.clone(),
        DocumentSchedule::Paused,
    );
    store.attach_embedding_worker().expect("attach");
    store
        .ingest(IngestBatch::new(
            (1..=3).map(|id| wf985_doc(id, 1, WF985_MAUVE)).collect(),
        ))
        .expect("paused writes");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("status");
    assert!(status.paused && !status.in_flight);
    assert_eq!(status.pending, 3);
    assert_eq!(paused.call_count(), 0);
}

#[test]
fn wf985_reuse_skips_pending() {
    let root = tempdir().expect("root");
    let model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(root.path(), model.clone(), DocumentSchedule::Immediate);
    store
        .ingest(IngestBatch::new(vec![wf985_doc(60001, 7, WF985_MAUVE)]))
        .expect("ready seed");
    assert_eq!(model.call_count(), 1);
    store.close().expect("close");
    drop(store);
    let model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(root.path(), model.clone(), DocumentSchedule::Paused);
    // Same text at a higher revision reuses the ready chunks at once.
    store
        .ingest(IngestBatch::new(vec![wf985_doc(60001, 8, WF985_MAUVE)]))
        .expect("same text");
    assert_eq!(model.call_count(), 0);
    assert_eq!(wf985_vector(&store, 60001), Some(vec![1., 0.]));
    assert_eq!(
        wf985_report(&store, "x", EmbeddedLeg::Lexical, 1).pending_rows,
        0
    );
    // Changed text becomes pending, and so does a new id.
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(60001, 9, WF985_CYAN),
            wf985_doc(60002, 1, WF985_MAUVE),
        ]))
        .expect("changed text");
    assert_eq!(model.call_count(), 0);
    assert_eq!(wf985_vector(&store, 60001), None);
    assert_eq!(wf985_vector(&store, 60002), None);
    assert_eq!(
        wf985_report(&store, "x", EmbeddedLeg::Lexical, 1).pending_rows,
        2
    );
    // A metadata-only edit of a pending row keeps one pending revision.
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(60001, 10, WF985_CYAN).with_metadata(b"FAKE600-v2".to_vec()),
        ]))
        .expect("metadata edit");
    let report = wf985_report(&store, "cyan", EmbeddedLeg::Lexical, 4);
    assert_eq!(report.pending_rows, 2);
    assert_eq!(
        wf985_hits(&report),
        vec![(60001, EmbeddedRowState::Pending)]
    );
    assert_eq!(
        reuse_row(&store, 60001).expect("row").metadata.as_deref(),
        Some(b"FAKE600-v2".as_slice())
    );
}

#[test]
fn wf985_worker_batches_rows() {
    let root = tempdir().expect("root");
    let texts = (0..10)
        .map(|n| format!("fictitious row {n}"))
        .collect::<Vec<_>>();
    let docs = || {
        texts
            .iter()
            .enumerate()
            .map(|(n, text)| wf985_doc(n as u128 + 1, 1, text))
            .collect::<Vec<_>>()
    };
    let sync = wf985_store(
        &root.path().join("sync"),
        Arc::new(namespace_model::GateModel::default()),
        DocumentSchedule::Immediate,
    );
    sync.ingest(IngestBatch::new(docs())).expect("sync");
    let model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(
        &root.path().join("bg"),
        model.clone(),
        DocumentSchedule::Paused,
    );
    store.ingest(IngestBatch::new(docs())).expect("pending");
    store.attach_embedding_worker().expect("attach");
    store.resume_embedding().expect("resume");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
    assert_eq!(
        (status.pending, status.last_error),
        (0, EmbeddingErrorCode::None)
    );
    let calls = model.calls.lock().expect("calls").clone();
    assert!(calls.len() < texts.len(), "{calls:?}");
    assert_eq!(calls.concat(), texts);
    for id in 1..=10 {
        let ready = reuse_row(&store, id).expect("ready");
        let expected = reuse_row(&sync, id).expect("sync row");
        assert_eq!(ready, expected);
    }
    let fields = DocumentFields::TEXT | DocumentFields::METADATA | DocumentFields::ATTRIBUTES;
    assert_eq!(
        store.get_documents(&[DocId::new(1)], fields).expect("get"),
        sync.get_documents(&[DocId::new(1)], fields).expect("get")
    );
}

#[test]
fn wf985_completion_checks_admission_identity() {
    let root = tempdir().expect("root");
    let model = Arc::new(namespace_model::GateModel::default());
    model.close_gate();
    let store = wf985_store(root.path(), model.clone(), DocumentSchedule::Background);
    store
        .ingest(IngestBatch::new(
            (1..=4).map(|id| wf985_doc(id, 7, WF985_MAUVE)).collect(),
        ))
        .expect("pending rows");
    store.attach_embedding_worker().expect("attach");
    assert!(model.wait_entered(1, WF985_WAIT));
    // Stop the next admission so only the blocked batch completes.
    store.pause_embedding().expect("pause");
    store
        .ingest(IngestBatch::new(vec![wf985_doc(1, 8, WF985_CYAN)]))
        .expect("edit");
    store
        .delete(DeleteBatch::new(vec![DocId::new(2)]))
        .expect("delete");
    store
        .delete(DeleteBatch::new(vec![DocId::new(3)]))
        .expect("delete before purge");
    let token = store.purge(&[DocId::new(3)]).expect("purge");
    store.await_physical_purge(token).expect("physical purge");
    store
        .ingest(IngestBatch::new(vec![wf985_doc(3, 7, WF985_MAUVE)]))
        .expect("reinsert with the purged revision");
    let before = store.snapshot().expect("snapshot").generation();
    model.open_gate();
    let status = store.wait_embedding_idle(WF985_WAIT).expect("settle");
    assert!(status.paused && !status.in_flight);
    assert_eq!(
        store.snapshot().expect("snapshot").generation(),
        before + 1,
        "one generation for the accepted completion"
    );
    assert_eq!(wf985_vector(&store, 4), Some(vec![1., 0.]));
    let edited = reuse_row(&store, 1).expect("edited");
    assert_eq!(
        (edited.revision, edited.text.as_deref(), edited.vector),
        (Revision::new(8), Some(WF985_CYAN), None)
    );
    assert!(reuse_row(&store, 2).is_none());
    let reinserted = reuse_row(&store, 3).expect("reinserted");
    assert_eq!(
        (reinserted.revision, reinserted.vector),
        (Revision::new(7), None)
    );
    assert_eq!(status.pending, 2);
    // Equal-revision replay leaves the ready row alone; the next revision wins.
    let generation = store.snapshot().expect("snapshot").generation();
    store
        .ingest(IngestBatch::new(vec![wf985_doc(4, 7, WF985_CYAN)]))
        .expect("replay");
    assert_eq!(store.snapshot().expect("snapshot").generation(), generation);
    assert_eq!(wf985_vector(&store, 4), Some(vec![1., 0.]));
    store
        .ingest(IngestBatch::new(vec![wf985_doc(4, 8, WF985_CYAN)]))
        .expect("next revision");
    store.resume_embedding().expect("resume");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
    assert_eq!(status.pending, 0);
    assert_eq!(wf985_vector(&store, 1), Some(vec![0., 1.]));
    assert_eq!(wf985_vector(&store, 3), Some(vec![1., 0.]));
    assert_eq!(wf985_vector(&store, 4), Some(vec![0., 1.]));
}

#[test]
fn wf985_pause_retry_and_close() {
    let root = tempdir().expect("root");
    let model = Arc::new(namespace_model::GateModel::default());
    model.fail.store(true, std::sync::atomic::Ordering::SeqCst);
    let store = wf985_store(root.path(), model.clone(), DocumentSchedule::Background);
    store.attach_embedding_worker().expect("attach");
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(1, 1, WF985_MAUVE),
            wf985_doc(2, 1, WF985_CYAN),
        ]))
        .expect("pending rows");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("failure");
    assert_eq!(
        (status.pending, status.in_flight, status.last_error),
        (2, false, EmbeddingErrorCode::Model)
    );
    // A parked failure waits for resume; repeated pause and resume are idempotent.
    store.pause_embedding().expect("pause");
    store.pause_embedding().expect("pause again");
    assert!(store.embedding_status().expect("status").paused);
    model.fail.store(false, std::sync::atomic::Ordering::SeqCst);
    store.resume_embedding().expect("resume");
    store.resume_embedding().expect("resume again");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("retry");
    assert_eq!(
        (status.pending, status.last_error),
        (0, EmbeddingErrorCode::None)
    );

    // Close from another thread while the model is blocked.
    model.close_gate();
    store
        .ingest(IngestBatch::new(vec![wf985_doc(3, 1, WF985_MAUVE)]))
        .expect("pending row");
    let entered = model.calls.lock().expect("calls").len();
    assert!(model.wait_entered(entered, WF985_WAIT));
    let closer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || store.close())
    };
    model.open_gate();
    closer.join().expect("close thread").expect("close");
    drop(store);
    assert_eq!(
        Arc::strong_count(&model),
        1,
        "the runtime lease outlived close"
    );
    let reopened = wf985_store(root.path(), model.clone(), DocumentSchedule::Paused);
    let status = reopened.embedding_status().expect("status");
    assert!(!status.running);
    assert!(reuse_row(&reopened, 3).is_some());
    reopened.attach_embedding_worker().expect("attach");
    reopened.resume_embedding().expect("resume");
    assert_eq!(
        reopened
            .wait_embedding_idle(WF985_WAIT)
            .expect("drain")
            .pending,
        0
    );
}

#[test]
fn wf985_sparse_rows_round_trip() {
    let root = tempdir().expect("root");
    let path = root.path().join("store");
    let ready_model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(&path, ready_model, DocumentSchedule::Immediate);
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(1, 1, WF985_MAUVE),
            wf985_doc(2, 1, "fictitious ready two"),
        ]))
        .expect("ready rows");
    store.seal().expect("legacy all-ready segment");
    store.close().expect("close");
    drop(store);

    let model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(&path, model.clone(), DocumentSchedule::Paused);
    let check = |store: &Store, stage: &str| {
        let expected: &[u128] =
            if stage.starts_with("active pending") || stage.starts_with("all-pending") {
                &[3, 4]
            } else {
                &[3, 4, 5]
            };
        assert_eq!(wf985_dense_ids(store, 10), vec![1, 2], "{stage}: dense");
        let lexical = wf985_report(store, "fictitious", EmbeddedLeg::Lexical, 10);
        assert_eq!(
            lexical.pending_rows,
            expected.len() as u64,
            "{stage}: pending count"
        );
        let mut pending = wf985_hits(&lexical)
            .into_iter()
            .filter(|(_, state)| *state == EmbeddedRowState::Pending)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        pending.sort_unstable();
        assert_eq!(pending, expected, "{stage}: lexical pending");
        let dense = wf985_report(store, "x", EmbeddedLeg::Dense, 10);
        assert!(
            dense
                .hits
                .iter()
                .all(|hit| hit.state == EmbeddedRowState::Ready && hit.version.doc_id().get() <= 2),
            "{stage}: dense chunks"
        );
        let hybrid = wf985_report(store, "pending", EmbeddedLeg::Hybrid, 10);
        assert!(
            hybrid.hits.iter().all(|hit| hit.score > 0.),
            "{stage}: hybrid keeps scored candidates only"
        );
        assert_eq!(wf985_vector(store, 3), None, "{stage}");
        assert_eq!(wf985_vector(store, 1), Some(vec![1., 0.]), "{stage}");
        assert_eq!(
            store
                .nearest_chunks(DocId::new(1), None, &NeighbourOptions::default(), running())
                .expect("neighbours")
                .iter()
                .map(|hit| hit.id.get())
                .collect::<Vec<_>>(),
            vec![2],
            "{stage}: neighbours skip pending rows"
        );
        assert!(
            matches!(
                store.nearest_chunks(DocId::new(3), None, &NeighbourOptions::default(), running()),
                Err(QueryError::Store(
                    StoreError::NeighbourSourceNotFound { .. }
                ))
            ),
            "{stage}: a pending source has no chunk"
        );
    };
    // All-pending segment, then a mixed one.
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(3, 1, "fictitious pending three"),
            wf985_doc(4, 1, "fictitious pending four"),
        ]))
        .expect("pending rows");
    check(&store, "active pending");
    store.seal().expect("all-pending segment");
    check(&store, "all-pending sealed");
    store
        .ingest(IngestBatch::new(vec![
            wf985_doc(1, 2, WF985_MAUVE),
            wf985_doc(5, 1, "fictitious pending five"),
        ]))
        .expect("mixed rows");
    assert_eq!(model.call_count(), 0);
    check(&store, "mixed active");
    store.seal().expect("mixed segment");
    check(&store, "mixed sealed");
    assert_eq!(store.snapshot().expect("snapshot").segments().len(), 3);
    let report = volantis_db::verify::verify_store(&path).expect("verify");
    assert!(report.is_clean(), "{report:?}");
    store.merge_sealed().expect("merge");
    assert_eq!(store.snapshot().expect("snapshot").segments().len(), 1);
    check(&store, "merged");
    let backup = root.path().join("backup");
    store.write_snapshot(&backup).expect("backup");
    store.close().expect("close");
    drop(store);
    let report = volantis_db::verify::verify_store(&path).expect("verify");
    assert!(report.is_clean(), "{report:?}");

    let reader = Store::open(
        &path,
        OpenOptions::read_only()
            .with_embedder(model.clone())
            .expect("model"),
    )
    .expect("read-only");
    check(&reader, "read-only");
    assert!(matches!(
        reader.pause_embedding(),
        Err(StoreError::ReadOnly)
    ));
    assert!(!reader.embedding_status().expect("status").running);
    drop(reader);
    let restored = wf985_store(&backup, model.clone(), DocumentSchedule::Paused);
    check(&restored, "restored backup");
    drop(restored);
    let store = wf985_store(&path, model.clone(), DocumentSchedule::Background);
    check(&store, "writable reopen");
    store.attach_embedding_worker().expect("attach");
    assert_eq!(
        store
            .wait_embedding_idle(WF985_WAIT)
            .expect("drain")
            .pending,
        0
    );
    assert_eq!(wf985_dense_ids(&store, 10), vec![1, 2, 3, 4, 5]);
    store.close().expect("close");
    drop(store);
    // Completions of sealed pending rows now sit in the WAL: replay applies
    // them once and tombstones the sealed pending rows.
    let report = volantis_db::verify::verify_store(&path).expect("verify");
    assert!(report.is_clean(), "{report:?}");
    let store = wf985_store(&path, model.clone(), DocumentSchedule::Paused);
    assert_eq!(wf985_dense_ids(&store, 10), vec![1, 2, 3, 4, 5]);
    assert_eq!(
        wf985_report(&store, "fictitious", EmbeddedLeg::Lexical, 10).pending_rows,
        0
    );
    store.seal().expect("seal completions");
    store.merge_sealed().expect("merge");
    assert_eq!(wf985_dense_ids(&store, 10), vec![1, 2, 3, 4, 5]);
    store.close().expect("close");
    let report = volantis_db::verify::verify_store(&path).expect("verify");
    assert!(report.is_clean(), "{report:?}");
}

#[test]
fn wf985_namespace_batch_atomic_visibility() {
    let root = tempdir().expect("root");
    let model = Arc::new(namespace_model::GateModel::default());
    let options = |schedule| {
        OpenOptions::new()
            .with_embedder(model.clone())
            .expect("model")
            .with_document_schedule(schedule)
    };
    let open = |name: &str, schedule| {
        Arc::new(Store::open(root.path().join(name), options(schedule)).expect("open"))
    };
    let a = open("a", DocumentSchedule::Paused);
    let b = open("b", DocumentSchedule::Paused);
    b.ingest(IngestBatch::new(vec![wf985_doc(9, 5, WF985_CYAN)]))
        .expect("seed");
    let live = |a_docs, b_docs| {
        namespace_batch_live(
            root.path(),
            vec![
                LiveNamespaceMutation {
                    store: &a,
                    mutation: mutation("a", options(DocumentSchedule::Paused), a_docs),
                },
                LiveNamespaceMutation {
                    store: &b,
                    mutation: mutation("b", options(DocumentSchedule::Paused), b_docs),
                },
            ],
        )
    };
    // A failed participant publishes no pending row anywhere.
    assert!(
        live(
            vec![wf985_doc(60001, 7, WF985_MAUVE)],
            vec![wf985_doc(9, 3, WF985_CYAN)]
        )
        .is_err()
    );
    assert!(reuse_row(&a, 60001).is_none());
    assert_eq!(a.embedding_status().expect("status").pending, 0);
    live(
        vec![wf985_doc(60001, 7, WF985_MAUVE)],
        vec![wf985_doc(60002, 1, WF985_CYAN)],
    )
    .expect("commit");
    for (store, id) in [(&a, 60001), (&b, 60002)] {
        let row = reuse_row(store, id).expect("visible at once");
        assert_eq!(row.vector, None);
        let report = wf985_report(store, "fictitious", EmbeddedLeg::Lexical, 4);
        assert!(wf985_hits(&report).contains(&(id, EmbeddedRowState::Pending)));
    }
    drop((a, b));
    // A portable batch has no worker; it leaves pending work for the next
    // writable handle.
    namespace_batch(
        root.path(),
        vec![
            mutation(
                "a",
                options(DocumentSchedule::Paused),
                vec![wf985_doc(60003, 1, WF985_CYAN)],
            ),
            mutation(
                "b",
                options(DocumentSchedule::Paused),
                vec![wf985_doc(60004, 1, WF985_MAUVE)],
            ),
        ],
    )
    .expect("path batch");
    assert_eq!(model.call_count(), 0);
    for (name, ids) in [("a", vec![60001, 60003]), ("b", vec![9, 60002, 60004])] {
        let store = open(name, DocumentSchedule::Background);
        store.attach_embedding_worker().expect("attach");
        let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
        assert_eq!(status.pending, 0, "{name}");
        for id in ids {
            assert!(wf985_vector(&store, id).is_some(), "{name} {id}");
        }
    }
}

/// A graph-linked document embeds at write time on a deferred schedule:
/// completion never rewrites graph-linked rows. A plain document of a
/// graph batch is pending and the worker completes it.
#[cfg(feature = "graph-cypher")]
#[test]
fn wf985_graph_linked_documents_embed_at_write_time() {
    use volantis_db::property_graph::{staging::*, *};
    let root = tempdir().expect("namespace");
    let model = Arc::new(namespace_model::GateModel::default());
    let store = wf985_store(root.path(), model.clone(), DocumentSchedule::Paused);
    store.enable_graph().expect("graph");
    let image = CanonicalContents::node(&mut [], &mut [], None, None).expect("image");
    let writes = [StructuredWrite {
        key: ApplicationKey::new(EntityKind::Node, "wf985", "linked").expect("key"),
        revision: GraphRevision::new(1).expect("revision"),
        operation: StructuredOperation::Create,
        image: Some(WriteImage::Node(&image)),
    }];
    let docs = [GraphNodeDocument {
        item_index: 0,
        document: wf985_doc(60001, 1, WF985_MAUVE),
    }];
    store
        .graph_apply(GraphBatch::with_node_documents(&writes, &docs), &running())
        .expect("linked write");
    assert_eq!(model.call_count(), 1);
    assert_eq!(wf985_vector(&store, 60001), Some(vec![1., 0.]));
    // A later plain upsert of the graph-linked id also embeds at once.
    store
        .ingest(IngestBatch::new(vec![wf985_doc(60001, 2, WF985_CYAN)]))
        .expect("linked upsert");
    assert_eq!(wf985_vector(&store, 60001), Some(vec![0., 1.]));
    let writes = [StructuredWrite {
        key: ApplicationKey::new(EntityKind::Node, "wf985", "other").expect("key"),
        revision: GraphRevision::new(1).expect("revision"),
        operation: StructuredOperation::Create,
        image: Some(WriteImage::Node(&image)),
    }];
    let documents = IngestBatch::new(vec![wf985_doc(60002, 1, WF985_CYAN)]);
    store
        .graph_apply(
            GraphBatch {
                writes: &writes,
                documents: Some(&documents),
                node_documents: &[],
            },
            &running(),
        )
        .expect("mixed write");
    assert_eq!(wf985_vector(&store, 60002), None);
    store.attach_embedding_worker().expect("attach");
    store.resume_embedding().expect("resume");
    let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
    assert_eq!(
        (status.pending, status.last_error),
        (0, EmbeddingErrorCode::None)
    );
    assert_eq!(wf985_vector(&store, 60002), Some(vec![0., 1.]));
}

/// Synthetic model with a fixed 200 ms delay per document-tower batch.
struct DelayModel(namespace_model::GateModel);
impl NamespaceEmbedder for DelayModel {
    fn epoch(&self) -> volantis_db::epoch::EmbeddingEpoch {
        self.0.epoch()
    }
    fn document(
        &self,
        text: &str,
    ) -> Result<Vec<EmbeddedChunk>, volantis_db::epoch::EmbedderError> {
        self.documents(&[text])?
            .pop()
            .ok_or_else(|| volantis_db::epoch::EmbedderFailure::new("missing").into())
    }
    fn documents(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<EmbeddedChunk>>, volantis_db::epoch::EmbedderError> {
        std::thread::sleep(std::time::Duration::from_millis(200));
        self.0.documents(texts)
    }
    fn query(&self, text: &str) -> Result<Vec<f32>, volantis_db::epoch::EmbedderError> {
        self.0.query(text)
    }
}

/// One measurement: acknowledgement time of 32 rows, immediate versus paused.
#[test]
#[ignore = "measurement; run once with --ignored --nocapture"]
fn wf985_ack_comparison_once() {
    let root = tempdir().expect("root");
    let docs = || {
        (0..32)
            .map(|n| wf985_doc(n + 1, 1, &format!("fictitious row {n}")))
            .collect::<Vec<_>>()
    };
    for (name, schedule) in [
        ("immediate", DocumentSchedule::Immediate),
        ("paused", DocumentSchedule::Paused),
    ] {
        let model = Arc::new(DelayModel(namespace_model::GateModel::default()));
        let options = OpenOptions::new()
            .with_embedder(model.clone())
            .expect("model")
            .with_document_schedule(schedule);
        let store = Arc::new(Store::open(root.path().join(name), options).expect("open"));
        let start = std::time::Instant::now();
        store.ingest(IngestBatch::new(docs())).expect("ingest");
        let ack = start.elapsed();
        let calls_at_ack = model.0.call_count();
        let mut drain = std::time::Duration::ZERO;
        if schedule == DocumentSchedule::Paused {
            store.attach_embedding_worker().expect("attach");
            let start = std::time::Instant::now();
            store.resume_embedding().expect("resume");
            let status = store.wait_embedding_idle(WF985_WAIT).expect("drain");
            drain = start.elapsed();
            assert_eq!(status.pending, 0);
        }
        let ready = (1..=32)
            .filter(|id| wf985_vector(&store, *id).is_some())
            .count();
        println!(
            "wf985 {name}: ack_ms={:.1} document_calls_at_ack={calls_at_ack} drain_ms={:.1} batches={} ready_rows={ready}",
            ack.as_secs_f64() * 1e3,
            drain.as_secs_f64() * 1e3,
            model.0.call_count(),
        );
    }
}
