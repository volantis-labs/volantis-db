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
    use volantis_db::property_graph::staging::{
        StructuredOperation, StructuredWrite, WriteImage,
    };
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
