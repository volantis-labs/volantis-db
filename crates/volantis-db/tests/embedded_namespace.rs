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
