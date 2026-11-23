#![allow(clippy::expect_used)]

use tempfile::tempdir;
use volantis_db_text::{
    IngestControl, IngestOptions, Legs, QueryOptions, TextDocument, TextError, TextFaultSite,
    TextStore,
};

mod common;

#[test]
fn ingest_text_over_a_corpus_produces_the_same_rows_and_vectors_as_the_serialized_path() {
    let directory = tempdir().expect("tempdir");
    let bundle_path = directory.path().join("fixture.zem");
    common::write_symmetric_fixture_bundle(&bundle_path);
    let pipeline = TextStore::open(
        directory.path().join("pipeline"),
        &bundle_path,
        Default::default(),
    )
    .expect("open pipeline store");
    let serial = TextStore::open(
        directory.path().join("serial"),
        &bundle_path,
        Default::default(),
    )
    .expect("open serial store");
    let documents = (1..=9)
        .map(|id| TextDocument::new(id, 1, format!("the bronze zeppelin {id}")))
        .collect::<Vec<_>>();

    let pipeline_report = pipeline
        .ingest_text(
            &documents,
            IngestOptions {
                embed_batch_size: 3,
                seal_every: 4,
                ..Default::default()
            },
        )
        .expect("pipeline ingest");
    let serial_report = serial
        .ingest_text_serialized(&documents, IngestOptions::default())
        .expect("serialized ingest");
    assert_eq!(pipeline_report.chunks, serial_report.chunks);

    let query = QueryOptions::new(9).with_legs(Legs::Dense);
    let pipeline_hits = pipeline
        .query_text("bronze zeppelin", query)
        .expect("pipeline query");
    let serial_hits = serial
        .query_text("bronze zeppelin", query)
        .expect("serial query");
    let pipeline_rows = pipeline_hits
        .iter()
        .map(|hit| (hit.doc_id, hit.revision, hit.chunk, hit.vector_squared_l2))
        .collect::<Vec<_>>();
    let serial_rows = serial_hits
        .iter()
        .map(|hit| (hit.doc_id, hit.revision, hit.chunk, hit.vector_squared_l2))
        .collect::<Vec<_>>();
    assert_eq!(pipeline_rows, serial_rows);
}

#[test]
fn a_failing_embed_worker_surfaces_as_a_typed_error_and_every_thread_is_joined() {
    let directory = tempdir().expect("tempdir");
    let bundle_path = directory.path().join("fixture.zem");
    common::write_symmetric_fixture_bundle(&bundle_path);
    let store = TextStore::open(
        directory.path().join("store"),
        &bundle_path,
        Default::default(),
    )
    .expect("open store");
    let control = IngestControl::new().with_fault(TextFaultSite::EmbedWorkerPanic);

    let error = store
        .ingest_text_controlled(
            &[TextDocument::new(1, 1, "bronze zeppelin")],
            IngestOptions::default(),
            control.clone(),
        )
        .expect_err("injected embed failure must surface");

    assert!(matches!(
        error,
        TextError::Pipeline {
            stage: "embed worker",
            ..
        }
    ));
    assert!(control.fault_fired(TextFaultSite::EmbedWorkerPanic));
    store
        .close()
        .expect("all workers join after the caught panic");
}

#[test]
fn back_pressure_bounds_in_flight_batches_to_the_channel_capacity() {
    let directory = tempdir().expect("tempdir");
    let bundle_path = directory.path().join("fixture.zem");
    common::write_symmetric_fixture_bundle(&bundle_path);
    let store = TextStore::open(
        directory.path().join("store"),
        &bundle_path,
        Default::default(),
    )
    .expect("open store");
    let documents = (1..=64)
        .map(|id| TextDocument::new(id, 1, "the bronze zeppelin"))
        .collect::<Vec<_>>();
    let capacity = 1;

    let report = store
        .ingest_text(
            &documents,
            IngestOptions {
                embed_batch_size: 1,
                seal_every: 64,
                channel_capacity: capacity,
                ..Default::default()
            },
        )
        .expect("bounded ingest");

    assert!(report.max_in_flight_batches <= capacity);
}

#[test]
fn namespace_model_chunks_and_queries_the_source_document() {
    use volantis_db::ingest::{DocId, DocumentVersion, IngestBatch, IngestDocument, Revision};
    use volantis_db::lifecycle::{
        CancelToken, OpenOptions, QueryControl, Store, embedded::EmbeddedLeg,
    };
    let dir = tempdir().expect("directory");
    let path = dir.path().join("fixture.zem");
    common::write_fixture_bundle(
        &path,
        &[
            common::FixtureTower::document("document"),
            common::FixtureTower::query("query"),
        ],
        b"pair",
    );
    let model = volantis_db_text::NamespaceModel::open_bundle(&path).expect("fixture model");
    let store = Store::open_embedded(
        dir.path().join("notes"),
        OpenOptions::new(),
        std::sync::Arc::new(model),
    )
    .expect("namespace");
    let text = "bronze zeppelin ".repeat(32);
    let id = DocId::new(u128::MAX);
    store
        .ingest(IngestBatch::new(vec![
            IngestDocument::new(DocumentVersion::new(id, Revision::new(6)), vec![])
                .with_text(&text),
        ]))
        .expect("ingest");
    let hits = store
        .query_embedded(
            "bronze",
            EmbeddedLeg::Hybrid,
            2,
            None,
            QueryControl::Cancel(CancelToken::new()),
        )
        .expect("query");
    assert_eq!(hits.len(), 2);
    for hit in hits {
        assert_eq!(hit.version.doc_id(), id);
        assert_eq!(text.get(hit.bytes), Some(hit.text.as_str()));
    }
}

#[path = "../../volantis-db/tests/test_support/namespace_model.rs"]
mod namespace_fixture;

#[test]
fn namespace_session_closed_store_releases_owner() {
    use std::sync::Arc;
    use volantis_db::lifecycle::{OpenOptions, Store};
    let dir = tempdir().expect("directory");
    let model = Arc::new(namespace_fixture::CountingModel::default());
    let weak = Arc::downgrade(&model);
    let writer =
        Store::open_embedded(dir.path().join("notes"), OpenOptions::new(), model).expect("writer");
    let view = writer.open_snapshot().expect("view");
    writer.close().expect("close writer");
    view.embed_query("fixture copper kite")
        .expect("query after writer close");
    assert_eq!(
        weak.upgrade()
            .expect("snapshot owner")
            .queries
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    view.close().expect("close snapshot");
    assert!(
        weak.upgrade().is_none(),
        "closed handles retained their embedder"
    );
    writer.close().expect("repeat writer close");
    view.close().expect("repeat snapshot close");
}

#[test]
fn namespace_session_admitted_embedding_keeps_its_owner_through_close() {
    use std::sync::{Arc, Barrier};
    use volantis_db::ingest::{
        DocId, DocumentVersion, IngestBatch, IngestDocument, IngestError, Revision,
    };
    use volantis_db::lifecycle::{OpenOptions, Store, StoreError};
    let root = tempdir().expect("root");
    let barrier = Arc::new(Barrier::new(2));
    let model = Arc::new(namespace_fixture::CountingModel {
        pause: Some(barrier.clone()),
        ..Default::default()
    });
    let weak = Arc::downgrade(&model);
    let writer = Store::open_embedded(root.path(), OpenOptions::new(), model).expect("writer");
    let doc = IngestDocument::new(
        DocumentVersion::new(DocId::new(59801), Revision::new(1)),
        vec![],
    )
    .with_text("pause");
    std::thread::scope(|scope| {
        let pending = scope.spawn(|| writer.ingest(IngestBatch::new(vec![doc])));
        barrier.wait();
        writer.close().expect("close while embedding");
        let still_owned = weak.strong_count();
        barrier.wait();
        assert_eq!(still_owned, 1);
        assert!(matches!(
            pending.join().expect("ingest thread"),
            Err(IngestError::Store(StoreError::Closed))
        ));
    });
    assert!(weak.upgrade().is_none());
}

#[test]
fn namespace_session_preserves_live_text_store() {
    let dir = tempdir().expect("directory");
    let path = dir.path().join("fixture.zem");
    common::write_symmetric_fixture_bundle(&path);
    let text = TextStore::open(dir.path().join("text"), &path, Default::default()).expect("text");
    text.ingest_text(&[TextDocument::new(1, 1, "bronze")], Default::default())
        .expect("write");
    let namespace = volantis_db_text::NamespaceModel::open_bundle(&path).expect("namespace");
    let query = || {
        text.query_text("bronze", QueryOptions::new(1).with_legs(Legs::Dense))
            .expect("query")
    };
    let before = query();
    assert_eq!(before.len(), 1);
    drop(namespace);
    assert_eq!(before, query());
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "one native sample with the installed default model"]
fn measure_namespace_session_lifetime() {
    use std::sync::Arc;
    use std::time::Instant;
    use volantis_db::ingest::{DocId, DocumentVersion, IngestBatch, IngestDocument, Revision};
    use volantis_db::lifecycle::{OpenOptions, Store};
    use volantis_db_text::{ModelOpenOptions, ModelRef, NamespaceModel};
    unsafe extern "C" {
        fn mlx_get_active_memory(result: *mut usize) -> libc::c_int;
        fn mlx_get_cache_memory(result: *mut usize) -> libc::c_int;
    }
    fn sample(label: &str) {
        let (mut active, mut cache) = (0, 0);
        // SAFETY: valid size_t outputs, sampled after synchronous writes/close on idle workers.
        unsafe {
            assert_eq!(mlx_get_active_memory(&mut active), 0);
            assert_eq!(mlx_get_cache_memory(&mut cache), 0);
        }
        println!("{label} active_bytes={active} cached_bytes={cache}");
    }
    let root = tempdir().expect("root");
    let pin = ModelRef::default_model();
    let open_write = |name: &str, id: u128, text: &str| {
        let start = Instant::now();
        let model = Arc::new(
            NamespaceModel::open_model(
                &pin,
                &ModelOpenOptions {
                    offline: true,
                    ..Default::default()
                },
            )
            .expect("model"),
        );
        let store = Store::open_embedded(root.path().join(name), OpenOptions::new(), model.clone())
            .expect("store");
        let open = start.elapsed();
        let start = Instant::now();
        store
            .ingest(IngestBatch::new(vec![
                IngestDocument::new(
                    DocumentVersion::new(DocId::new(id), Revision::new(1)),
                    vec![],
                )
                .with_text(text),
            ]))
            .expect("ingest");
        println!(
            "{name} open={open:?} write={:?} workers={} document_loads={}",
            start.elapsed(),
            NamespaceModel::worker_starts(),
            model.document_load_count().expect("loads")
        );
        assert_eq!(model.document_load_count().expect("loads"), 1);
        store
    };
    let writer = open_write("fake_notes", 59801, "fixture copper kite");
    let peer = open_write("fake_segments", 59802, "fixture violet spool");
    assert_eq!(NamespaceModel::worker_starts(), 1);
    sample("both writes");
    peer.close().expect("peer close");
    sample("peer closed");
    writer.close().expect("final close");
    sample("final close");
}
