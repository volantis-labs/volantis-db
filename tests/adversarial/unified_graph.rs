//! Unified authority fault operations, using the existing scheduled VFS.
use super::coverage::CoverageRegistry;
use super::storage_durability::release_fixture as fixture;

pub const REQUIRED: &[&str] = &[
    "storage-durability.graph-commit.enable",
    "storage-durability.graph-commit.artifact-write",
    "storage-durability.graph-commit.artifact-sync",
    "storage-durability.graph-commit.wal-append",
    "storage-durability.graph-commit.wal-sync",
    "storage-durability.graph-fold.manifest-rename",
    "storage-durability.mixed-batch.append-member-0",
    "storage-durability.mixed-batch.append-member-1",
    "storage-durability.mixed-batch.append-member-2",
    "storage-durability.mixed-batch.sync",
    "storage-durability.mixed-batch.comparator",
    "storage-durability.embedded-batch.encoder-fault",
    "storage-durability.embedded-batch.clean",
    "storage-durability.embedded-batch.neighbour-visibility",
    "storage-durability.deferred.pending-commit-restart",
    "storage-durability.deferred.completion-restart",
    "storage-durability.deferred.stale-completion",
];

pub const PART_A_REQUIRED: &[&str] = &[
    "op.enable_graph",
    "op.graph_apply",
    "op.mixed_batch",
    "storage-durability.graph-commit.fire",
    "storage-durability.graph-commit.clean",
    "storage-durability.graph-commit.enable.fire",
    "storage-durability.graph-commit.enable.clean",
    "storage-durability.graph-commit.artifact-write.fire",
    "storage-durability.graph-commit.artifact-write.clean",
    "storage-durability.graph-commit.artifact-sync.fire",
    "storage-durability.graph-commit.artifact-sync.clean",
    "storage-durability.graph-commit.node-text.preparation-cancel",
    "storage-durability.graph-commit.wal-append.fire",
    "storage-durability.graph-commit.wal-append.clean",
    "storage-durability.graph-commit.wal-sync.fire",
    "storage-durability.graph-commit.wal-sync.clean",
    "storage-durability.graph-fold.fire",
    "storage-durability.graph-fold.clean",
    "storage-durability.graph-fold.manifest-rename.fire",
    "storage-durability.graph-fold.manifest-rename.clean",
    "storage-durability.graph-fold.rotation.fire",
    "storage-durability.graph-fold.rotation.clean",
    "storage-durability.mixed-batch.fire",
    "storage-durability.mixed-batch.clean",
    "storage-durability.mixed-batch.members.fire",
    "storage-durability.mixed-batch.members.clean",
    "storage-durability.mixed-batch.sync.fire",
    "storage-durability.mixed-batch.sync.clean",
    "storage-durability.mixed-batch.torn-final.fire",
    "storage-durability.mixed-batch.torn-final.clean",
];

#[derive(Clone, Copy)]
pub enum Operation {
    EnableGraph,
    GraphApply,
    MixedBatch,
    DocumentRelationship,
    DocumentDelete,
    MissingDocumentEndpoint,
}

pub fn run(operation: Operation, seed: u64, coverage: &mut CoverageRegistry) -> Result<(), String> {
    match operation {
        Operation::EnableGraph => enable_graph(seed, coverage),
        Operation::GraphApply => super::graph_recovery::probe_commit_boundaries(seed, coverage),
        Operation::MixedBatch => mixed_batch(seed, coverage),
        Operation::DocumentRelationship
        | Operation::DocumentDelete
        | Operation::MissingDocumentEndpoint => {
            super::runner::unified::exercise_document_operation(operation, seed, coverage)
        }
    }
}

fn enable_graph(seed: u64, coverage: &mut CoverageRegistry) -> Result<(), String> {
    use super::fault_vfs::{FaultEvent, FaultMode, FaultSchedule, FaultSite, Layer, ScheduledVfs};
    use std::sync::Arc;
    use volantis_db::lifecycle::{Store, StoreTestDependencies, SystemMonotonicClock};
    use volantis_db::manifest::decode_manifest;
    use volantis_db::vfs::StdVfs;
    let root = tempfile::tempdir().map_err(|error| error.to_string())?;
    for (ordinal, (site, needle)) in [
        (FaultSite::Write, ".zgraph"),
        (FaultSite::Sync, ".zgraph"),
        (FaultSite::Rename, "manifest.ze"),
    ]
    .into_iter()
    .enumerate()
    {
        for mode in [FaultMode::Eio, FaultMode::PostCommitError] {
            let path = root.path().join(format!("{seed}-{ordinal}-{mode:?}"));
            let event = FaultEvent {
                id: "unified-enable".into(),
                op_index: 1,
                layer: Layer::Io,
                site,
                mode,
                nth_match: 1,
                expected_matches: None,
                deadline_budget_seconds: None,
                path_contains: Some(needle.into()),
                fired: false,
                fire_count: 0,
                path: None,
            };
            let vfs = Arc::new(ScheduledVfs::new(StdVfs, FaultSchedule::single(event)));
            let store = Store::open_with_test_dependencies(
                &path,
                fixture::options(false),
                StoreTestDependencies::new(vfs.clone(), Arc::new(SystemMonotonicClock)),
            )
            .map_err(|error| error.to_string())?;
            vfs.set_operation(1);
            store.enable_graph().expect_err("enable fault must refuse");
            let events = vfs.events();
            if events.len() != 1 || events[0].fire_count != 1 || !events[0].fired {
                return Err("enable graph fault did not fire exactly once".into());
            }
            drop(store);
            let manifest_path = path.join("manifest.ze");
            let bytes = std::fs::read(&manifest_path).map_err(|error| error.to_string())?;
            let manifest =
                decode_manifest("enable fault", &bytes).map_err(|error| error.to_string())?;
            let committed = site == FaultSite::Rename && mode == FaultMode::PostCommitError;
            if manifest.graph.is_some() != committed || manifest.generation != u64::from(committed)
            {
                return Err("enable graph crossed its manifest commit boundary".into());
            }
            let reader =
                Store::open(&path, fixture::options(true)).map_err(|error| error.to_string())?;
            if reader
                .count_documents(None, None)
                .map_err(|error| error.to_string())?
                .count
                != 0
            {
                return Err("enable graph changed document population".into());
            }
            reader.close().map_err(|error| error.to_string())?;
            if std::fs::read(&manifest_path).map_err(|error| error.to_string())? != bytes {
                return Err("read-only recovery changed the manifest".into());
            }
            coverage.hit(REQUIRED[0]);
        }
    }
    let control = root.path().join("control");
    let store =
        Store::open(&control, fixture::options(false)).map_err(|error| error.to_string())?;
    if store.enable_graph().map_err(|error| error.to_string())? != 1 {
        return Err("enable control did not advance one Store generation".into());
    }
    store.close().map_err(|error| error.to_string())?;
    coverage.hit("op.EnableGraph");
    Ok(())
}

pub fn probe(seed: u64, coverage: &mut CoverageRegistry) -> Result<(), String> {
    for operation in [
        Operation::EnableGraph,
        Operation::GraphApply,
        Operation::MixedBatch,
        Operation::DocumentRelationship,
        Operation::DocumentDelete,
        Operation::MissingDocumentEndpoint,
    ] {
        run(operation, seed, coverage)?;
    }
    Ok(())
}

fn mixed_batch(seed: u64, coverage: &mut CoverageRegistry) -> Result<(), String> {
    use super::fault_vfs::{FaultEvent, FaultMode, FaultSchedule, FaultSite, Layer, ScheduledVfs};
    use std::sync::Arc;
    use volantis_db::graph_recovery_test_support::{apply_mixed_batch, mixed_node_present};
    use volantis_db::lifecycle::{Store, StoreTestDependencies, SystemMonotonicClock};
    use volantis_db::property_graph::NodeId;
    use volantis_db::vfs::StdVfs;
    let root = tempfile::tempdir().map_err(|e| e.to_string())?;
    let batch = fixture::batch(vec![
        fixture::document(91, "mixed orchard"),
        fixture::document(92, "mixed harbor"),
    ]);
    let clean_path = root.path().join("control");
    let clean = Store::open(
        &clean_path,
        fixture::options(false)
            .with_durability(
                volantis_db::lifecycle::durability::DurabilityMode::Durable,
                volantis_db::lifecycle::durability::CommitTier::Durable,
            )
            .with_schema(fixture::schema()),
    )
    .map_err(|e| e.to_string())?;
    clean.enable_graph().map_err(|e| e.to_string())?;
    let (node, generation) = apply_mixed_batch(&clean, &batch, "first")?;
    if generation != 2
        || node != NodeId::new(1).map_err(|e| e.to_string())?
        || clean
            .count_documents(None, None)
            .map_err(|e| e.to_string())?
            .count
            != 2
        || !mixed_node_present(&clean, node)
    {
        return Err("mixed control differs from literal first-batch state".into());
    }
    drop(clean);
    for (site, nth, mode, key, present) in [
        (FaultSite::Append, 1, FaultMode::Eio, REQUIRED[6], false),
        (FaultSite::Append, 2, FaultMode::Eio, REQUIRED[7], false),
        (FaultSite::Append, 3, FaultMode::Eio, REQUIRED[8], false),
        (
            FaultSite::Append,
            3,
            FaultMode::PostCommitError,
            REQUIRED[8],
            true,
        ),
        (FaultSite::Sync, 1, FaultMode::Eio, REQUIRED[9], true),
    ] {
        let path = root.path().join(format!("{seed}-{site:?}-{nth}-{mode:?}"));
        let event = FaultEvent {
            id: "mixed-batch".into(),
            op_index: 1,
            layer: Layer::Io,
            site,
            mode,
            nth_match: nth,
            expected_matches: None,
            deadline_budget_seconds: None,
            path_contains: Some("wal.ze".into()),
            fired: false,
            fire_count: 0,
            path: None,
        };
        let vfs = Arc::new(ScheduledVfs::new(StdVfs, FaultSchedule::single(event)));
        let store = Store::open_with_test_dependencies(
            &path,
            fixture::options(false)
                .with_durability(
                    volantis_db::lifecycle::durability::DurabilityMode::Durable,
                    volantis_db::lifecycle::durability::CommitTier::Durable,
                )
                .with_schema(fixture::schema()),
            StoreTestDependencies::new(vfs.clone(), Arc::new(SystemMonotonicClock)),
        )
        .map_err(|e| e.to_string())?;
        store.enable_graph().map_err(|e| e.to_string())?;
        vfs.set_operation(1);
        if apply_mixed_batch(&store, &batch, "first").is_ok() {
            return Err(format!(
                "mixed {site:?}/{nth}/{mode:?} acknowledged a fault"
            ));
        }
        let events = vfs.events();
        if events.len() != 1 || events[0].fire_count != 1 {
            return Err(format!(
                "mixed member {nth} fault did not fire exactly once"
            ));
        }
        if store
            .ingest(fixture::batch(vec![fixture::document(93, "fenced")]))
            .is_ok()
        {
            return Err("mixed indeterminate failure did not fence the shared writer".into());
        }
        drop(store);
        for read_only in [false, true, false] {
            let recovered = Store::open(
                &path,
                fixture::options(read_only).with_durability(
                    volantis_db::lifecycle::durability::DurabilityMode::Durable,
                    volantis_db::lifecycle::durability::CommitTier::Durable,
                ),
            )
            .map_err(|e| e.to_string())?;
            let count = recovered
                .count_documents(None, None)
                .map_err(|e| e.to_string())?
                .count;
            let actual_generation = recovered
                .snapshot()
                .map_err(|e| e.to_string())?
                .generation();
            if count != if present { 2 } else { 0 }
                || mixed_node_present(&recovered, node) != present
                || actual_generation != if present { 2 } else { 1 }
            {
                return Err(format!(
                    "mixed member {nth} crossed atomic boundary: count={count}, gen={actual_generation}"
                ));
            }
        }
        coverage.hit(key);
    }
    coverage.hit(REQUIRED[10]);
    embedded_batch(seed, coverage).map_err(|e| e.to_string())?;
    deferred_restart(seed, coverage).map_err(|e| e.to_string())?;
    coverage.hit("op.MixedBatch");
    Ok(())
}

#[path = "../../crates/volantis-db/tests/test_support/namespace_model.rs"]
mod namespace_model;

fn embedded_batch(
    seed: u64,
    coverage: &mut CoverageRegistry,
) -> Result<(), Box<dyn std::error::Error>> {
    use volantis_db::ingest::{DocId, DocumentVersion, IngestDocument, Revision};
    use volantis_db::lifecycle::embedded::NeighbourOptions;
    use volantis_db::lifecycle::{CancelToken, QueryControl, QueryError};
    use volantis_db::lifecycle::{
        DocumentFields, NamespaceMutation, OpenOptions, Store, StoreError, namespace_batch,
    };
    let root = tempfile::tempdir()?;
    let options = OpenOptions::new().with_embedder(std::sync::Arc::new(namespace_model::Model))?;
    let id = DocId::new(u128::MAX - u128::from(seed));
    for name in ["first", "second"] {
        drop(Store::open(root.path().join(name), options.clone())?);
    }
    for (text, expected) in [("!fail", false), ("clean", true)] {
        let batch = [("first", "first"), ("second", text)]
            .into_iter()
            .map(|(name, source)| NamespaceMutation {
                name: name.into(),
                options: options.clone(),
                deletes: vec![],
                delete_where: None,
                upserts: vec![
                    IngestDocument::new(DocumentVersion::new(id, Revision::new(1)), vec![])
                        .with_text(source),
                ],
            })
            .collect();
        let result = namespace_batch(root.path(), batch);
        match result {
            Ok(_) if expected => {}
            Err(StoreError::Embedding(_)) if !expected => {}
            other => return Err(format!("unexpected encoder outcome {other:?}").into()),
        }
        for name in ["first", "second"] {
            let store = Store::open(root.path().join(name), options.clone())?;
            if store
                .get_documents(&[id], DocumentFields::ALL)?
                .into_iter()
                .flatten()
                .count()
                != usize::from(expected)
            {
                return Err("encoder fault crossed the atomic publication boundary".into());
            }
            let result = store.nearest_chunks(
                id,
                None,
                &NeighbourOptions {
                    include_source: true,
                    ..Default::default()
                },
                QueryControl::Cancel(CancelToken::new()),
            );
            match result {
                Ok(hits)
                    if expected
                        && matches!(hits.as_slice(), [hit] if hit.id == id && hit.chunk == 0 && hit.score == 0.) =>
                    {}
                Err(QueryError::Store(StoreError::NeighbourSourceNotFound {
                    id: missing,
                    chunk: None,
                })) if !expected && missing == id => {}
                other => {
                    return Err(
                        format!("neighbour visibility after namespace batch: {other:?}").into(),
                    );
                }
            }
        }
        coverage.hit(if expected {
            "storage-durability.embedded-batch.clean"
        } else {
            "storage-durability.embedded-batch.encoder-fault"
        });
    }
    coverage.hit("storage-durability.embedded-batch.neighbour-visibility");
    Ok(())
}

/// Stored vector and pending-row count read back from one crash image.
type RowImage = (Option<Vec<f32>>, u64);

/// WF-985: a crash image taken after each publish boundary of deferred
/// embedding (pending commit, completion, stale completion after purge and
/// reinsert) reopens to exactly the acknowledged row state.
fn deferred_restart(
    seed: u64,
    coverage: &mut CoverageRegistry,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Duration;
    use volantis_db::ingest::{
        DeleteBatch, DocId, DocumentVersion, IngestBatch, IngestDocument, Revision,
    };
    use volantis_db::lifecycle::embedded::{DocumentSchedule, EmbeddedLeg, EmbeddedRowState};
    use volantis_db::lifecycle::{CancelToken, DocumentFields, OpenOptions, QueryControl, Store};
    let root = tempfile::tempdir()?;
    let path = root.path().join("deferred");
    let model = Arc::new(namespace_model::GateModel::default());
    let open =
        |path: &std::path::Path, schedule| -> Result<Arc<Store>, Box<dyn std::error::Error>> {
            let options = OpenOptions::new()
                .with_embedder(model.clone())?
                .with_document_schedule(schedule);
            Ok(Arc::new(Store::open(path, options)?))
        };
    let doc = |id: DocId, text: &str| {
        IngestDocument::new(DocumentVersion::new(id, Revision::new(7)), vec![]).with_text(text)
    };
    // Reopens a copy of the store directory taken while the writer is open.
    let image = |id: DocId| -> Result<RowImage, Box<dyn std::error::Error>> {
        let copy = super::storage_durability::copy_directory(&path)?;
        let store = open(copy.path(), DocumentSchedule::Paused)?;
        let row = store
            .get_documents(&[id], DocumentFields::ALL)?
            .pop()
            .flatten()
            .ok_or("acknowledged row missing after restart")?;
        let report = store.query_embedded_report(
            "fictitious",
            EmbeddedLeg::Lexical,
            4,
            None,
            QueryControl::Cancel(CancelToken::new()),
        )?;
        if report.hits.iter().any(|hit| {
            hit.version.doc_id() == id
                && (hit.state == EmbeddedRowState::Pending) != row.vector.is_none()
        }) {
            return Err("row state and stored vector disagree after restart".into());
        }
        if !volantis_db::verify::verify_store(copy.path())?.is_clean() {
            return Err("crash image does not verify".into());
        }
        Ok((row.vector, report.pending_rows))
    };
    let first = DocId::new(u128::MAX - 2 * u128::from(seed) - 1);
    let second = DocId::new(u128::MAX - 2 * u128::from(seed) - 2);
    let store = open(&path, DocumentSchedule::Paused)?;
    store.ingest(IngestBatch::new(vec![doc(
        first,
        "fictitious mauve sprocket",
    )]))?;
    if image(first)? != (None, 1) {
        return Err("pending commit did not survive restart".into());
    }
    coverage.hit("storage-durability.deferred.pending-commit-restart");
    store.attach_embedding_worker()?;
    store.resume_embedding()?;
    if store.wait_embedding_idle(Duration::from_secs(30))?.pending != 0 {
        return Err("worker did not drain".into());
    }
    if image(first)? != (Some(vec![1., 0.]), 0) {
        return Err("completion did not survive restart".into());
    }
    coverage.hit("storage-durability.deferred.completion-restart");
    model.close_gate();
    store.ingest(IngestBatch::new(vec![doc(
        second,
        "fictitious cyan thimble",
    )]))?;
    if !model.wait_entered(2, Duration::from_secs(30)) {
        return Err("worker did not admit the second batch".into());
    }
    store.pause_embedding()?;
    store.delete(DeleteBatch::new(vec![second]))?;
    store.await_physical_purge(store.purge(&[second])?)?;
    store.ingest(IngestBatch::new(vec![doc(
        second,
        "fictitious cyan thimble",
    )]))?;
    model.open_gate();
    store.wait_embedding_idle(Duration::from_secs(30))?;
    if image(second)? != (None, 1) {
        return Err("a stale completion resurrected a purged admission".into());
    }
    coverage.hit("storage-durability.deferred.stale-completion");
    store.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_batch_append_member_faults_fire() {
        let mut coverage = CoverageRegistry::default();
        mixed_batch(256, &mut coverage).unwrap();
        for key in &REQUIRED[6..] {
            assert!(coverage.count(key) > 0, "missing {key}");
        }
    }

    #[test]
    fn unified_graph_fault_sites_are_registered_and_fire() {
        let mut coverage = CoverageRegistry::default();
        probe(256, &mut coverage).unwrap();
        for key in REQUIRED {
            assert!(
                coverage.count(key) > 0,
                "missing unified fault receipt: {key}"
            );
        }
    }

    #[test]
    fn document_endpoint_operations_are_registered_and_executed() {
        let mut coverage = CoverageRegistry::default();
        probe(358, &mut coverage).unwrap();
        let required: Vec<_> = super::super::coverage::required_smoke_coverage().collect();
        let expected = crate::coverage_key_expectations::expected_smoke_keys();
        for key in [
            "op.graph_apply.document-endpoint",
            "op.delete.document-no-relationships",
            "op.delete.document-restrict",
            "op.graph_apply.missing-document-endpoint",
        ] {
            assert!(coverage.count(key) > 0, "operation did not execute: {key}");
            assert!(required.contains(&key), "unregistered {key}");
            assert!(expected.contains(&key), "expectations omit {key}");
        }
    }
}
