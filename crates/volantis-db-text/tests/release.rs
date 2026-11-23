#![allow(clippy::expect_used)]

use std::path::PathBuf;
use tempfile::tempdir;
use volantis_db_text::bundle::{Bundle, TensorDtype};
use volantis_db_text::{IngestOptions, Legs, QueryOptions, TextDocument, TextStore};

/// The published default bundle, or the one `.zem` in `WF987_RELEASE`.
fn release_bundle() -> PathBuf {
    let Some(directory) = std::env::var_os("WF987_RELEASE").map(PathBuf::from) else {
        return volantis_db_text::resolve_model(
            &volantis_db_text::ModelRef::default_model(),
            &Default::default(),
        )
        .expect("resolve published release model");
    };
    let mut paths = std::fs::read_dir(directory)
        .expect("release directory")
        .map(|p| p.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "zem"));
    let bundle = paths.next().expect("one .zem");
    assert!(
        paths.next().is_none(),
        "release must contain exactly one bundle"
    );
    bundle
}

#[test]
#[ignore = "requires cached HF model (or WF987_RELEASE) and native MLX"]
fn release_opens_ingests_three_documents_and_queries_all_legs() {
    let bundle = release_bundle();
    // WF-987 ships MLX only: no Core ML query companion beside the bundle.
    assert!(!bundle.with_extension("mlmodelc").exists());
    let scratch = tempdir().expect("scratch store");
    let store = TextStore::open(scratch.path(), &bundle, Default::default()).expect("release open");
    let report = store
        .ingest_text(
            &[
                TextDocument::new(1, 1, "The bronze zeppelin flies above the city."),
                TextDocument::new(2, 1, "Fresh bread is baked every morning."),
                TextDocument::new(3, 1, "Whales swim in the ocean."),
            ],
            IngestOptions::default(),
        )
        .expect("ingest three documents");
    assert_eq!(report.chunks, 3);
    for legs in [Legs::Dense, Legs::Lexical, Legs::Hybrid] {
        let result = store
            .query_text_with_diagnostics("bronze zeppelin", QueryOptions::new(3).with_legs(legs))
            .expect("release query");
        assert_eq!(
            result.hits.first().map(|hit| hit.doc_id),
            Some(1),
            "{legs:?}"
        );
        if legs != Legs::Lexical {
            assert_eq!(
                result.backend.expect("dense query backend").runtime.name,
                "mlx-c"
            );
        }
    }
    store.close().expect("close store");
}

#[test]
#[ignore = "requires cached HF model (or WF987_RELEASE); no MLX"]
fn release_bundle_is_the_fp16_e5_pair() {
    let bundle = Bundle::open(release_bundle()).expect("release bundle");
    assert!(!bundle.is_symmetric());
    for (role, tower, prefix, limit) in [
        ("query", bundle.query_tower(), "query: ", 96),
        ("document", bundle.document_tower(), "passage: ", 512),
    ] {
        assert_eq!(tower.embedding.prompt_prefix, prefix, "{role}");
        assert_eq!(tower.embedding.max_tokens, limit, "{role}");
        assert_eq!(tower.embedding.dims, 384, "{role}");
        assert_eq!(
            bundle
                .tensor(&format!("{role}/embeddings.word_embeddings.weight"))
                .expect("word weights")
                .dtype,
            TensorDtype::F16,
            "{role}"
        );
    }
}
