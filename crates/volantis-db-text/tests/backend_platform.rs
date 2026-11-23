//! WF-990: a target without MLX fails every model open before model I/O.
#![cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#![allow(clippy::expect_used, clippy::panic)]

use std::cell::Cell;
use std::path::Path;

use volantis_db_text::runtime::RuntimeError;
use volantis_db_text::{
    ModelError, ModelOpenOptions, ModelRef, NamespaceModel, TextError, TextOpenOptions, TextStore,
    resolve_model,
};

fn fake_model() -> ModelRef {
    ModelRef {
        repo: "fixture-owner/fake-990".to_owned(),
        revision: "1".repeat(40),
        manifest_sha256: "2".repeat(64),
    }
}

fn assert_unsupported<T>(result: Result<T, TextError>, call: &str) {
    match result {
        Err(TextError::Runtime(RuntimeError::UnsupportedBackend { backend })) => {
            assert_eq!(backend, "cpu", "{call}");
        }
        Err(other) => panic!("{call}: expected an unsupported backend, got {other}"),
        Ok(_) => panic!("{call}: opened without a model runtime"),
    }
}

fn is_empty(directory: &Path) -> bool {
    std::fs::read_dir(directory)
        .expect("read cache directory")
        .next()
        .is_none()
}

#[test]
fn unsupported_backend_precedes_model_io() {
    let root = tempfile::tempdir().expect("temporary root");
    let cache = root.path().join("cache");
    std::fs::create_dir(&cache).expect("private empty cache");
    let store = root.path().join("fake990model");
    let bundle = root.path().join("fake-990-missing.zem");
    let model = fake_model();
    let progress_calls = Cell::new(0_u32);
    let progress = |_: u64, _: u64| progress_calls.set(progress_calls.get() + 1);
    let options = |limit: usize| ModelOpenOptions {
        text: TextOpenOptions::default().with_mlx_cache_limit_bytes(limit),
        cache_dir: Some(cache.clone()),
        offline: true,
        progress: Some(&progress),
        ..ModelOpenOptions::default()
    };

    // Two different explicit cache limits: neither open may install a limit,
    // so the second open must not report a cache-limit conflict.
    for limit in [1, 2] {
        assert_unsupported(
            TextStore::open(
                &store,
                &bundle,
                TextOpenOptions::default().with_mlx_cache_limit_bytes(limit),
            ),
            "TextStore::open",
        );
        assert_unsupported(
            TextStore::open_model(&store, &model, options(limit)),
            "TextStore::open_model",
        );
        assert_unsupported(
            TextStore::open_default_model(&store, options(limit)),
            "TextStore::open_default_model",
        );
        assert_unsupported(
            NamespaceModel::open_bundle(&bundle),
            "NamespaceModel::open_bundle",
        );
        assert_unsupported(
            NamespaceModel::open_model(&model, &options(limit)),
            "NamespaceModel::open_model",
        );
    }
    assert!(!store.exists(), "a rejected open created the store");
    assert!(is_empty(&cache), "a rejected open touched the model cache");
    assert_eq!(progress_calls.get(), 0, "a rejected open reported progress");

    // Direct resolution needs no runtime: macOS resolves, other targets refuse.
    let resolved = resolve_model(&model, &options(1));
    #[cfg(target_os = "macos")]
    assert!(
        matches!(resolved, Err(ModelError::OfflineMiss)),
        "{resolved:?}"
    );
    #[cfg(not(target_os = "macos"))]
    assert!(
        matches!(resolved, Err(ModelError::UnsupportedBackend)),
        "{resolved:?}"
    );
    assert!(
        is_empty(&cache),
        "offline resolution touched the model cache"
    );
}
