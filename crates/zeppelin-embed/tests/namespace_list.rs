use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::Path;

use zeppelin_embed::lifecycle::{OpenOptions, Store, StoreError, StoreErrorKind, list_namespaces};
use zeppelin_embed::meta::Schema;

#[test]
fn list_namespaces_returns_sorted_direct_manifest_directories() {
    let root = tempfile::tempdir().expect("namespace root");
    let root = root.path();
    assert!(list_namespaces(root).expect("list empty root").is_empty());

    let schema = Schema::new(Vec::new()).expect("namespace schema");
    for name in ["zeta", "alpha"] {
        let options = OpenOptions::new().with_schema(schema.clone());
        let store = Store::open(root.join(name), options).expect("create namespace");
        store.close().expect("close namespace");
    }
    std::fs::create_dir_all(root.join("unrelated/nested")).expect("nested directory");
    std::fs::write(root.join("unrelated/nested/manifest.ze"), []).expect("nested manifest");
    std::fs::create_dir_all(root.join("directory-manifest/manifest.ze"))
        .expect("non-file manifest");
    std::fs::write(root.join("regular-file"), []).expect("unrelated file");

    assert_eq!(
        list_namespaces(root).expect("list namespaces"),
        [OsString::from("alpha"), OsString::from("zeta")]
    );
}

#[test]
fn list_namespaces_rejects_invalid_root_paths() {
    for root in [Path::new(""), Path::new("root\0invalid")] {
        let error = list_namespaces(root).expect_err("invalid root must be rejected");
        assert_eq!(error.kind(), StoreErrorKind::InvalidArgument);
    }
}

#[test]
fn list_namespaces_reports_missing_root_as_io() {
    let root = tempfile::tempdir().expect("parent directory");
    let missing = root.path().join("missing");
    let error = list_namespaces(&missing).expect_err("missing root must fail");
    assert!(matches!(
        error,
        StoreError::Io { path, source }
            if path == missing && source.kind() == ErrorKind::NotFound
    ));
}
