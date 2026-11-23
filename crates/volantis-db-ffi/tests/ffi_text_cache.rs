#![cfg(feature = "text")]

mod common;
use volantis_db_ffi::*;

#[test]
#[ignore = "requires native Metal"]
fn text_open_cache_limit_reaches_mlx_and_conflicts_survive_close() {
    unsafe extern "C" {
        fn mlx_set_cache_limit(old: *mut usize, bytes: usize) -> i32;
    }
    let dir = tempfile::tempdir().expect("store directory");
    let path = dir.path().to_str().expect("UTF-8");
    let bundle = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../volantis-db-text/tests/fixtures/ze115/bert.zem"
    );
    let mut request: ZeTextOpenRequestV2 = common::sized_zeroed();
    request.open = common::sized_zeroed();
    request.open.store = common::sized_zeroed();
    request.open.store.commit_tier = 1;
    request.open.store.reader_drain_timeout_ms = 100;
    request.open.store.path = path.as_ptr();
    request.open.store.path_len = path.len();
    request.open.store.max_resident_bytes = u64::MAX;
    request.open.store.max_temp_bytes = u64::MAX;
    request.open.bundle_path = bundle.as_ptr();
    request.open.bundle_path_len = bundle.len();
    request.mlx_cache_limit_bytes = 256 * 1024 * 1024;
    let mut handle = 0;
    for _ in 0..2 {
        assert_eq!(ze_text_open_v2(&request, &mut handle), ZeErrorCode::ZeOk);
        assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
    }
    // MLX returns the previous native limit. Setting the same value leaves it intact.
    let mut previous = 0;
    assert_eq!(
        unsafe { mlx_set_cache_limit(&mut previous, 256 * 1024 * 1024) },
        0
    );
    assert_eq!(previous, 256 * 1024 * 1024);
    request.mlx_cache_limit_bytes = 0;
    assert_eq!(
        ze_text_open_v2(&request, &mut handle),
        ZeErrorCode::ZeErrMlxCacheLimitConflict
    );
}

#[test]
#[ignore = "requires native Metal"]
fn namespace_cache_limit_reaches_native_mlx() {
    use common::namespace_cache::*;
    unsafe extern "C" {
        fn mlx_set_cache_limit(old: *mut usize, bytes: usize) -> i32;
    }
    isolated(
        "namespace_cache_limit_reaches_native_mlx",
        true,
        |root, mut spec| {
            let (status, handle) = open(root, b"toy-a", &spec.embedded.spec);
            assert_eq!(status, ZeErrorCode::ZeOk);
            let doc = row(42);
            let request = ZeUpsertRequest {
                documents: &doc,
                document_count: 1,
                ..common::sized_zeroed()
            };
            let mut report = common::sized_zeroed();
            assert_eq!(ze_upsert(handle, &request, &mut report), ZeErrorCode::ZeOk);
            let mut previous = 0;
            assert_eq!(unsafe { mlx_set_cache_limit(&mut previous, 536870912) }, 0);
            println!("native MLX cache limit: {previous} bytes");
            assert_eq!(previous, 536870912);
            let mut legacy = spec.embedded;
            legacy.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpec>() as u32;
            legacy.spec.has_vector_space = 2;
            let (status, peer) = open(root, b"toy-b", &legacy.spec);
            assert_eq!(status, ZeErrorCode::ZeOk);
            spec.mlx_cache_limit_bytes = 0;
            assert_eq!(
                open(root, b"toy-c", &spec.embedded.spec),
                (ZeErrorCode::ZeErrMlxCacheLimitConflict, 0)
            );
            for owner in [peer, handle] {
                assert_eq!(ze_close(owner), ZeErrorCode::ZeOk);
            }
        },
    );
}
