#![cfg(feature = "text")]

mod common;
use zeppelin_embed_ffi::*;

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
        "/../zeppelin-embed-text/tests/fixtures/ze115/bert.zem"
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
