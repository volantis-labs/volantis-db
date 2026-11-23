//! WF-990: a text build without MLX keeps its C ABI and refuses model work.
#![cfg(all(
    feature = "text",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]

mod common;
use std::mem::size_of;
use std::path::Path;
use volantis_db_ffi::*;

const SENTINEL: ZeHandle = 0x0990_0990;
const REPO: &[u8] = b"fixture-owner/fake-990";
const REVISION: &[u8] = b"1111111111111111111111111111111111111111";
const MANIFEST: &[u8] = b"2222222222222222222222222222222222222222222222222222222222222222";

fn bytes(path: &Path) -> Vec<u8> {
    path.to_str().expect("UTF-8 path").as_bytes().to_vec()
}

fn last_error() -> String {
    let mut buffer = [0_u8; 512];
    let mut written = 0;
    assert_eq!(
        ze_last_error_message(0, buffer.as_mut_ptr().cast(), buffer.len(), &mut written),
        ZeErrorCode::ZeOk
    );
    String::from_utf8(buffer[..written].to_vec()).expect("UTF-8 last error")
}

fn assert_unsupported(code: ZeErrorCode, handle: ZeHandle, call: &str) {
    assert_eq!(code, ZeErrorCode::ZeErrUnsupported, "{call}");
    assert_eq!(code as i32, 11, "{call}");
    assert_eq!(handle, SENTINEL, "{call} wrote the handle");
    assert!(
        last_error().contains("cpu model runtime is not available for this build target"),
        "{call}: {}",
        last_error()
    );
}

fn store_request(path: &[u8]) -> ZeOpenRequest {
    ZeOpenRequest {
        path: path.as_ptr(),
        path_len: path.len(),
        commit_tier: 1,
        reader_drain_timeout_ms: 100,
        max_resident_bytes: u64::MAX,
        max_temp_bytes: u64::MAX,
        ..common::sized_zeroed()
    }
}

fn model_ref() -> ZeTextModelRef {
    ZeTextModelRef {
        repo: REPO.as_ptr(),
        repo_len: REPO.len(),
        revision: REVISION.as_ptr(),
        revision_len: REVISION.len(),
        manifest_sha256: MANIFEST.as_ptr(),
        manifest_sha256_len: MANIFEST.len(),
        ..common::sized_zeroed()
    }
}

/// Tag 5 holds tags 2, 3 and 4 as prefixes with their own sizes.
fn embedded_spec(cache: &[u8]) -> ZeEmbeddedNamespaceSpecV4 {
    let mut spec: ZeEmbeddedNamespaceSpecV4 = common::sized_zeroed();
    spec.v3 = common::sized_zeroed();
    spec.v3.v2 = common::sized_zeroed();
    spec.v3.v2.embedded = common::sized_zeroed();
    spec.v3.v2.embedded.spec = common::sized_zeroed();
    spec.v3.v2.embedded.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpecV4>() as u32;
    spec.v3.v2.embedded.spec.has_vector_space = 5;
    spec.v3.v2.embedded.model = model_ref();
    spec.v3.v2.mlx_cache_limit_bytes = 1 << 20;
    spec.v3.has_mlx_cache_limit = 1;
    spec.v3.model_cache_dir = cache.as_ptr();
    spec.v3.model_cache_dir_len = cache.len();
    spec.v3.offline = 1;
    spec.document_schedule = 1;
    spec
}

fn with_tag(mut spec: ZeEmbeddedNamespaceSpecV4, tag: u32) -> ZeEmbeddedNamespaceSpecV4 {
    spec.v3.v2.embedded.spec.has_vector_space = tag;
    spec.v3.v2.embedded.spec.abi_size = match tag {
        2 => size_of::<ZeEmbeddedNamespaceSpec>(),
        3 => size_of::<ZeEmbeddedNamespaceSpecV2>(),
        4 => size_of::<ZeEmbeddedNamespaceSpecV3>(),
        _ => size_of::<ZeEmbeddedNamespaceSpecV4>(),
    } as u32;
    spec
}

fn plain_spec() -> ZeNamespaceSpec {
    ZeNamespaceSpec {
        has_vector_space: 1,
        dimensions: 4,
        ..common::sized_zeroed()
    }
}

fn namespace_open(root: &[u8], name: &[u8], spec: &ZeNamespaceSpec) -> (ZeErrorCode, ZeHandle) {
    let request = ZeNamespaceOpenRequest {
        root: root.as_ptr(),
        root_len: root.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        spec,
        open: store_request(b""),
        ..common::sized_zeroed()
    };
    let mut handle = SENTINEL;
    (ze_namespace_open(&request, &mut handle), handle)
}

/// Returns the generation and hit count of a nearest-neighbour search.
fn search(handle: ZeHandle) -> (u64, usize) {
    let vector = [0.25, 0.5, 0.75, 1.0];
    let request = common::valid_search_request(&vector);
    let mut result: ZeSearchResult = common::sized_zeroed();
    assert_eq!(ze_search(handle, &request, &mut result), ZeErrorCode::ZeOk);
    let observed = (result.generation, result.hit_count);
    assert_eq!(ze_search_result_free(&mut result), ZeErrorCode::ZeOk);
    observed
}

fn write_plain_row(handle: ZeHandle) {
    let vector = [0.25_f32, 0.5, 0.75, 1.0];
    let text = b"violet teapot 990";
    let document = ZeIngestDocument {
        doc_id: ZeDocId { high: 0, low: 990 },
        revision: 1,
        vector: vector.as_ptr(),
        vector_len: vector.len(),
        text: text.as_ptr(),
        text_len: text.len(),
        ..common::sized_zeroed()
    };
    let request = ZeIngestRequest {
        documents: &document,
        document_count: 1,
        dimension: vector.len(),
        ..common::sized_zeroed()
    };
    let mut report: ZeMutationReport = common::sized_zeroed();
    assert_eq!(ze_ingest(handle, &request, &mut report), ZeErrorCode::ZeOk);
}

#[test]
fn text_opens_return_unsupported_on_x64() {
    let root = tempfile::tempdir().expect("temporary root");
    let cache = root.path().join("cache");
    std::fs::create_dir(&cache).expect("private empty cache");
    let cache_bytes = bytes(&cache);
    let store = bytes(&root.path().join("fake990model"));
    let bundle = bytes(&root.path().join("fake-990-missing.zem"));

    let mut open: ZeTextOpenRequest = common::sized_zeroed();
    open.store = store_request(&store);
    open.bundle_path = bundle.as_ptr();
    open.bundle_path_len = bundle.len();
    let mut open_v2: ZeTextOpenRequestV2 = common::sized_zeroed();
    open_v2.open = open;
    open_v2.mlx_cache_limit_bytes = 1 << 20;
    let mut model: ZeTextOpenModelRequest = common::sized_zeroed();
    model.store = store_request(&store);
    model.repo = REPO.as_ptr();
    model.repo_len = REPO.len();
    model.revision = REVISION.as_ptr();
    model.revision_len = REVISION.len();
    model.manifest_sha256 = MANIFEST.as_ptr();
    model.manifest_sha256_len = MANIFEST.len();
    model.cache_dir = cache_bytes.as_ptr();
    model.cache_dir_len = cache_bytes.len();
    model.offline = 1;
    let mut model_v2: ZeTextOpenModelRequestV2 = common::sized_zeroed();
    model_v2.open = model;
    model_v2.mlx_cache_limit_bytes = 1 << 20;

    let mut handle = SENTINEL;
    assert_unsupported(ze_text_open(&open, &mut handle), handle, "ze_text_open");
    assert_unsupported(
        ze_text_open_v2(&open_v2, &mut handle),
        handle,
        "ze_text_open_v2",
    );
    assert_unsupported(
        ze_text_open_model(&model, &mut handle),
        handle,
        "ze_text_open_model",
    );
    assert_unsupported(
        ze_text_open_model_v2(&model_v2, &mut handle),
        handle,
        "ze_text_open_model_v2",
    );

    let namespaces = root.path().join("namespaces");
    let namespace_root = bytes(&namespaces);
    let spec = embedded_spec(&cache_bytes);
    for tag in 2..=5 {
        let spec = with_tag(spec, tag);
        let (code, handle) =
            namespace_open(&namespace_root, b"fake990model", &spec.v3.v2.embedded.spec);
        assert_unsupported(code, handle, &format!("embedded tag {tag}"));
    }

    // ABI validation still runs first and keeps its own code.
    open.abi_size = 4;
    assert_eq!(
        ze_text_open(&open, &mut handle),
        ZeErrorCode::ZeErrInvalidArgument
    );
    let mut truncated = with_tag(spec, 5);
    truncated.v3.v2.embedded.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpecV3>() as u32;
    let (code, handle) = namespace_open(
        &namespace_root,
        b"fake990model",
        &truncated.v3.v2.embedded.spec,
    );
    assert_eq!(code, ZeErrorCode::ZeErrInvalidArgument);
    assert_eq!(handle, SENTINEL);

    assert!(!Path::new(std::str::from_utf8(&store).unwrap()).exists());
    assert!(!namespaces.join("fake990model").exists());
    assert!(std::fs::read_dir(&cache).unwrap().next().is_none());
}

#[test]
fn unsupported_batch_has_no_partial_participants() {
    let root = tempfile::tempdir().expect("temporary root");
    let cache = bytes(&root.path().join("cache"));
    let path = bytes(root.path());
    let plain = plain_spec();
    let (code, handle) = namespace_open(&path, b"fake990plain", &plain);
    assert_eq!(code, ZeErrorCode::ZeOk);
    write_plain_row(handle);
    let before = search(handle);
    assert_eq!(before.1, 1);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);

    // Every participant decodes before the batch starts. A live embedded
    // handle cannot exist here, because every embedded open returns 11.
    let deleted = ZeDocId { high: 0, low: 990 };
    let embedded = with_tag(embedded_spec(&cache), 4);
    let participants = [
        (b"fake990plain".as_slice(), &plain),
        (b"fake990model".as_slice(), &embedded.v3.v2.embedded.spec),
    ]
    .map(|(name, spec)| {
        let mut participant: ZeNamespaceMutation = common::sized_zeroed();
        participant.name = name.as_ptr();
        participant.name_len = name.len();
        participant.spec = spec;
        participant.upserts = common::sized_zeroed();
        participant.upserts.batch = common::sized_zeroed();
        participant.deletes = &deleted;
        participant.delete_count = 1;
        participant
    });
    let mut generations = [u64::MAX; 2];
    let mut batch: ZeNamespaceBatchRequest = common::sized_zeroed();
    batch.root = path.as_ptr();
    batch.root_len = path.len();
    batch.participants = participants.as_ptr();
    batch.participant_count = participants.len();
    batch.generations = generations.as_mut_ptr();
    assert_unsupported(ze_namespace_batch(&batch), SENTINEL, "ze_namespace_batch");
    assert_eq!(generations, [u64::MAX; 2]);

    let (code, handle) = namespace_open(&path, b"fake990plain", &plain);
    assert_eq!(code, ZeErrorCode::ZeOk);
    assert_eq!(search(handle), before);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
    assert!(!root.path().join("fake990model").exists());
}

#[test]
fn portable_text_build_keeps_model_metadata_and_plain_namespaces() {
    let mut model: ZeTextModelRef = common::sized_zeroed();
    assert_eq!(ze_text_default_model(&mut model), ZeErrorCode::ZeOk);
    let field = |pointer: *const u8, len: usize| {
        // The export returns static UTF-8 slices of exactly len bytes.
        unsafe { std::slice::from_raw_parts(pointer, len) }
    };
    assert_eq!(
        field(model.repo, model.repo_len),
        volantis_db_text::DEFAULT_MODEL_REPO.as_bytes()
    );
    assert_eq!(
        field(model.revision, model.revision_len),
        volantis_db_text::DEFAULT_MODEL_REVISION.as_bytes()
    );
    assert_eq!(
        field(model.manifest_sha256, model.manifest_sha256_len),
        volantis_db_text::DEFAULT_MODEL_MANIFEST_SHA256.as_bytes()
    );

    let root = tempfile::tempdir().expect("temporary root");
    let path = bytes(root.path());
    let plain = plain_spec();
    let (code, handle) = namespace_open(&path, b"fake990plain", &plain);
    assert_eq!(code, ZeErrorCode::ZeOk);
    write_plain_row(handle);
    let written = search(handle);
    assert_eq!(written.1, 1);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
    let (code, handle) = namespace_open(&path, b"fake990plain", &plain);
    assert_eq!(code, ZeErrorCode::ZeOk);
    assert_eq!(search(handle), written);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
}
