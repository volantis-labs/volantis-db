mod common;

use std::mem::size_of;

use volantis_db_ffi::*;

fn open_settings() -> ZeOpenRequest {
    ZeOpenRequest {
        abi_size: size_of::<ZeOpenRequest>() as u32,
        abi_reserved: 0,
        path: std::ptr::null(),
        path_len: 0,
        access_mode: 0,
        durability_mode: 0,
        commit_tier: 1,
        reader_drain_timeout_ms: 250,
        max_resident_bytes: u64::MAX,
        max_temp_bytes: u64::MAX,
    }
}

fn canonical_epoch_identity(model_id: &[u8], dimensions: u32) -> ZeEpochIdentity {
    let version = b"1";
    let tower = ZeEmbeddingTower {
        model_id: model_id.as_ptr(),
        model_id_len: model_id.len(),
        model_version: version.as_ptr(),
        model_version_len: version.len(),
        weights_digest: std::ptr::null(),
        weights_digest_len: 0,
        dims: dimensions,
        normalization: 0,
        prompt_prefix: std::ptr::null(),
        prompt_prefix_len: 0,
        max_tokens: 0,
        runtime: 3,
        compute_units: 1,
        has_os_build: 0,
        os_build: std::ptr::null(),
        os_build_len: 0,
    };
    let request = ZeEpochRequest {
        abi_size: size_of::<ZeEpochRequest>() as u32,
        abi_reserved: 0,
        embedding: ZeEmbeddingEpoch {
            document: tower,
            query: tower,
            alignment_digest: std::ptr::null(),
            alignment_digest_len: 0,
        },
        tokenizer_profile: 0,
        reserved: 0,
    };
    let mut identity: ZeEpochIdentity = common::sized_zeroed();
    assert_eq!(
        ze_epoch_identity(&request, &mut identity),
        ZeErrorCode::ZeOk
    );
    identity
}

#[test]
fn namespace_create_reopen_identical_spec_is_usable_for_ingest_and_search() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"documents";
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let request = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &spec,
    };

    let mut handle = 0;
    assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
    assert_ne!(handle, 0);
    let expected = canonical_epoch_identity(b"zeppelin.vector-space", 4);
    let mut current: ZeEpochIdentity = common::sized_zeroed();
    assert_eq!(ze_epoch_current(handle, &mut current), ZeErrorCode::ZeOk);
    assert_eq!(current.embedding_epoch, expected.embedding_epoch);
    assert_eq!(current.tokenizer_epoch, expected.tokenizer_epoch);
    assert_eq!(common::ingest_rows(handle, 1, 4), ZeErrorCode::ZeOk);
    let query = vec![0.25_f32, 0.5, 0.75, 1.0];
    let search = common::valid_search_request(&query);
    let mut result: ZeSearchResult = common::sized_zeroed();
    assert_eq!(ze_search(handle, &search, &mut result), ZeErrorCode::ZeOk);
    assert_eq!(result.hit_count, 1);
    assert_eq!(ze_search_result_free(&mut result), ZeErrorCode::ZeOk);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);

    let mut reopened = 0;
    assert_eq!(
        ze_namespace_open(&request, &mut reopened),
        ZeErrorCode::ZeOk
    );
    assert_ne!(reopened, 0);
    assert_eq!(ze_close(reopened), ZeErrorCode::ZeOk);
}

#[test]
fn namespace_reopen_with_different_schema_returns_schema_mismatch() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"schema";
    let first_name = b"category";
    let first_attribute = ZeAttributeDefinition {
        attribute_id: 1,
        name: first_name.as_ptr(),
        name_len: first_name.len(),
        attribute_type: 5,
        nullable: 1,
    };
    let first_spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: &first_attribute,
        attribute_count: 1,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let mut request = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &first_spec,
    };
    let mut handle = 0;
    assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);

    let second_name = b"language";
    let second_attribute = ZeAttributeDefinition {
        name: second_name.as_ptr(),
        name_len: second_name.len(),
        ..first_attribute
    };
    let second_spec = ZeNamespaceSpec {
        attributes: &second_attribute,
        ..first_spec
    };
    request.spec = &second_spec;
    assert_eq!(
        ze_namespace_open(&request, &mut handle),
        ZeErrorCode::ZeErrSchemaMismatch
    );
}

#[test]
fn namespace_reopen_with_an_added_nullable_attribute_evolves_the_schema() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"evolving";
    let attribute = |attribute_id: u32, name: &'static [u8], nullable: u32| ZeAttributeDefinition {
        attribute_id,
        name: name.as_ptr(),
        name_len: name.len(),
        attribute_type: 5,
        nullable,
    };
    let release_one = [attribute(1, b"title", 1)];
    let release_two = [attribute(1, b"title", 1), attribute(2, b"language", 1)];
    let reordered = [attribute(2, b"language", 1), attribute(1, b"title", 1)];
    let added_required = [
        attribute(1, b"title", 1),
        attribute(2, b"language", 1),
        attribute(3, b"author", 0),
    ];
    let spec = |attributes: &[ZeAttributeDefinition]| ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: attributes.as_ptr(),
        attribute_count: attributes.len(),
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let open = |attributes: &[ZeAttributeDefinition], access_mode: i32| {
        let spec = spec(attributes);
        let request = ZeNamespaceOpenRequest {
            abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
            abi_reserved: 0,
            root: root_bytes.as_ptr(),
            root_len: root_bytes.len(),
            name: name.as_ptr(),
            name_len: name.len(),
            open: ZeOpenRequest {
                access_mode,
                ..open_settings()
            },
            spec: &spec,
        };
        let mut handle = 0;
        let code = ze_namespace_open(&request, &mut handle);
        if code == ZeErrorCode::ZeOk {
            assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
        }
        code
    };

    assert_eq!(open(&release_one, 0), ZeErrorCode::ZeOk);
    assert_eq!(
        open(&release_two, 1),
        ZeErrorCode::ZeErrSchemaMismatch,
        "a read-only open cannot persist an added attribute"
    );
    assert_eq!(open(&release_two, 0), ZeErrorCode::ZeOk);
    assert_eq!(open(&release_two, 1), ZeErrorCode::ZeOk);
    assert_eq!(open(&reordered, 0), ZeErrorCode::ZeOk);
    assert_eq!(
        open(&release_one, 0),
        ZeErrorCode::ZeErrSchemaMismatch,
        "an attribute cannot be removed"
    );
    assert_eq!(
        open(&added_required, 0),
        ZeErrorCode::ZeErrSchemaMismatch,
        "an added attribute must be nullable"
    );
}

#[test]
fn namespace_reopen_with_different_epoch_returns_epoch_mismatch() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"epochs";
    let first_epoch = common::EpochFixture::new(4);
    let first_epoch_request = first_epoch.request();
    let first_spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: &first_epoch_request,
    };
    let mut request = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &first_spec,
    };
    let mut handle = 0;
    assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);

    let mut second_epoch = common::EpochFixture::new(4);
    second_epoch.model_version = b"2".to_vec();
    let second_epoch_request = second_epoch.request();
    let second_spec = ZeNamespaceSpec {
        epoch: &second_epoch_request,
        ..first_spec
    };
    request.spec = &second_spec;
    assert_eq!(
        ze_namespace_open(&request, &mut handle),
        ZeErrorCode::ZeErrEpochMismatch
    );
}

#[test]
fn record_only_namespace_creates_reopens_and_reports_canonical_epoch() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"records";
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 0,
        dimensions: 0,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let request = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &spec,
    };
    let expected = canonical_epoch_identity(b"zeppelin.record-only", 1);
    let mut handle = 0;
    assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
    let mut current: ZeEpochIdentity = common::sized_zeroed();
    assert_eq!(ze_epoch_current(handle, &mut current), ZeErrorCode::ZeOk);
    assert_eq!(current.embedding_epoch, expected.embedding_epoch);
    assert_eq!(current.tokenizer_epoch, expected.tokenizer_epoch);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
    assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
}

#[test]
fn namespace_open_rejects_every_invalid_name() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let invalid_names = vec![
        Vec::new(),
        vec![b'a'; 256],
        b".leading-dot".to_vec(),
        b"-leading-dash".to_vec(),
        b"_leading-underscore".to_vec(),
        b"embedded/slash".to_vec(),
        b"..".to_vec(),
        b"embedded\0nul".to_vec(),
        "non-ascii-é".as_bytes().to_vec(),
    ];
    for name in invalid_names {
        let request = ZeNamespaceOpenRequest {
            abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
            abi_reserved: 0,
            root: root_bytes.as_ptr(),
            root_len: root_bytes.len(),
            name: name.as_ptr(),
            name_len: name.len(),
            open: open_settings(),
            spec: &spec,
        };
        let mut handle = 0;
        let code = ze_namespace_open(&request, &mut handle);
        if handle != 0 {
            assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
        }
        assert_eq!(
            code,
            ZeErrorCode::ZeErrInvalidArgument,
            "invalid namespace name {name:?}"
        );
    }
}

#[test]
fn namespace_list_returns_sorted_manifest_directories_and_empty_root() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let list_request = ZeNamespaceListRequest {
        abi_size: size_of::<ZeNamespaceListRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
    };
    let missing = root.path().join("missing");
    let missing_bytes = missing.to_string_lossy().into_owned().into_bytes();
    let missing_request = ZeNamespaceListRequest {
        root: missing_bytes.as_ptr(),
        root_len: missing_bytes.len(),
        ..list_request
    };
    let mut result: ZeNamespaceListResult = common::sized_zeroed();
    assert_eq!(
        ze_namespace_list(&missing_request, &mut result),
        ZeErrorCode::ZeErrIo
    );
    result = common::sized_zeroed();
    assert_eq!(
        ze_namespace_list(&list_request, &mut result),
        ZeErrorCode::ZeOk
    );
    assert_eq!(result.entry_count, 0);
    assert!(result.entries.is_null());
    assert_eq!(
        ze_namespace_list_result_free(&mut result),
        ZeErrorCode::ZeOk
    );

    std::fs::create_dir(root.path().join("not-a-namespace"))
        .expect("create non-namespace directory");
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    for name in [b"zeta".as_slice(), b"alpha".as_slice()] {
        let request = ZeNamespaceOpenRequest {
            abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
            abi_reserved: 0,
            root: root_bytes.as_ptr(),
            root_len: root_bytes.len(),
            name: name.as_ptr(),
            name_len: name.len(),
            open: open_settings(),
            spec: &spec,
        };
        let mut handle = 0;
        assert_eq!(ze_namespace_open(&request, &mut handle), ZeErrorCode::ZeOk);
        assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
    }

    result = common::sized_zeroed();
    assert_eq!(
        ze_namespace_list(&list_request, &mut result),
        ZeErrorCode::ZeOk
    );
    let entries = unsafe { std::slice::from_raw_parts(result.entries, result.entry_count) };
    let names = entries
        .iter()
        .map(|entry| unsafe { std::slice::from_raw_parts(entry.name, entry.name_len) })
        .collect::<Vec<_>>();
    assert_eq!(names, vec![b"alpha".as_slice(), b"zeta".as_slice()]);
    let mut forged = result;
    forged.entry_count += 1;
    assert_eq!(
        ze_namespace_list_result_free(&mut forged),
        ZeErrorCode::ZeErrInvalidArgument
    );
    assert_eq!(
        ze_namespace_list_result_free(&mut result),
        ZeErrorCode::ZeOk
    );
    assert_eq!(
        ze_namespace_list_result_free(&mut result),
        ZeErrorCode::ZeOk
    );
}

#[test]
fn namespace_requests_reject_null_zero_and_wrong_sizes_without_crashing() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"validation";
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 1,
        dimensions: 4,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let valid_open = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &spec,
    };
    let mut handle = 0;
    assert_eq!(
        ze_namespace_open(std::ptr::null(), &mut handle),
        ZeErrorCode::ZeErrInvalidArgument
    );
    assert_eq!(
        ze_namespace_open(&valid_open, std::ptr::null_mut()),
        ZeErrorCode::ZeErrInvalidArgument
    );
    for abi_size in [0, size_of::<ZeNamespaceOpenRequest>() as u32 - 1] {
        let request = ZeNamespaceOpenRequest {
            abi_size,
            ..valid_open
        };
        assert_eq!(
            ze_namespace_open(&request, &mut handle),
            ZeErrorCode::ZeErrInvalidArgument
        );
    }

    let valid_list = ZeNamespaceListRequest {
        abi_size: size_of::<ZeNamespaceListRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
    };
    let mut result: ZeNamespaceListResult = common::sized_zeroed();
    assert_eq!(
        ze_namespace_list(std::ptr::null(), &mut result),
        ZeErrorCode::ZeErrInvalidArgument
    );
    assert_eq!(
        ze_namespace_list(&valid_list, std::ptr::null_mut()),
        ZeErrorCode::ZeErrInvalidArgument
    );
    for abi_size in [0, size_of::<ZeNamespaceListRequest>() as u32 - 1] {
        let request = ZeNamespaceListRequest {
            abi_size,
            ..valid_list
        };
        result = common::sized_zeroed();
        assert_eq!(
            ze_namespace_list(&request, &mut result),
            ZeErrorCode::ZeErrInvalidArgument
        );
    }
}

#[test]
fn record_only_namespace_rejects_dimensions_above_one_and_a_caller_epoch() {
    let root = tempfile::tempdir().expect("temporary namespace root");
    let root_bytes = root.path().to_string_lossy().into_owned().into_bytes();
    let name = b"bad-records";
    let spec = ZeNamespaceSpec {
        abi_size: size_of::<ZeNamespaceSpec>() as u32,
        abi_reserved: 0,
        attributes: std::ptr::null(),
        attribute_count: 0,
        has_vector_space: 0,
        dimensions: 2,
        normalization: 0,
        epoch: std::ptr::null(),
    };
    let request = ZeNamespaceOpenRequest {
        abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
        abi_reserved: 0,
        root: root_bytes.as_ptr(),
        root_len: root_bytes.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        open: open_settings(),
        spec: &spec,
    };
    let mut handle = 0;
    let code = ze_namespace_open(&request, &mut handle);
    if handle != 0 {
        assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
        handle = 0;
    }
    assert_eq!(code, ZeErrorCode::ZeErrInvalidArgument);

    let epoch = common::EpochFixture::new(1);
    let epoch_request = epoch.request();
    let epoch_spec = ZeNamespaceSpec {
        dimensions: 1,
        epoch: &epoch_request,
        ..spec
    };
    let epoch_request = ZeNamespaceOpenRequest {
        spec: &epoch_spec,
        ..request
    };
    assert_eq!(
        ze_namespace_open(&epoch_request, &mut handle),
        ZeErrorCode::ZeErrInvalidArgument
    );
}

#[test]
fn namespace_tokenizer_profiles_persist_without_embedding_epoch() {
    use volantis_db::fts::tokenizer::Profile;
    let root = tempfile::tempdir().expect("root");
    let root_bytes = root.path().to_string_lossy();
    for (profile, config) in [
        (0, Profile::TextDefault),
        (1, Profile::Code),
        (2, Profile::Voice),
    ] {
        for vectors in [0, 1] {
            let name = format!("profile-{profile}-{vectors}");
            let mut spec: ZeNamespaceSpec = common::sized_zeroed();
            spec.has_vector_space = vectors;
            spec.dimensions = vectors * 4;
            let request = ZeNamespaceOpenRequest {
                abi_size: size_of::<ZeNamespaceOpenRequest>() as u32,
                abi_reserved: 0,
                root: root_bytes.as_ptr(),
                root_len: root_bytes.len(),
                name: name.as_ptr(),
                name_len: name.len(),
                open: open_settings(),
                spec: &spec,
            };
            for _ in 0..2 {
                let mut handle = 0;
                assert_eq!(
                    ze_namespace_open_with_tokenizer(&request, profile, &mut handle),
                    ZeErrorCode::ZeOk
                );
                let mut epoch: ZeEpochIdentity = common::sized_zeroed();
                assert_eq!(ze_epoch_current(handle, &mut epoch), ZeErrorCode::ZeOk);
                assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
                assert_eq!(epoch.tokenizer_epoch, config.config().epoch().value());
            }
            let mut handle = 0;
            assert_eq!(
                ze_namespace_open_with_tokenizer(&request, (profile + 1) % 3, &mut handle),
                ZeErrorCode::ZeErrEpochMismatch
            );
            assert_eq!(
                ze_namespace_open_with_tokenizer(&request, 99, &mut handle),
                ZeErrorCode::ZeErrInvalidArgument
            );
        }
    }
}

#[test]
fn embedded_cache_spec_checks_size_before_reading_limit() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().as_os_str().as_encoded_bytes();
    let open = |spec: &ZeNamespaceSpec| {
        let req = ZeNamespaceOpenRequest {
            root: path.as_ptr(),
            root_len: path.len(),
            name: b"toy-a".as_ptr(),
            name_len: 5,
            spec,
            open: open_settings(),
            ..common::sized_zeroed()
        };
        let mut handle = 0;
        let code = ze_namespace_open(&req, &mut handle);
        assert_eq!(handle, 0);
        code
    };
    // The allocation itself ends at the V1 boundary.
    let mut short: ZeEmbeddedNamespaceSpec = common::sized_zeroed();
    short.spec.has_vector_space = 3;
    assert_eq!(open(&short.spec), ZeErrorCode::ZeErrInvalidArgument);
    let mut full: ZeEmbeddedNamespaceSpecV2 = common::sized_zeroed();
    full.embedded.spec.has_vector_space = 3;
    full.embedded.spec.abi_reserved = 1;
    assert_eq!(open(&full.embedded.spec), ZeErrorCode::ZeErrInvalidArgument);
    full.embedded.spec.abi_reserved = 0;
    full.embedded.spec.has_vector_space = 4;
    assert_eq!(open(&full.embedded.spec), ZeErrorCode::ZeErrInvalidArgument);
    #[cfg(not(feature = "text"))]
    {
        let rev = b"0123456789abcdef0123456789abcdef01234567";
        let sha = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        full.embedded.model = ZeTextModelRef {
            repo: b"fixture/tiny".as_ptr(),
            repo_len: 12,
            revision: rev.as_ptr(),
            revision_len: rev.len(),
            manifest_sha256: sha.as_ptr(),
            manifest_sha256_len: sha.len(),
            ..common::sized_zeroed()
        };
        full.embedded.spec.has_vector_space = 3;
        assert_eq!(open(&full.embedded.spec), ZeErrorCode::ZeErrUnsupported);
        let mut legacy = full.embedded;
        legacy.spec.has_vector_space = 2;
        legacy.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpec>() as u32;
        assert_eq!(open(&legacy.spec), ZeErrorCode::ZeErrUnsupported);
    }
}

#[test]
fn wf989_v3_namespace_spec_validation() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let path = root.path().as_os_str().as_encoded_bytes();
    let cache_path = cache.path().as_os_str().as_encoded_bytes();
    let open = |spec: &ZeEmbeddedNamespaceSpecV3| {
        let req = ZeNamespaceOpenRequest {
            root: path.as_ptr(),
            root_len: path.len(),
            name: b"toy-a".as_ptr(),
            name_len: 5,
            spec: &spec.v2.embedded.spec,
            open: open_settings(),
            ..common::sized_zeroed()
        };
        let mut handle = 0;
        let code = ze_namespace_open(&req, &mut handle);
        assert_eq!(handle, 0);
        code
    };
    let rev = b"1111111111111111111111111111111111111111";
    let sha = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let mut valid: ZeEmbeddedNamespaceSpecV3 = common::sized_zeroed();
    valid.v2.embedded.spec.has_vector_space = 4;
    valid.v2.embedded.model = ZeTextModelRef {
        repo: b"fixture/fake-wf989".as_ptr(),
        repo_len: 18,
        revision: rev.as_ptr(),
        revision_len: rev.len(),
        manifest_sha256: sha.as_ptr(),
        manifest_sha256_len: sha.len(),
        ..common::sized_zeroed()
    };
    valid.model_cache_dir = cache_path.as_ptr();
    valid.model_cache_dir_len = cache_path.len();
    valid.offline = 1;
    // The well-formed control proves the policy reaches resolution: the empty
    // explicit root is an offline miss (text) or the build refuses it.
    let expected = if cfg!(feature = "text") {
        ZeErrorCode::ZeErrNotFound
    } else {
        ZeErrorCode::ZeErrUnsupported
    };
    assert_eq!(open(&valid), expected);
    assert!(std::fs::read_dir(cache.path()).unwrap().next().is_none());

    let mut truncated = valid;
    truncated.v2.embedded.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpecV2>() as u32;
    let invalid_utf8 = [b'/', 0xff];
    let interior_nul = [b'/', 0, b'x'];
    let cases: [(&str, ZeEmbeddedNamespaceSpecV3); 6] = [
        ("truncated tag 4", truncated),
        (
            "offline 2",
            ZeEmbeddedNamespaceSpecV3 {
                offline: 2,
                ..valid
            },
        ),
        (
            "has_mlx_cache_limit 2",
            ZeEmbeddedNamespaceSpecV3 {
                has_mlx_cache_limit: 2,
                ..valid
            },
        ),
        (
            "invalid UTF-8",
            ZeEmbeddedNamespaceSpecV3 {
                model_cache_dir: invalid_utf8.as_ptr(),
                model_cache_dir_len: invalid_utf8.len(),
                ..valid
            },
        ),
        (
            "interior NUL",
            ZeEmbeddedNamespaceSpecV3 {
                model_cache_dir: interior_nul.as_ptr(),
                model_cache_dir_len: interior_nul.len(),
                ..valid
            },
        ),
        (
            "null with length",
            ZeEmbeddedNamespaceSpecV3 {
                model_cache_dir: std::ptr::null(),
                model_cache_dir_len: 1,
                ..valid
            },
        ),
    ];
    for (label, spec) in cases {
        assert_eq!(open(&spec), ZeErrorCode::ZeErrInvalidArgument, "{label}");
    }
}

#[test]
fn wf985_deferred_abi_contract() {
    let root = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let path = root.path().as_os_str().as_encoded_bytes();
    let cache_path = cache.path().as_os_str().as_encoded_bytes();
    let open = |spec: &ZeEmbeddedNamespaceSpecV4| {
        let req = ZeNamespaceOpenRequest {
            root: path.as_ptr(),
            root_len: path.len(),
            name: b"toy-a".as_ptr(),
            name_len: 5,
            spec: &spec.v3.v2.embedded.spec,
            open: open_settings(),
            ..common::sized_zeroed()
        };
        let mut handle = 0;
        let code = ze_namespace_open(&req, &mut handle);
        assert_eq!(handle, 0);
        code
    };
    assert_eq!(size_of::<ZeEmbeddedNamespaceSpecV4>(), 144);
    assert_eq!(size_of::<ZeEmbeddingStatus>(), 32);
    assert_eq!(size_of::<ZeEmbeddedReport>(), 16);
    let rev = b"1111111111111111111111111111111111111111";
    let sha = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let mut valid: ZeEmbeddedNamespaceSpecV4 = common::sized_zeroed();
    valid.v3.v2.embedded.spec.has_vector_space = 5;
    valid.v3.v2.embedded.model = ZeTextModelRef {
        repo: b"fixture/fake-wf985".as_ptr(),
        repo_len: 18,
        revision: rev.as_ptr(),
        revision_len: rev.len(),
        manifest_sha256: sha.as_ptr(),
        manifest_sha256_len: sha.len(),
        ..common::sized_zeroed()
    };
    valid.v3.model_cache_dir = cache_path.as_ptr();
    valid.v3.model_cache_dir_len = cache_path.len();
    valid.v3.offline = 1;
    valid.document_schedule = 2;
    // The well-formed control reaches model resolution: an offline miss in a
    // text build, a refusal in a build without text.
    let expected = if cfg!(feature = "text") {
        ZeErrorCode::ZeErrNotFound
    } else {
        ZeErrorCode::ZeErrUnsupported
    };
    assert_eq!(open(&valid), expected);
    for schedule in [0, 1] {
        assert_eq!(
            open(&ZeEmbeddedNamespaceSpecV4 {
                document_schedule: schedule,
                ..valid
            }),
            expected,
            "schedule {schedule}"
        );
    }
    let mut truncated = valid;
    truncated.v3.v2.embedded.spec.abi_size = size_of::<ZeEmbeddedNamespaceSpecV3>() as u32;
    let cases = [
        ("truncated tag 5", truncated),
        (
            "unknown schedule",
            ZeEmbeddedNamespaceSpecV4 {
                document_schedule: 3,
                ..valid
            },
        ),
        (
            "reserved",
            ZeEmbeddedNamespaceSpecV4 {
                reserved: 1,
                ..valid
            },
        ),
    ];
    for (label, spec) in cases {
        assert_eq!(open(&spec), ZeErrorCode::ZeErrInvalidArgument, "{label}");
    }

    // Control calls on a store without a model fail typed.
    let (code, handle) = common::open_path(&root.path().join("plain"));
    assert_eq!(code, ZeErrorCode::ZeOk);
    let mut status: ZeEmbeddingStatus = common::sized_zeroed();
    for code in [
        ze_namespace_embedding_pause(handle),
        ze_namespace_embedding_resume(handle),
        ze_namespace_embedding_status(handle, &mut status),
    ] {
        assert_ne!(code, ZeErrorCode::ZeOk);
        assert_ne!(code, ZeErrorCode::ZeErrPanic);
    }
    assert_eq!(ze_close(handle), ZeErrorCode::ZeOk);
}
