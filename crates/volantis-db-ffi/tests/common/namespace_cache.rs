//! Isolated offline model fixture for process-wide MLX policy tests.
use super::*;
use std::process::Command;

const REVISION: &[u8] = b"0123456789abcdef0123456789abcdef01234567";

fn digest(path: &Path) -> String {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned()
}

pub fn isolated(name: &str, ignored: bool, run: impl FnOnce(&[u8], ZeEmbeddedNamespaceSpecV2)) {
    if let Some(root) = std::env::var_os("ZE599_FIXTURE_ROOT") {
        let root = PathBuf::from(root);
        let sha = std::env::var("ZE599_MANIFEST").unwrap();
        let mut v2: ZeEmbeddedNamespaceSpecV2 = sized_zeroed();
        v2.embedded.spec.has_vector_space = 3;
        let repo = b"fixture/tiny";
        v2.embedded.model = ZeTextModelRef {
            repo: repo.as_ptr(),
            repo_len: repo.len(),
            revision: REVISION.as_ptr(),
            revision_len: REVISION.len(),
            manifest_sha256: sha.as_ptr(),
            manifest_sha256_len: sha.len(),
            ..sized_zeroed()
        };
        v2.mlx_cache_limit_bytes = 536870912;
        run(root.as_os_str().as_encoded_bytes(), v2);
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let cache = root.path().join("cache");
    let entry = cache
        .join("fixture%2Ftiny")
        .join(std::str::from_utf8(REVISION).unwrap());
    std::fs::create_dir_all(&entry).unwrap();
    let bundle = entry.join("tiny.zem");
    let len = std::fs::copy(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../volantis-db-text/tests/fixtures/ze115/gte.zem"
        ),
        &bundle,
    )
    .unwrap();
    let hash = digest(&bundle);
    let manifest = entry.join("volantis-model.json");
    std::fs::write(&manifest, format!(r#"{{"format":1,"name":"tiny","bundle":"tiny.zem","query_coreml":null,"files":[{{"path":"tiny.zem","size":{len},"sha256":"{hash}"}}]}}"#)).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap());
    if ignored {
        child.arg("--ignored");
    }
    assert!(
        child
            .args(["--exact", name, "--nocapture"])
            .env("ZE599_FIXTURE_ROOT", root.path())
            .env("ZE599_MANIFEST", digest(&manifest))
            .env("ZE_MODEL_CACHE", cache)
            .env("ZE_MODEL_OFFLINE", "1")
            .status()
            .unwrap()
            .success()
    );
}

pub fn open(root: &[u8], name: &[u8], spec: &ZeNamespaceSpec) -> (ZeErrorCode, ZeHandle) {
    let request = ZeNamespaceOpenRequest {
        root: root.as_ptr(),
        root_len: root.len(),
        name: name.as_ptr(),
        name_len: name.len(),
        spec,
        open: ZeOpenRequest {
            commit_tier: 1,
            max_resident_bytes: u64::MAX,
            max_temp_bytes: u64::MAX,
            ..sized_zeroed()
        },
        ..sized_zeroed()
    };
    let mut handle = 0;
    (ze_namespace_open(&request, &mut handle), handle)
}

pub fn row(id: u64) -> ZeUpsertDocument {
    let text = b"bronze zeppelin";
    ZeUpsertDocument {
        document: ZeIngestDocument {
            doc_id: ZeDocId { high: 0, low: id },
            revision: 1,
            text: text.as_ptr(),
            text_len: text.len(),
            ..sized_zeroed()
        },
        ..sized_zeroed()
    }
}
