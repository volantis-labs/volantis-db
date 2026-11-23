use std::path::Path;
pub fn pinned_fixture(root: &Path) -> volantis_db_text::ModelRef {
    pinned_fixture_for(root, "fixture/fake-ze598", &"1".repeat(40))
}

pub fn pinned_fixture_for(root: &Path, repo: &str, revision: &str) -> volantis_db_text::ModelRef {
    let model = volantis_db_text::ModelRef {
        repo: repo.into(),
        revision: revision.into(),
        manifest_sha256: "ea8cef12ec40e1bdd8f34d3dfce45bfc212026a86af00576a1adb92e71309306".into(),
    };
    let entry = root.join(repo.replace('/', "%2F")).join(&model.revision);
    std::fs::create_dir_all(&entry).expect("cache root");
    std::fs::write(entry.join("volantis-model.json"),br#"{"format":1,"name":"tiny","bundle":"tiny.zem","query_coreml":null,"files":[{"path":"tiny.zem","size":9132,"sha256":"548472279832e8ebe6eb57129bb038cc2162ecc4bd5584d07bc5603c97c6fac3"}]}"#).expect("manifest");
    std::fs::write(
        entry.join("tiny.zem"),
        include_bytes!("../fixtures/ze115/gte.zem"),
    )
    .expect("bundle");
    model
}
