use std::path::Path;
pub fn pinned_fixture(root: &Path) -> volantis_db_text::ModelRef {
    let model = volantis_db_text::ModelRef {
        repo: "fixture/fake-ze598".into(),
        revision: "1".repeat(40),
        manifest_sha256: "ea8cef12ec40e1bdd8f34d3dfce45bfc212026a86af00576a1adb92e71309306".into(),
    };
    let entry = root.join("fixture%2Ffake-ze598").join(&model.revision);
    std::fs::create_dir_all(&entry).expect("cache root");
    std::fs::write(entry.join("volantis-model.json"),br#"{"format":1,"name":"tiny","bundle":"tiny.zem","query_coreml":null,"files":[{"path":"tiny.zem","size":9132,"sha256":"548472279832e8ebe6eb57129bb038cc2162ecc4bd5584d07bc5603c97c6fac3"}]}"#).expect("manifest");
    std::fs::write(
        entry.join("tiny.zem"),
        include_bytes!("../fixtures/ze115/gte.zem"),
    )
    .expect("bundle");
    model
}
