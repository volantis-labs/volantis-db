fn main() {
    println!("cargo:rerun-if-changed=objc/ze_coreml.m");
    println!("cargo:rerun-if-changed=objc/ze_model.m");
    // Cross builds need the Cargo target, not the build host.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // Model download and verification serve every macOS target.
    cc::Build::new()
        .file("objc/ze_model.m")
        .flag("-Wno-deprecated-declarations")
        // No linker-synthesized __objc_stubs: ld lays them out in a different
        // order between runs, which breaks the pinned framework checksum.
        .flag("-fno-objc-msgsend-selector-stubs")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .compile("ze_model");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // The Core ML query runtime pairs with MLX on Apple silicon only.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64") {
        return;
    }
    cc::Build::new()
        .file("objc/ze_coreml.m")
        .flag("-Wno-deprecated-declarations")
        // No linker-synthesized __objc_stubs: ld lays them out in a different
        // order between runs, which breaks the pinned framework checksum.
        .flag("-fno-objc-msgsend-selector-stubs")
        .flag("-fobjc-arc")
        .flag("-fmodules")
        .compile("ze_coreml");
    println!("cargo:rustc-link-lib=framework=CoreML");
}
