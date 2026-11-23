//! Compares the CoreML query tower against MLX for agreement and speed.
//!
//! MLX exists on macOS arm64 only. Other targets build a stub that fails.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "apple/coreml_placement.rs"]
mod apple;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    apple::main();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("coreml_placement needs the MLX runtime, which exists on macOS arm64 only");
    std::process::exit(2);
}
