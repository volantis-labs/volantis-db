//! Runs the placement_bench MLX benchmark.
//!
//! MLX exists on macOS arm64 only. Other targets build a stub that fails.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "apple/placement_bench.rs"]
mod apple;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    apple::main();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("placement_bench needs the MLX runtime, which exists on macOS arm64 only");
    std::process::exit(2);
}
