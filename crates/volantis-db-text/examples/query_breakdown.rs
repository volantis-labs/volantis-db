//! Runs the query_breakdown MLX benchmark.
//!
//! MLX exists on macOS arm64 only. Other targets build a stub that fails.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "apple/query_breakdown.rs"]
mod apple;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    apple::main();
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("query_breakdown needs the MLX runtime, which exists on macOS arm64 only");
    std::process::exit(2);
}
