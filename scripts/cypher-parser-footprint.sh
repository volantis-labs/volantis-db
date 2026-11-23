#!/usr/bin/env bash
# Current reachable frontend/core/FFI footprint. Not graph release size acceptance.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != Darwin ]]; then
    echo "error: this measurement is defined for macOS" >&2
    exit 2
fi
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
if [[ "$TARGET" != /* ]]; then TARGET="$ROOT/$TARGET"; fi
MEASURE="$TARGET/cypher-parser-footprint"
mkdir -p "$MEASURE"
cargo build --locked --release -p volantis-db-cypher -p volantis-db-ffi
C_SOURCE=crates/volantis-db-cypher/tests/fixtures/footprint_consumer.c
SHIM=crates/volantis-db-cypher/tests/fixtures/footprint_shim.rs
for variant in baseline frontend; do
    RUST_EXTRA=()
    C_EXTRA=()
    if [[ "$variant" == frontend ]]; then RUST_EXTRA=(--cfg frontend); C_EXTRA=(-DZE54_WITH_FRONTEND); fi
    # One Rust link composes rlibs so the runtime is present exactly once.
    rustc --edition 2024 --crate-type staticlib --crate-name ze54_combined_probe \
        -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=unwind \
        --extern "volantis_db_cypher=$TARGET/release/libvolantis_db_cypher.rlib" \
        --extern "volantis_db_ffi=$TARGET/release/libvolantis_db_ffi.rlib" \
        -L "dependency=$TARGET/release/deps" ${RUST_EXTRA[@]+"${RUST_EXTRA[@]}"} \
        "$SHIM" -o "$MEASURE/lib$variant.a"
    clang -O3 -I crates/volantis-db-ffi/include "$C_SOURCE" ${C_EXTRA[@]+"${C_EXTRA[@]}"} \
        "$MEASURE/lib$variant.a" -framework Security -framework CoreFoundation \
        -lpthread -ldl -lm -o "$MEASURE/$variant"
    strip -S -x "$MEASURE/$variant"
    STORE_DIR="$(mktemp -d "$MEASURE/store-$variant.XXXXXX")"
    "$MEASURE/$variant" "$STORE_DIR" 'MATCH (n:A {x: $x})-[r:R*1..3]->(m) RETURN n, collect(m) AS ms'
    rm -rf "$STORE_DIR"
done
rustc --edition 2024 --crate-type staticlib --crate-name ze54_parser_probe \
    -C opt-level=3 -C lto=fat -C codegen-units=1 -C panic=unwind --cfg frontend --cfg standalone \
    --extern "volantis_db_cypher=$TARGET/release/libvolantis_db_cypher.rlib" \
    -L "dependency=$TARGET/release/deps" "$SHIM" -o "$MEASURE/parser-stripped.a"
strip -S -x "$MEASURE/parser-stripped.a"
for variant in baseline frontend; do
    cp "$MEASURE/lib$variant.a" "$MEASURE/$variant-stripped.a"
    strip -S -x "$MEASURE/$variant-stripped.a"
done
for artifact in baseline frontend parser-stripped.a baseline-stripped.a frontend-stripped.a; do
    LINKABLE_BYTES="$(size -m "$MEASURE/$artifact" | awk '/^[[:space:]]*Section / && $0 !~ /\(__LLVM,/ { bytes = ($NF == "(zerofill)") ? $(NF-1) : $NF; total += bytes } END { print total + 0 }')"
    PHYSICAL_KIB="$(du -k "$MEASURE/$artifact" | awk '{ print $1 }')"
    echo "ZE54_SIZE artifact=$artifact linkable_bytes=$LINKABLE_BYTES physical_KiB=$PHYSICAL_KIB"
done
