#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Owner decision 2026-10-05: graph-free core/FFI budget is 5,632 KB
# on every platform, including arm64 and Intel.
# WF-992, owner decision 2026-11-23: raised to 6,144 KB on every platform.
# The legacy XCFramework macOS x86_64 slice measured 5,696 KB after E24.
BUDGET_KB="${ZE_SIZE_BUDGET_KB:-6144}"
# Owner decision 2026-09-27 (CLAUDE.md, ZE-253): complete graph FFI
# archive (core + FFI + Cypher) is gated at 12,288 KB.
# WF-992, owner decision 2026-11-23: raised to 13,312 KB on every platform.
# The Windows x64 graph archive measured 12,478 KB after the E24 engine work.
GRAPH_BUDGET_KB="${ZE_GRAPH_SIZE_BUDGET_KB:-13312}"
# ZE-506, orchestrator decision 2026-10-19 (flagged to the owner for veto):
# the arm64 full graph+text FFI archive is gated at 28,672 KB.
TEXT_BUDGET_KB=28672
# WF-992, owner decision 2026-11-23: the macOS x86_64 graph+text FFI archive
# (pure-Rust CPU runtime, no MLX) is gated at 14,336 KB. WF-991 measured
# 12,474 KB. The graph-only gate (13,312 KB) and the arm64 28,672 KB gate are
# separate.
X64_TEXT_BUDGET_KB=14336

if [[ ! "$BUDGET_KB" =~ ^[0-9]+$ ]]; then
    echo "error: ZE_SIZE_BUDGET_KB must be a non-negative integer, got '$BUDGET_KB'" >&2
    exit 2
fi

if [[ ! "$GRAPH_BUDGET_KB" =~ ^[0-9]+$ ]]; then
    echo "error: ZE_GRAPH_SIZE_BUDGET_KB must be a non-negative integer" >&2
    exit 2
fi

cd "$PROJECT_ROOT"

TARGET_ROOT="${CARGO_TARGET_DIR:-$PROJECT_ROOT/target}"
MEASURE_DIR="$TARGET_ROOT/size-budget"

STRIP_TOOL=strip
SIZE_TOOL=size
case "$(uname -s)" in
    MINGW*|MSYS*) STRIP_TOOL=llvm-strip; SIZE_TOOL=llvm-size ;;
esac
if ! command -v "$STRIP_TOOL" >/dev/null 2>&1; then
    echo "error: platform 'strip' tool is required for the size gate" >&2
    exit 2
fi
if ! command -v "$SIZE_TOOL" >/dev/null 2>&1; then
    echo "error: platform 'size' tool is required for the size gate" >&2
    exit 2
fi

mkdir -p "$MEASURE_DIR"

# Measures one static library's post-strip linkable sections. Task 22 (D9)
# added the FFI staticlib as a second measured artifact under the same bar:
# adding an artifact strengthens the gate, and the core measurement is never
# traded away for it. Moving either bar itself remains an owner call.
measure_artifact() {
    local label="$1"
    local artifact="$2"
    local gate="${3:-gate}"
    local budget_kb="${4:-$BUDGET_KB}"
    local stripped="$MEASURE_DIR/${label// /-}-$(basename "${artifact%.a}")-stripped.a"
    local size_bytes size_kb archive_kb physical_bytes
    local raw_size="$MEASURE_DIR/${label// /-}-$(basename "$artifact")-size.txt"

    if [[ ! -f "$artifact" ]]; then
        echo "error: expected static library not found at $artifact" >&2
        exit 2
    fi

    cp "$artifact" "$stripped"
    case "$(uname -s)" in
        Darwin)
            strip -S -x "$stripped"
            size -m "$stripped" > "$raw_size"
            size_bytes="$(awk '
                /^[[:space:]]*Section \(/ && $0 !~ /\(__LLVM,/ { total += ($NF == "(zerofill)") ? $(NF - 1) : $NF }
                END { print total + 0 }
            ' "$raw_size")"
            ;;
        Linux)
            strip --strip-unneeded "$stripped"
            size -A "$stripped" > "$raw_size"
            size_bytes="$(awk '
                $1 ~ /^\./ && $1 !~ /^\.llvm/ { total += $2 }
                END { print total + 0 }
            ' "$raw_size")"
            ;;
        MINGW*|MSYS*)
            python3 "$SCRIPT_DIR/release/package-native.py" measure-coff "$artifact" "$stripped"
            "$SIZE_TOOL" -A "$stripped" > "$raw_size"
            size_bytes="$(awk '
                $1 ~ /^\./ && $1 !~ /^\.llvm/ { total += $2 }
                END { print total + 0 }
            ' "$raw_size")"
            ;;
        *)
            echo "error: size-budget.sh supports Darwin, Linux and Windows (LLVM)" >&2
            exit 2
            ;;
    esac

    if (( size_bytes <= 0 )); then
        echo "error: no linkable sections found in $artifact" >&2
        exit 2
    fi
    if [[ "$(uname -s)" == Darwin ]]; then
        physical_bytes="$(stat -f %z "$artifact")"
    else
        physical_bytes="$(stat -c %s "$artifact")"
    fi
    echo "$label linked_section_bytes=$size_bytes physical_archive_bytes=$physical_bytes"
    size_kb="$(( (size_bytes + 1023) / 1024 ))"
    archive_kb="$(du -k "$stripped" | awk '{print $1}')"
    if [[ "$gate" == "gate" ]]; then
        echo "$label stripped staticlib linked size: $size_kb KB (archive: $archive_kb KB; budget: $budget_kb KB)"
    else
        echo "$label stripped staticlib linked size: $size_kb KB (archive: $archive_kb KB; recorded only; no budget introduced)"
    fi

    # Tooling receipt: raw platform output plus exact original bytes. A measured
    # archive is not installed reachability or final release qualification.
    python3 - "$artifact" "$raw_size" "$label" "$size_bytes" "$physical_bytes" "$budget_kb" "$gate" <<'PY_RECEIPT'
import hashlib
import json
from pathlib import Path
import sys
artifact, raw, label, sections, physical, budget, gate = sys.argv[1:]
source = Path(artifact)
report = dict(artifact=str(source.resolve()), label=label, section_bytes=int(sections),
              physical_bytes=int(physical), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              raw_size=str(Path(raw).resolve()), raw_sha256=hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
              section_bytes_max=int(budget)*1024 if gate == 'gate' else None,
              state='measured', exit_status=0 if gate != 'gate' or int(sections) <= int(budget)*1024 else 1)
Path(raw + '.json').write_text(json.dumps(report, indent=2) + '\n')
PY_RECEIPT

    if [[ "$gate" == "gate" ]] && (( size_kb > budget_kb )); then
        echo "error: $label stripped staticlib linked size $size_kb KB exceeds budget $budget_kb KB" >&2
        exit 1
    fi
}

# CI can gate the exact prebuilt archive without rebuilding or changing features.
case "${1:-}" in
    --graph-archive|--text-graph-archive|--x64-text-graph-archive|--graph-core-archive|--ffi-archive|--core-archive)
        if [[ $# -ne 2 ]]; then
            echo "usage: $0 [--graph-archive|--text-graph-archive|--x64-text-graph-archive|--graph-core-archive|--ffi-archive|--core-archive ARCHIVE]" >&2
            exit 2
        fi
        if [[ "$1" == "--graph-archive" ]]; then
            measure_artifact "graph ffi" "$2" gate "$GRAPH_BUDGET_KB"
        elif [[ "$1" == "--text-graph-archive" ]]; then
            measure_artifact "text graph ffi" "$2" gate "$TEXT_BUDGET_KB"
        elif [[ "$1" == "--x64-text-graph-archive" ]]; then
            measure_artifact "x64 text graph ffi" "$2" gate "$X64_TEXT_BUDGET_KB"
        elif [[ "$1" == "--graph-core-archive" ]]; then
            measure_artifact "graph core" "$2" report
        else
            measure_artifact "${1#--}" "$2"
        fi
        exit 0
        ;;
    "") ;;
    *) echo "error: unknown size gate option: $1" >&2; exit 2 ;;
esac

cargo build --release -p volantis-db -p volantis-db-ffi

measure_artifact "core" "$TARGET_ROOT/release/libvolantis_db.a"
measure_artifact "ffi" "$TARGET_ROOT/release/libvolantis_db_ffi.a"

# Graph uses a separate output directory, preserving the graph-free archive.
if [[ "$(uname -s)" == "Darwin" ]]; then
    cargo build --locked --release -p volantis-db -p volantis-db-ffi --features graph-cypher \
        --target-dir "$TARGET_ROOT/size-budget-graph"
    measure_artifact "graph core" "$TARGET_ROOT/size-budget-graph/release/libvolantis_db.a" report
    measure_artifact "graph ffi" \
        "$TARGET_ROOT/size-budget-graph/release/libvolantis_db_ffi.a" gate "$GRAPH_BUDGET_KB"
fi

# Text ships on macOS only: MLX on arm64, the CPU runtime on x86_64 (WF-991).
# Non-macOS paths remain text-free. Preserve the separate graph-free core/FFI
# and graph-only measurements above.
if [[ "$(uname -s)" == Darwin ]]; then
    case "$(uname -m)" in
        arm64) text_label="full text ffi"; text_budget_kb="$TEXT_BUDGET_KB" ;;
        x86_64) text_label="x64 text graph ffi"; text_budget_kb="$X64_TEXT_BUDGET_KB" ;;
        *) echo "error: no text size gate for macOS $(uname -m)" >&2; exit 2 ;;
    esac
    cargo build --locked --release -p volantis-db-ffi --features graph-cypher,text \
        --target-dir "$TARGET_ROOT/size-budget-text"
    measure_artifact "$text_label" \
        "$TARGET_ROOT/size-budget-text/release/libvolantis_db_ffi.a" gate "$text_budget_kb"
fi

CONSUMER_MANIFEST="$PROJECT_ROOT/tools/size-consumer/Cargo.toml"
CONSUMER_TARGET="$MEASURE_DIR/consumer-target"
cargo build --offline --locked --release --manifest-path "$CONSUMER_MANIFEST" --target-dir "$CONSUMER_TARGET"
CONSUMER_BINARY="$CONSUMER_TARGET/release/volantis-db-size-consumer"
CONSUMER_STRIPPED="$MEASURE_DIR/volantis-db-size-consumer-stripped"
if [[ ! -f "$CONSUMER_BINARY" ]]; then
    echo "error: expected minimal consumer binary not found at $CONSUMER_BINARY" >&2
    exit 2
fi
cp "$CONSUMER_BINARY" "$CONSUMER_STRIPPED"
case "$(uname -s)" in
    Darwin)
        strip -S -x "$CONSUMER_STRIPPED"
        CONSUMER_SIZE_BYTES="$(size -m "$CONSUMER_STRIPPED" | awk '
            /^[[:space:]]*Section / && $0 !~ /\(__LLVM,/ {
                bytes = ($NF == "(zerofill)") ? $(NF - 1) : $NF
                total += bytes
            }
            END { print total + 0 }
        ')"
        ;;
    Linux)
        strip --strip-unneeded "$CONSUMER_STRIPPED"
        CONSUMER_SIZE_BYTES="$(size -A "$CONSUMER_STRIPPED" | awk '
            $1 ~ /^\./ && $1 !~ /^\.llvm/ { total += $2 }
            END { print total + 0 }
        ')"
        ;;
esac
CONSUMER_SIZE_KB="$(( (CONSUMER_SIZE_BYTES + 1023) / 1024 ))"
CONSUMER_FILE_KB="$(du -k "$CONSUMER_STRIPPED" | awk '{print $1}')"
echo "minimal volantis-db consumer stripped linked size: $CONSUMER_SIZE_KB KB (file: $CONSUMER_FILE_KB KB; reported only; no budget introduced)"
