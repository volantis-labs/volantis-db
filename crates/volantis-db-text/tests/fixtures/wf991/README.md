# WF-991 tiny FP16 BERT pair

`generate.py` writes `pair-f16.zem` (74,320 bytes) and `reference.rs`. The
bundle is layout 3 with two invented BERT towers that share one WordPiece
record: document `FAKE-WF991-D` (2 layers) and query `FAKE-WF991-Q`
(1 layer, prefix `query: `). Both have hidden width 32, 2 heads (head width
16), intermediate width 64, 16 positions, 2 token types, 32 vocabulary rows,
LayerNorm epsilon 1e-12 and mean pooling. The widths are larger than the plan
minimum so the forward pass crosses NEON (16) and AVX (32) block boundaries.

Weights come from a fixed integer formula, `k / 61` with `-48 <= k <= 48`
(norm gains `1 + k / 244`). Most of these values are not exact in binary16.
Each value is rounded once to IEEE binary16 (round to nearest, ties to even)
and stored as little-endian F16, so the runtime must widen real halves.

The oracle is an independent float64 PyTorch CPU forward pass over the
rounded weights (PyTorch layer norm, scaled dot-product attention with the
additive `(mask - 1) * 1e9` bias, exact GELU). It records the raw masked-mean
rows before L2 for two ragged rows, "copper kite 991" `[2, 4, 5, 6, 3]` and
"violet spool" `[2, 7, 8, 3, 0]` with the last position masked. No value is
copied from the Rust or MLX runtimes.

Generation used Python 3.9.6, torch 2.6.0, xxhash 3.5.0 and rustfmt 1.8.0
(Rust 1.93.0). These are offline fixture tools, not workspace dependencies.
Run `python3 generate.py`. The tests read the committed bytes only.
