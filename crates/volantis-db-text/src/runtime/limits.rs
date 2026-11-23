//! Evaluation batch limits shared by every runtime backend.

pub(crate) const MAX_EVAL_ROWS: usize = 32;
// ZE-525 native FP16 probes on M3 Max under GPU contention: at 4,096 tokens,
// warm evaluations took up to ~0.53 s; a cold 8 x 512 evaluation took ~2.96 s.
// Throughput/token was flat above this budget, with contention-dependent
// timeouts at larger shapes. An over-budget row must run alone.
pub(crate) const MAX_EVAL_TOKENS: usize = 4_096;

pub(crate) fn fits_eval(rows: usize, width: usize) -> bool {
    rows == 1 || (rows <= MAX_EVAL_ROWS && rows <= MAX_EVAL_TOKENS / width.max(1))
}

/// Test observation of evaluated batch shapes: (rows, row widths).
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
pub(crate) type EvalObserver = std::sync::Arc<std::sync::Mutex<Vec<(usize, Vec<usize>)>>>;
