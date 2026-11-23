//! Evaluation batch limits shared by every runtime backend.

use crate::runtime::RuntimeError;
use crate::tower::TokenBatch;

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
#[cfg(test)]
pub(crate) type EvalObserver = std::sync::Arc<std::sync::Mutex<Vec<(usize, Vec<usize>)>>>;

/// Greedily retain caller order, padding only to this evaluation's longest row.
pub(crate) fn next_eval_chunk(
    tokens: &TokenBatch,
    start: usize,
) -> Result<TokenBatch, RuntimeError> {
    let expected = tokens
        .rows
        .checked_mul(tokens.tokens_per_row)
        .ok_or_else(|| RuntimeError::Shape("token batch shape overflow".to_owned()))?;
    if tokens.tokens_per_row == 0 || start >= tokens.rows || tokens.token_ids.len() != expected {
        return Err(RuntimeError::Shape(
            "token batch chunk is truncated".to_owned(),
        ));
    }
    if tokens.attention_mask.len() != expected {
        return Err(RuntimeError::Shape(
            "attention chunk is truncated".to_owned(),
        ));
    }
    let mut rows = 0;
    let mut width = 1;
    for mask in tokens
        .attention_mask
        .chunks_exact(tokens.tokens_per_row)
        .skip(start)
        .take(MAX_EVAL_ROWS)
    {
        // Keep masked holes and token positions; remove only trailing padding.
        let row_width = mask
            .iter()
            .rposition(|value| *value != 0.0)
            .map_or(1, |position| position + 1);
        let next_width = width.max(row_width);
        if !fits_eval(rows + 1, next_width) {
            break;
        }
        rows += 1;
        width = next_width;
    }
    let mut token_ids = Vec::with_capacity(rows * width);
    let mut attention_mask = Vec::with_capacity(rows * width);
    for (ids, mask) in tokens
        .token_ids
        .chunks_exact(tokens.tokens_per_row)
        .zip(tokens.attention_mask.chunks_exact(tokens.tokens_per_row))
        .skip(start)
        .take(rows)
    {
        token_ids.extend(ids.iter().take(width).copied());
        attention_mask.extend(mask.iter().take(width).copied());
    }
    TokenBatch::new(token_ids, attention_mask, rows, width)
        .map_err(|detail| RuntimeError::Shape(detail.to_owned()))
}
