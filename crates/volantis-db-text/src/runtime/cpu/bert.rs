//! fp32 BERT forward pass in the order of `arch/bert.rs`.
//!
//! Post-norm encoder: word rows plus positions `0..seq` plus token type 0,
//! LayerNorm; per layer separate Q/K/V, masked scaled dot-product attention,
//! output projection, residual and LayerNorm, then intermediate, erf GELU,
//! output, residual and LayerNorm. Then masked pooling, the optional dense head
//! and truncation to the declared dimensions. L2 is applied by the caller.

use crate::bundle::Bundle;
use crate::runtime::RuntimeError;
use crate::runtime::cpu::kernels::{Kernels, TILE_FLOATS, WeightView, linear};
use crate::runtime::cpu::{CpuWeights, Linear, Norm, Tensor};
use crate::tower::{Pooling, TokenBatch, TowerSpec};

/// Below this many multiply-adds a product runs on the caller alone.
pub(crate) const PARALLEL_MIN_MACS: usize = 1 << 21;

/// Scratch for one worker: one weight tile and one attention score row.
#[derive(Default)]
pub(crate) struct WorkerScratch {
    tile: Vec<f32>,
    scores: Vec<f32>,
}

/// Transient buffers for one evaluation; dropped when the batch returns.
pub(crate) struct Workspace {
    workers: Vec<WorkerScratch>,
    parallel_min_macs: usize,
    /// Largest weight tile held at one time, summed over workers, in bytes.
    pub(crate) peak_tile_bytes: usize,
    /// Largest number of workers that ran one product.
    pub(crate) peak_workers: usize,
}

impl Workspace {
    /// Creates scratch for at most `workers` participants. Products below
    /// `parallel_min_macs` multiply-adds run on the caller alone.
    pub(crate) fn new(workers: usize, parallel_min_macs: usize) -> Self {
        Self {
            workers: (0..workers.max(1))
                .map(|_| WorkerScratch::default())
                .collect(),
            parallel_min_macs,
            peak_tile_bytes: 0,
            peak_workers: 0,
        }
    }

    fn participants(&self, tokens: usize, macs_per_token: usize) -> usize {
        let work = tokens.saturating_mul(macs_per_token);
        if work < self.parallel_min_macs {
            return 1;
        }
        self.workers.len().min(tokens).max(1)
    }

    fn record(&mut self, participants: usize) {
        let tiles = self
            .workers
            .iter()
            .take(participants)
            .map(|worker| worker.tile.capacity().saturating_mul(4))
            .sum::<usize>();
        self.peak_tile_bytes = self.peak_tile_bytes.max(tiles);
        self.peak_workers = self.peak_workers.max(participants);
    }
}

/// Runs `task` over contiguous token ranges, on the caller and on up to
/// `participants - 1` scoped threads. Every thread is joined before return.
///
/// `task(first_token, input, output, scratch)` receives the input and output
/// rows of its range. The split never changes the arithmetic of one token.
fn split_tokens<F>(
    participants: usize,
    tokens: usize,
    input: (&[f32], usize),
    output: (&mut [f32], usize),
    workers: &mut [WorkerScratch],
    task: F,
) -> Result<(), RuntimeError>
where
    F: Fn(usize, &[f32], &mut [f32], &mut WorkerScratch) -> Result<(), RuntimeError> + Sync,
{
    let (input, input_width) = input;
    let (output, output_width) = output;
    if input.len() != tokens.saturating_mul(input_width)
        || output.len() != tokens.saturating_mul(output_width)
    {
        return Err(RuntimeError::Shape(
            "worker split does not match the activations".to_owned(),
        ));
    }
    let participants = participants
        .clamp(1, workers.len().max(1))
        .min(tokens.max(1));
    let per_worker = tokens.div_ceil(participants).max(1);
    let mut parts = input
        .chunks(per_worker.saturating_mul(input_width).max(1))
        .zip(output.chunks_mut(per_worker.saturating_mul(output_width).max(1)))
        .zip(workers.iter_mut())
        .enumerate()
        .map(|(index, ((input, output), scratch))| {
            (index.saturating_mul(per_worker), input, output, scratch)
        });
    let Some((first, own_input, own_output, own_scratch)) = parts.next() else {
        return Ok(());
    };
    let task = &task;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut spawn_error = None;
        for (start, input, output, scratch) in parts {
            match std::thread::Builder::new()
                .name("ze-text-cpu".to_owned())
                .spawn_scoped(scope, move || task(start, input, output, scratch))
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    spawn_error = Some(RuntimeError::Cpu(format!(
                        "could not start a compute worker: {error}"
                    )));
                    break;
                }
            }
        }
        // The caller works its own range only when every worker started.
        let own = match spawn_error {
            Some(error) => Err(error),
            None => task(first, own_input, own_output, own_scratch),
        };
        let mut result = own;
        for handle in handles {
            let joined = handle.join().unwrap_or(Err(RuntimeError::WorkerPanicked));
            if result.is_ok() {
                result = joined;
            }
        }
        result
    })
}

/// One linear product split over workers, each with its own weight tile.
fn parallel_linear(
    workspace: &mut Workspace,
    kernels: Kernels,
    input: &[f32],
    weight: WeightView<'_>,
    bias: &[f32],
    output: &mut [f32],
    gelu_output: bool,
) -> Result<(), RuntimeError> {
    let tokens = input.len().checked_div(weight.cols).unwrap_or(0);
    let participants = workspace.participants(tokens, weight.rows.saturating_mul(weight.cols));
    for scratch in workspace.workers.iter_mut().take(participants) {
        if scratch.tile.len() < TILE_FLOATS {
            scratch.tile.resize(TILE_FLOATS, 0.0);
        }
    }
    workspace.record(participants);
    split_tokens(
        participants,
        tokens,
        (input, weight.cols),
        (output, weight.rows),
        &mut workspace.workers,
        |_, input, output, scratch| {
            linear(
                kernels,
                input,
                weight,
                Some(bias),
                output,
                &mut scratch.tile,
            )?;
            if gelu_output {
                output.iter_mut().for_each(|value| *value = gelu(*value));
            }
            Ok(())
        },
    )
}

/// Exact-form GELU, `x * (1 + erf(x / sqrt(2))) / 2`, evaluated in f64.
pub(crate) fn gelu(value: f32) -> f32 {
    let x = f64::from(value);
    (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// erf with an absolute error below 1.5e-7 (Abramowitz and Stegun 7.1.26),
/// which is below the f32 rounding of the GELU result.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Accessor for one tower's checked tensor bytes in the mapped bundle.
pub(crate) struct Weights<'a> {
    pub(crate) bundle: &'a Bundle,
    pub(crate) tensors: &'a CpuWeights,
}

impl<'a> Weights<'a> {
    fn view(&self, tensor: &Tensor) -> Result<WeightView<'a>, RuntimeError> {
        let bytes = self
            .bundle
            .tensor_bytes(&tensor.descriptor)
            .map_err(|error| RuntimeError::Shape(error.to_string()))?;
        Ok(WeightView {
            bytes,
            dtype: tensor.descriptor.dtype,
            rows: tensor.rows,
            cols: tensor.cols,
        })
    }

    /// Widens a whole vector tensor (a bias or norm parameter) into `out`.
    fn vector(
        &self,
        kernels: Kernels,
        tensor: &Tensor,
        out: &mut Vec<f32>,
    ) -> Result<(), RuntimeError> {
        let view = self.view(tensor)?;
        let values = view.rows.saturating_mul(view.cols);
        out.clear();
        out.resize(values, 0.0);
        view.widen_rows(kernels, 0..view.rows, out).map(|_| ())
    }
}

fn layer_norm(
    weights: &Weights<'_>,
    kernels: Kernels,
    norm: &Norm,
    states: &mut [f32],
    width: usize,
    epsilon: f32,
    scratch: &mut (Vec<f32>, Vec<f32>),
) -> Result<(), RuntimeError> {
    weights.vector(kernels, &norm.weight, &mut scratch.0)?;
    weights.vector(kernels, &norm.bias, &mut scratch.1)?;
    if scratch.0.len() != width || scratch.1.len() != width || width == 0 {
        return Err(RuntimeError::Shape(
            "layer norm width does not match the hidden width".to_owned(),
        ));
    }
    let count = width as f32;
    for row in states.chunks_exact_mut(width) {
        let mean = row.iter().sum::<f32>() / count;
        let variance = row
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / count;
        let scale = 1.0 / (variance + epsilon).sqrt();
        for ((value, gain), shift) in row.iter_mut().zip(&scratch.0).zip(&scratch.1) {
            *value = (*value - mean) * scale * gain + shift;
        }
    }
    Ok(())
}

fn add_in_place(target: &mut [f32], addend: &[f32]) {
    for (value, other) in target.iter_mut().zip(addend) {
        *value += other;
    }
}

/// Validated shape of one evaluation.
struct Shape {
    rows: usize,
    sequence: usize,
    tokens: usize,
    hidden: usize,
    head_dim: usize,
    intermediate: usize,
}

fn shape(
    tower: &TowerSpec,
    weights: &CpuWeights,
    batch: &TokenBatch,
) -> Result<Shape, RuntimeError> {
    let tokens = batch
        .rows
        .checked_mul(batch.tokens_per_row)
        .ok_or_else(|| RuntimeError::Shape("token batch shape overflow".to_owned()))?;
    if batch.rows == 0 || batch.tokens_per_row == 0 || batch.token_ids.len() != tokens {
        return Err(RuntimeError::Shape(
            "token ids do not match the batch shape".to_owned(),
        ));
    }
    if batch.attention_mask.len() != tokens {
        return Err(RuntimeError::Shape(
            "attention mask does not match the batch shape".to_owned(),
        ));
    }
    let hidden = tower.config.hidden as usize;
    let heads = usize::from(tower.config.heads);
    if heads == 0 || !hidden.is_multiple_of(heads) {
        return Err(RuntimeError::Shape(
            "hidden width is not divisible by heads".to_owned(),
        ));
    }
    if batch.tokens_per_row > weights.position.rows {
        return Err(RuntimeError::Shape(format!(
            "{} tokens exceed {} position embeddings",
            batch.tokens_per_row, weights.position.rows
        )));
    }
    if let Some(id) = batch
        .token_ids
        .iter()
        .find(|id| usize::try_from(**id).map_or(true, |id| id >= weights.word.rows))
    {
        return Err(RuntimeError::Shape(format!(
            "token id {id} is outside {} embedding rows",
            weights.word.rows
        )));
    }
    if batch.attention_mask.iter().any(|value| !value.is_finite()) {
        return Err(RuntimeError::Shape(
            "attention mask holds a non-finite value".to_owned(),
        ));
    }
    Ok(Shape {
        rows: batch.rows,
        sequence: batch.tokens_per_row,
        tokens,
        hidden,
        head_dim: hidden / heads,
        intermediate: tower.config.intermediate as usize,
    })
}

/// Evaluates one rectangular batch and returns raw (not L2) pooled rows.
pub(crate) fn forward<E: From<RuntimeError>>(
    weights: &Weights<'_>,
    tower: &TowerSpec,
    kernels: Kernels,
    batch: &TokenBatch,
    workspace: &mut Workspace,
    checkpoint: &mut impl FnMut() -> Result<(), E>,
) -> Result<Vec<f32>, E> {
    let shape = shape(tower, weights.tensors, batch)?;
    let epsilon = tower.config.layer_norm_eps;
    let hidden = shape.hidden;
    let activations = shape
        .tokens
        .checked_mul(hidden.max(shape.intermediate))
        .ok_or_else(|| RuntimeError::Shape("activation size overflow".to_owned()))?;
    let mut states = vec![0.0_f32; shape.tokens * hidden];
    let mut row = vec![0.0_f32; hidden];
    let mut norm = (Vec::new(), Vec::new());
    let mut bias = Vec::new();

    // Embeddings: word row, position row (0..seq per batch row), type row 0.
    let word = weights.view(&weights.tensors.word)?;
    let position = weights.view(&weights.tensors.position)?;
    let mut positions = vec![0.0_f32; shape.sequence * hidden];
    position.widen_rows(kernels, 0..shape.sequence, &mut positions)?;
    let token_type = match &weights.tensors.token_type {
        Some(tensor) => {
            let mut values = vec![0.0_f32; hidden];
            weights
                .view(tensor)?
                .widen_rows(kernels, 0..1, &mut values)?;
            Some(values)
        }
        None => None,
    };
    for ((state, id), position) in states
        .chunks_exact_mut(hidden)
        .zip(&batch.token_ids)
        .zip(positions.chunks_exact(hidden).cycle())
    {
        let id = usize::try_from(*id)
            .map_err(|_| RuntimeError::Shape("negative token id".to_owned()))?;
        word.widen_rows(kernels, id..id.saturating_add(1), &mut row)?;
        for (((value, word), position), kind) in state
            .iter_mut()
            .zip(&row)
            .zip(position)
            .zip(token_type.iter().flatten().chain(std::iter::repeat(&0.0)))
        {
            *value = word + position + kind;
        }
    }
    drop(positions);
    layer_norm(
        weights,
        kernels,
        &weights.tensors.embedding_norm,
        &mut states,
        hidden,
        epsilon,
        &mut norm,
    )?;

    // Additive key bias, as in arch/bert.rs: (mask - 1) * 1e9.
    let key_bias = batch
        .attention_mask
        .iter()
        .map(|mask| (mask - 1.0) * 1.0e9)
        .collect::<Vec<_>>();
    let mut query = vec![0.0_f32; shape.tokens * hidden];
    let mut key = vec![0.0_f32; shape.tokens * hidden];
    let mut value = vec![0.0_f32; shape.tokens * hidden];
    let mut attended = vec![0.0_f32; shape.tokens * hidden];
    let mut wide = vec![0.0_f32; activations];
    let scale = 1.0 / (shape.head_dim as f32).sqrt();

    for layer in &weights.tensors.layers {
        checkpoint()?;
        for (projection, out) in [
            (&layer.query, &mut query),
            (&layer.key, &mut key),
            (&layer.value, &mut value),
        ] {
            run_linear(
                weights, workspace, kernels, projection, &states, out, &mut bias, false,
            )?;
        }
        attention(
            workspace,
            kernels,
            &shape,
            (&query, &key, &value),
            &key_bias,
            scale,
            &mut attended,
        )?;
        // `query` now holds the attention output projection.
        run_linear(
            weights,
            workspace,
            kernels,
            &layer.attention_output,
            &attended,
            &mut query,
            &mut bias,
            false,
        )?;
        add_in_place(&mut states, &query);
        layer_norm(
            weights,
            kernels,
            &layer.attention_norm,
            &mut states,
            hidden,
            epsilon,
            &mut norm,
        )?;
        let intermediate = wide
            .get_mut(..shape.tokens * shape.intermediate)
            .ok_or_else(|| RuntimeError::Shape("intermediate buffer is short".to_owned()))?;
        run_linear(
            weights,
            workspace,
            kernels,
            &layer.intermediate,
            &states,
            intermediate,
            &mut bias,
            true,
        )?;
        run_linear(
            weights,
            workspace,
            kernels,
            &layer.output,
            intermediate,
            &mut query,
            &mut bias,
            false,
        )?;
        add_in_place(&mut states, &query);
        layer_norm(
            weights,
            kernels,
            &layer.output_norm,
            &mut states,
            hidden,
            epsilon,
            &mut norm,
        )?;
    }
    checkpoint()?;
    drop((query, key, value, attended, wide));
    finish(
        weights, tower, kernels, &shape, batch, &states, workspace, &mut bias,
    )
    .map_err(E::from)
}

#[allow(clippy::too_many_arguments)]
fn run_linear(
    weights: &Weights<'_>,
    workspace: &mut Workspace,
    kernels: Kernels,
    layer: &Linear,
    input: &[f32],
    output: &mut [f32],
    bias: &mut Vec<f32>,
    gelu_output: bool,
) -> Result<(), RuntimeError> {
    let view = weights.view(&layer.weight)?;
    weights.vector(kernels, &layer.bias, bias)?;
    parallel_linear(workspace, kernels, input, view, bias, output, gelu_output)
}

/// Masked softmax attention per query token and head, split over query tokens.
fn attention(
    workspace: &mut Workspace,
    kernels: Kernels,
    shape: &Shape,
    (query, key, value): (&[f32], &[f32], &[f32]),
    key_bias: &[f32],
    scale: f32,
    output: &mut [f32],
) -> Result<(), RuntimeError> {
    let per_token = shape
        .sequence
        .saturating_mul(shape.hidden)
        .saturating_mul(2);
    let participants = workspace.participants(shape.tokens, per_token);
    for scratch in workspace.workers.iter_mut().take(participants) {
        scratch.scores.resize(shape.sequence, 0.0);
    }
    let (sequence, hidden, head_dim) = (shape.sequence, shape.hidden, shape.head_dim);
    split_tokens(
        participants,
        shape.tokens,
        (query, hidden),
        (output, hidden),
        &mut workspace.workers,
        |first, queries, outputs, scratch| {
            let scores = scratch
                .scores
                .get_mut(..sequence)
                .ok_or_else(|| RuntimeError::Shape("score buffer is short".to_owned()))?;
            for (offset, (query_row, output_row)) in queries
                .chunks_exact(hidden)
                .zip(outputs.chunks_exact_mut(hidden))
                .enumerate()
            {
                let token = first.saturating_add(offset);
                let start = (token / sequence).saturating_mul(sequence);
                let end = start.saturating_add(sequence);
                let keys = key
                    .get(start * hidden..end * hidden)
                    .ok_or_else(|| RuntimeError::Shape("key rows are short".to_owned()))?;
                let values = value
                    .get(start * hidden..end * hidden)
                    .ok_or_else(|| RuntimeError::Shape("value rows are short".to_owned()))?;
                let biases = key_bias
                    .get(start..end)
                    .ok_or_else(|| RuntimeError::Shape("mask row is short".to_owned()))?;
                for ((query_head, output_head), head) in query_row
                    .chunks_exact(head_dim)
                    .zip(output_row.chunks_exact_mut(head_dim))
                    .zip(0..)
                {
                    let lane = head * head_dim..(head + 1) * head_dim;
                    let mut maximum = f32::NEG_INFINITY;
                    for ((score, key_row), bias) in
                        scores.iter_mut().zip(keys.chunks_exact(hidden)).zip(biases)
                    {
                        let key_head = key_row.get(lane.clone()).unwrap_or(&[]);
                        *score = kernels.dot(query_head, key_head) * scale + bias;
                        maximum = maximum.max(*score);
                    }
                    let mut total = 0.0_f32;
                    for score in scores.iter_mut() {
                        *score = (*score - maximum).exp();
                        total += *score;
                    }
                    output_head.fill(0.0);
                    for (score, value_row) in scores.iter().zip(values.chunks_exact(hidden)) {
                        let weight = score / total;
                        for (out, value) in output_head
                            .iter_mut()
                            .zip(value_row.get(lane.clone()).unwrap_or(&[]))
                        {
                            *out += weight * value;
                        }
                    }
                }
            }
            Ok(())
        },
    )
}

/// Pooling, the optional dense head and truncation, as in `arch/bert.rs::finish`.
#[allow(clippy::too_many_arguments)]
fn finish(
    weights: &Weights<'_>,
    tower: &TowerSpec,
    kernels: Kernels,
    shape: &Shape,
    batch: &TokenBatch,
    states: &[f32],
    workspace: &mut Workspace,
    bias: &mut Vec<f32>,
) -> Result<Vec<f32>, RuntimeError> {
    let hidden = shape.hidden;
    let mut pooled = vec![0.0_f32; shape.rows * hidden];
    for ((output, rows), mask) in pooled
        .chunks_exact_mut(hidden)
        .zip(states.chunks_exact(shape.sequence * hidden))
        .zip(batch.attention_mask.chunks_exact(shape.sequence))
    {
        match tower.pooling {
            Pooling::Mean => {
                let count = mask.iter().sum::<f32>();
                if count <= 0.0 {
                    return Err(RuntimeError::Shape(
                        "mean pooling row has no attended token".to_owned(),
                    ));
                }
                for (state, weight) in rows.chunks_exact(hidden).zip(mask) {
                    for (sum, value) in output.iter_mut().zip(state) {
                        *sum += value * weight;
                    }
                }
                output.iter_mut().for_each(|value| *value /= count);
            }
            Pooling::Cls => output.copy_from_slice(rows.get(..hidden).unwrap_or(&[])),
            Pooling::Last => {
                // arch/bert.rs selects position (number of attended tokens - 1).
                let length = mask.iter().filter(|value| **value > 0.0).count();
                if let Some(last) = length.checked_sub(1) {
                    let start = last.saturating_mul(hidden);
                    let state = rows
                        .get(start..start.saturating_add(hidden))
                        .ok_or_else(|| {
                            RuntimeError::Shape("last token is out of range".to_owned())
                        })?;
                    output.copy_from_slice(state);
                }
            }
        }
    }
    let (pooled, width) = match &weights.tensors.dense {
        Some(dense) => {
            let mut projected = vec![0.0_f32; shape.rows * dense.weight.rows];
            run_linear(
                weights,
                workspace,
                kernels,
                dense,
                &pooled,
                &mut projected,
                bias,
                false,
            )?;
            (projected, dense.weight.rows)
        }
        None => (pooled, hidden),
    };
    let dims = tower.embedding.dims as usize;
    if dims > width || dims == 0 {
        return Err(RuntimeError::Shape(
            "declared output dimensions exceed architecture output".to_owned(),
        ));
    }
    if dims == width {
        return Ok(pooled);
    }
    Ok(pooled
        .chunks_exact(width)
        .flat_map(|row| row.iter().take(dims).copied())
        .collect())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use std::sync::Arc;

    use crate::bundle::Bundle;
    use crate::runtime::ModelRuntime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::runtime::cpu::kernels::Kernels;
    use crate::tower::{TokenBatch, TowerRole};

    #[allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]
    mod wf991 {
        include!("../../../tests/fixtures/wf991/reference.rs");
    }

    #[allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]
    mod ze115 {
        include!("../../../tests/fixtures/ze115/reference.rs");
    }

    fn bundle(bytes: &[u8]) -> (tempfile::TempDir, Arc<Bundle>) {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("fixture.zem");
        std::fs::write(&path, bytes).expect("write fixture");
        let bundle = Arc::new(Bundle::open(&path).expect("valid fixture"));
        (directory, bundle)
    }

    fn batch(rows: &[usize]) -> TokenBatch {
        let width = rows
            .iter()
            .map(|row| wf991::MASK[*row].iter().filter(|m| **m > 0.0).count())
            .max()
            .expect("rows");
        TokenBatch::new(
            rows.iter()
                .flat_map(|row| wf991::IDS[*row][..width].to_vec())
                .collect(),
            rows.iter()
                .flat_map(|row| wf991::MASK[*row][..width].to_vec())
                .collect(),
            rows.len(),
            width,
        )
        .expect("token batch")
    }

    #[test]
    fn wf991_tiny_bert_f16_oracle() {
        let bytes = include_bytes!("../../../tests/fixtures/wf991/pair-f16.zem");
        let (_directory, bundle) = bundle(bytes);
        for (role, reference) in [
            (TowerRole::Document, wf991::DOCUMENT_RAW),
            (TowerRole::Query, wf991::QUERY_RAW),
        ] {
            let mut runtime = CpuRuntime::load(Arc::clone(&bundle), role).expect("FP16 tower");
            assert_eq!(runtime.identity().name, "cpu-fp32");
            for kernels in Kernels::available() {
                runtime.set_kernels(kernels);
                let mut outputs = Vec::new();
                for workers in [1, 4] {
                    runtime.force_workers(workers);
                    runtime.last_workspace = (0, 0);
                    let raw = runtime.embed_batch(&batch(&[0, 1])).expect("ragged batch");
                    assert_eq!((raw.rows(), raw.dims()), (2, 32));
                    let mut worst = 0.0_f32;
                    for (row, expected) in raw.values().chunks_exact(32).zip(reference) {
                        for (actual, expected) in row.iter().zip(expected) {
                            worst = worst.max((actual - expected).abs());
                        }
                    }
                    println!(
                        "{role:?} {} workers={workers} (peak {}): max |cpu - fp64 oracle| = {worst:e}",
                        kernels.name(),
                        runtime.last_workspace.1
                    );
                    assert!(worst <= 2.0e-5, "{role:?} {} {worst}", kernels.name());
                    assert_eq!(runtime.last_workspace.1, workers, "workers really ran");
                    // Each row alone equals its row in the ragged batch.
                    for row in 0..2 {
                        let single = runtime.embed_batch(&batch(&[row])).expect("single row");
                        assert_eq!(single.values(), &raw.values()[row * 32..(row + 1) * 32]);
                    }
                    outputs.push(raw.into_values());
                }
                assert_eq!(outputs[0], outputs[1], "1 worker equals 4 workers");
            }
        }
    }

    /// ze115 F32 BERT: mean, CLS and last pooling plus the dense head and
    /// truncation, against its independent PyTorch reference.
    #[test]
    fn wf991_tiny_bert_pooling_and_dense_head() {
        let ids = [[2, 5, 6, 3, 0], [2, 6, 3, 0, 0]];
        let tokens = TokenBatch::new(
            ids.concat(),
            vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            2,
            5,
        )
        .expect("tokens");
        for (index, pooling) in [1_u8, 2, 3].into_iter().enumerate() {
            for dims in [1_u32, 3] {
                let mut bytes = include_bytes!("../../../tests/fixtures/ze115/bert.zem").to_vec();
                bytes[ze115::POOLING_OFFSET] = pooling;
                bytes[ze115::DIMS_OFFSET..ze115::DIMS_OFFSET + 4]
                    .copy_from_slice(&dims.to_le_bytes());
                let end = bytes.len() - 16;
                let digest = xxhash_rust::xxh3::xxh3_128(&bytes[..end]);
                bytes[end..].copy_from_slice(&digest.to_le_bytes());
                let (_directory, bundle) = bundle(&bytes);
                let mut runtime =
                    CpuRuntime::load(bundle, TowerRole::Query).expect("symmetric tower");
                let output = runtime.embed_batch(&tokens).expect("batch");
                for (row, values) in output.values().chunks_exact(dims as usize).enumerate() {
                    for (actual, expected) in values.iter().zip(ze115::BERT_REFERENCE[index][row]) {
                        assert!(
                            (actual - expected).abs() <= 2.0e-5,
                            "pooling {pooling} dims {dims} row {row}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}
