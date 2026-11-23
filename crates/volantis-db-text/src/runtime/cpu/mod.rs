//! Portable fp32 CPU runtime for BERT towers.
//!
//! Weights stay in the mapped bundle in their stored dtype (F16 or F32). Each
//! linear layer widens one tile of weight rows at a time into a per-worker
//! buffer; the runtime keeps no fp32 copy of a tower. Large products split over
//! at most [`MAX_WORKERS`] scoped threads, joined before every return.

use std::sync::Arc;

use crate::bundle::{Bundle, TensorDescriptor, TensorDtype};
use crate::runtime::limits::{MAX_EVAL_ROWS, next_eval_chunk};
use crate::runtime::tensors::{required_tensor_names, tensor_prefix};
use crate::runtime::{Backend, EmbeddingBatch, ModelRuntime, RuntimeError, RuntimeIdentity};
use crate::tower::{Architecture, TokenBatch, TowerRole, TowerSpec};

mod bert;
mod kernels;

use kernels::Kernels;

/// Most threads, the caller included, that one product uses.
pub const MAX_WORKERS: usize = 4;

/// Stable runtime name reported in diagnostics.
pub const CPU_RUNTIME_NAME: &str = "cpu-fp32";

/// One checked tensor: its bundle descriptor and its 2-D view.
#[derive(Clone, Debug)]
pub(crate) struct Tensor {
    pub(crate) descriptor: TensorDescriptor,
    /// Leading dimension (output features or table rows); 1 for a vector.
    pub(crate) rows: usize,
    /// Trailing dimension (input features or table width).
    pub(crate) cols: usize,
}

/// Weight and bias of one linear layer.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
}

/// Gain and shift of one LayerNorm.
#[derive(Clone, Debug)]
pub(crate) struct Norm {
    pub(crate) weight: Tensor,
    pub(crate) bias: Tensor,
}

/// One post-norm encoder layer.
#[derive(Clone, Debug)]
pub(crate) struct Layer {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) attention_output: Linear,
    pub(crate) attention_norm: Norm,
    pub(crate) intermediate: Linear,
    pub(crate) output: Linear,
    pub(crate) output_norm: Norm,
}

/// Every tensor one BERT tower reads, resolved and checked once at load.
#[derive(Clone, Debug)]
pub(crate) struct CpuWeights {
    pub(crate) word: Tensor,
    pub(crate) position: Tensor,
    pub(crate) token_type: Option<Tensor>,
    pub(crate) embedding_norm: Norm,
    pub(crate) layers: Vec<Layer>,
    pub(crate) dense: Option<Linear>,
}

struct Resolver<'a> {
    bundle: &'a Bundle,
    prefix: &'static str,
}

impl Resolver<'_> {
    /// Resolves `name` and checks its dtype, shape and byte length.
    fn tensor(&self, name: &str, shape: &[usize]) -> Result<Tensor, RuntimeError> {
        let qualified = format!("{}{name}", self.prefix);
        let descriptor = self
            .bundle
            .tensor(&qualified)
            .ok_or_else(|| RuntimeError::MissingTensor(qualified.clone()))?;
        let element = match descriptor.dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            _ => return Err(RuntimeError::UnsupportedDtype(qualified)),
        };
        let actual = descriptor
            .shape
            .iter()
            .map(|dim| *dim as usize)
            .collect::<Vec<_>>();
        if actual != shape {
            return Err(RuntimeError::Shape(format!(
                "tensor {qualified} has shape {actual:?}, expected {shape:?}"
            )));
        }
        let values = shape
            .iter()
            .try_fold(1_usize, |total, dim| total.checked_mul(*dim))
            .ok_or_else(|| RuntimeError::Shape(format!("tensor {qualified} size overflows")))?;
        let bytes = self
            .bundle
            .tensor_bytes(descriptor)
            .map_err(|error| RuntimeError::Shape(error.to_string()))?;
        if values.checked_mul(element) != Some(bytes.len()) || values == 0 {
            return Err(RuntimeError::Shape(format!(
                "tensor {qualified} has {} bytes for {values} values",
                bytes.len()
            )));
        }
        let (rows, cols) = match shape {
            [cols] => (1, *cols),
            [rows, cols] => (*rows, *cols),
            _ => {
                return Err(RuntimeError::Shape(format!(
                    "tensor {qualified} is not a vector or matrix"
                )));
            }
        };
        Ok(Tensor {
            descriptor: descriptor.clone(),
            rows,
            cols,
        })
    }

    fn linear(&self, base: &str, out: usize, input: usize) -> Result<Linear, RuntimeError> {
        Ok(Linear {
            weight: self.tensor(&format!("{base}.weight"), &[out, input])?,
            bias: self.tensor(&format!("{base}.bias"), &[out])?,
        })
    }

    fn norm(&self, base: &str, width: usize) -> Result<Norm, RuntimeError> {
        Ok(Norm {
            weight: self.tensor(&format!("{base}.weight"), &[width])?,
            bias: self.tensor(&format!("{base}.bias"), &[width])?,
        })
    }
}

impl CpuWeights {
    fn resolve(bundle: &Bundle, role: TowerRole, tower: &TowerSpec) -> Result<Self, RuntimeError> {
        let resolver = Resolver {
            bundle,
            prefix: tensor_prefix(bundle, role),
        };
        // The shared list is the contract with the MLX runtime: every name it
        // requires must exist, even where the CPU pass reads it by structure.
        for name in required_tensor_names(tower) {
            let qualified = format!("{}{name}", resolver.prefix);
            if bundle.tensor(&qualified).is_none() {
                return Err(RuntimeError::MissingTensor(qualified));
            }
        }
        let config = &tower.config;
        let hidden = config.hidden as usize;
        let intermediate = config.intermediate as usize;
        let word = bundle
            .tensor(&format!(
                "{}embeddings.word_embeddings.weight",
                resolver.prefix
            ))
            .and_then(|tensor| tensor.shape.first())
            .map_or(0, |rows| *rows as usize);
        let positions = bundle
            .tensor(&format!(
                "{}embeddings.position_embeddings.weight",
                resolver.prefix
            ))
            .and_then(|tensor| tensor.shape.first())
            .map_or(0, |rows| *rows as usize);
        let layers = (0..config.layer_count)
            .map(|layer| {
                let base = format!("encoder.layer.{layer}");
                Ok(Layer {
                    query: resolver.linear(
                        &format!("{base}.attention.self.query"),
                        hidden,
                        hidden,
                    )?,
                    key: resolver.linear(&format!("{base}.attention.self.key"), hidden, hidden)?,
                    value: resolver.linear(
                        &format!("{base}.attention.self.value"),
                        hidden,
                        hidden,
                    )?,
                    attention_output: resolver.linear(
                        &format!("{base}.attention.output.dense"),
                        hidden,
                        hidden,
                    )?,
                    attention_norm: resolver
                        .norm(&format!("{base}.attention.output.LayerNorm"), hidden)?,
                    intermediate: resolver.linear(
                        &format!("{base}.intermediate.dense"),
                        intermediate,
                        hidden,
                    )?,
                    output: resolver.linear(
                        &format!("{base}.output.dense"),
                        hidden,
                        intermediate,
                    )?,
                    output_norm: resolver.norm(&format!("{base}.output.LayerNorm"), hidden)?,
                })
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        Ok(Self {
            word: resolver.tensor("embeddings.word_embeddings.weight", &[word, hidden])?,
            position: resolver.tensor(
                "embeddings.position_embeddings.weight",
                &[positions, hidden],
            )?,
            token_type: if config.type_vocab > 0 {
                Some(resolver.tensor(
                    "embeddings.token_type_embeddings.weight",
                    &[config.type_vocab as usize, hidden],
                )?)
            } else {
                None
            },
            embedding_norm: resolver.norm("embeddings.LayerNorm", hidden)?,
            layers,
            dense: if config.dense_out > 0 {
                Some(resolver.linear("dense", config.dense_out as usize, hidden)?)
            } else {
                None
            },
        })
    }
}

/// One BERT tower evaluated in fp32 on the CPU.
pub struct CpuRuntime {
    bundle: Arc<Bundle>,
    tower: TowerSpec,
    weights: CpuWeights,
    kernels: Kernels,
    workers: usize,
    parallel_min_macs: usize,
    #[cfg(test)]
    pub(crate) observer: Option<crate::runtime::limits::EvalObserver>,
    #[cfg(test)]
    pub(crate) last_workspace: (usize, usize),
}

impl CpuRuntime {
    /// Verifies one tower's digests, then resolves and checks its tensors.
    pub fn load(bundle: Arc<Bundle>, role: TowerRole) -> Result<Self, RuntimeError> {
        bundle
            .verify_tower(role)
            .map_err(|error| RuntimeError::Shape(error.to_string()))?;
        let tower = match role {
            TowerRole::Document => bundle.document_tower(),
            TowerRole::Query => bundle.query_tower(),
        }
        .clone();
        if tower.architecture != Architecture::Bert {
            return Err(RuntimeError::UnsupportedArchitecture {
                backend: Backend::Cpu.name(),
                architecture: tower.architecture,
            });
        }
        let weights = CpuWeights::resolve(&bundle, role, &tower)?;
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .clamp(1, MAX_WORKERS);
        Ok(Self {
            bundle,
            tower,
            weights,
            kernels: Kernels::detect(),
            workers,
            parallel_min_macs: bert::PARALLEL_MIN_MACS,
            #[cfg(test)]
            observer: None,
            #[cfg(test)]
            last_workspace: (0, 0),
        })
    }

    /// Kernel family selected for this CPU, for diagnostics and evidence.
    #[must_use]
    pub fn kernels(&self) -> &'static str {
        self.kernels.name()
    }

    /// Thread count, the caller included, that one large product may use.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Forces `workers` participants for every product, however small.
    #[cfg(test)]
    pub(crate) fn force_workers(&mut self, workers: usize) {
        self.workers = workers.clamp(1, MAX_WORKERS);
        self.parallel_min_macs = 0;
    }

    #[cfg(test)]
    pub(crate) fn set_kernels(&mut self, kernels: Kernels) {
        self.kernels = kernels;
    }

    /// Evaluates a batch in chunks bounded like the MLX runtime. The
    /// checkpoint runs between chunks and layers; an error from it returns
    /// with every worker joined and no partial batch.
    pub(crate) fn embed_batch_controlled<E: From<RuntimeError>>(
        &mut self,
        tokens: &TokenBatch,
        checkpoint: &mut impl FnMut() -> Result<(), E>,
    ) -> Result<EmbeddingBatch, E> {
        checkpoint()?;
        let dims = self.tower.embedding.dims as usize;
        let capacity = tokens
            .rows
            .checked_mul(dims)
            .ok_or_else(|| RuntimeError::Shape("embedding output size overflow".to_owned()))?;
        let mut values = Vec::with_capacity(capacity);
        #[cfg(test)]
        let mut evaluations = Vec::new();
        let weights = bert::Weights {
            bundle: &self.bundle,
            tensors: &self.weights,
        };
        let mut start = 0_usize;
        while start < tokens.rows {
            checkpoint()?;
            let chunk = next_eval_chunk(tokens, start)?;
            #[cfg(test)]
            evaluations.push(chunk.rows);
            let mut workspace = bert::Workspace::new(self.workers, self.parallel_min_macs);
            let output = bert::forward(
                &weights,
                &self.tower,
                self.kernels,
                &chunk,
                &mut workspace,
                checkpoint,
            )?;
            #[cfg(test)]
            {
                self.last_workspace = (
                    self.last_workspace.0.max(workspace.peak_tile_bytes),
                    self.last_workspace.1.max(workspace.peak_workers),
                );
            }
            values.extend(output);
            start = start.saturating_add(chunk.rows);
        }
        checkpoint()?;
        #[cfg(test)]
        if let Some(observer) = &self.observer {
            observer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((tokens.rows, evaluations));
        }
        EmbeddingBatch::new(values, tokens.rows, dims).map_err(Into::into)
    }

    /// Returns the row ceiling of one evaluation.
    #[doc(hidden)]
    #[must_use]
    pub const fn max_eval_rows() -> usize {
        MAX_EVAL_ROWS
    }
}

impl ModelRuntime for CpuRuntime {
    fn embed_batch(&mut self, tokens: &TokenBatch) -> Result<EmbeddingBatch, RuntimeError> {
        self.embed_batch_controlled(tokens, &mut || Ok::<(), RuntimeError>(()))
    }

    fn warm(&mut self) -> Result<(), RuntimeError> {
        let tokens = TokenBatch::new(vec![0], vec![1.0], 1, 1)
            .map_err(|detail| RuntimeError::Shape(detail.to_owned()))?;
        self.embed_batch(&tokens).map(|_| ())
    }

    fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            name: CPU_RUNTIME_NAME,
            gpu: false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;

    const PAIR: &[u8] = include_bytes!("../../../tests/fixtures/wf991/pair-f16.zem");

    fn open(bytes: &[u8]) -> (tempfile::TempDir, Arc<Bundle>) {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("fixture.zem");
        let mut bytes = bytes.to_vec();
        let end = bytes.len() - 16;
        let digest = xxhash_rust::xxh3::xxh3_128(&bytes[..end]);
        bytes[end..].copy_from_slice(&digest.to_le_bytes());
        std::fs::write(&path, bytes).expect("write fixture");
        let bundle = Arc::new(Bundle::open(&path).expect("bundle opens"));
        (directory, bundle)
    }

    /// Byte position of the dtype field of one tensor record in the header.
    fn record(bytes: &[u8], name: &str) -> usize {
        let mut field = (name.len() as u32).to_le_bytes().to_vec();
        field.extend_from_slice(name.as_bytes());
        bytes
            .windows(field.len())
            .position(|window| window == field)
            .expect("tensor record")
            + field.len()
    }

    fn tokens(ids: Vec<i32>, mask: Vec<f32>, rows: usize, width: usize) -> TokenBatch {
        TokenBatch {
            token_ids: ids,
            attention_mask: mask,
            rows,
            tokens_per_row: width,
        }
    }

    #[test]
    fn wf991_rejects_invalid_input() {
        let (_directory, bundle) = open(PAIR);
        let word = bundle
            .tensor("document/embeddings.word_embeddings.weight")
            .expect("word table")
            .clone();

        // A flipped weight byte fails tower verification before any load.
        let mut flipped = PAIR.to_vec();
        flipped[word.offset + 3] ^= 0x40;
        let (_flipped_dir, flipped) = open(&flipped);
        assert!(matches!(
            CpuRuntime::load(flipped, TowerRole::Document),
            Err(RuntimeError::Shape(detail)) if detail.contains("digest")
        ));

        // A wrong shape keeps every digest valid and still fails at load.
        let mut reshaped = PAIR.to_vec();
        let at = record(
            &reshaped,
            "document/encoder.layer.0.attention.self.key.weight",
        );
        reshaped[at + 4..at + 8].copy_from_slice(&16_u32.to_le_bytes());
        reshaped[at + 8..at + 12].copy_from_slice(&64_u32.to_le_bytes());
        let (_reshaped_dir, reshaped) = open(&reshaped);
        assert!(matches!(
            CpuRuntime::load(reshaped, TowerRole::Document),
            Err(RuntimeError::Shape(detail)) if detail.contains("shape")
        ));

        // A dtype the CPU runtime does not widen is a typed error.
        let mut bf16 = PAIR.to_vec();
        let at = record(&bf16, "document/encoder.layer.1.output.dense.bias");
        bf16[at] = 2;
        let (_bf16_dir, bf16) = open(&bf16);
        assert!(matches!(
            CpuRuntime::load(bf16, TowerRole::Document),
            Err(RuntimeError::UnsupportedDtype(name)) if name.ends_with("output.dense.bias")
        ));

        // GTE and ArcticQuery towers are not run on the CPU.
        let (_gte_dir, gte) = open(include_bytes!("../../../tests/fixtures/ze115/gte.zem"));
        assert!(matches!(
            CpuRuntime::load(gte, TowerRole::Document),
            Err(RuntimeError::UnsupportedArchitecture {
                backend: "cpu",
                architecture: Architecture::Gte
            })
        ));

        let mut runtime =
            CpuRuntime::load(Arc::clone(&bundle), TowerRole::Document).expect("document tower");
        runtime.force_workers(4);
        let good = tokens(vec![2, 4, 5, 6, 3], vec![1.0; 5], 1, 5);
        let expected = runtime.embed_batch(&good).expect("valid batch");
        for (bad, why) in [
            (
                tokens(vec![2, 32, 3], vec![1.0; 3], 1, 3),
                "id past the table",
            ),
            (tokens(vec![2, -1, 3], vec![1.0; 3], 1, 3), "negative id"),
            (tokens(vec![2, 4, 3], vec![1.0; 2], 1, 3), "short mask"),
            (
                tokens(vec![2; 17], vec![1.0; 17], 1, 17),
                "past the positions",
            ),
            (
                tokens(vec![2, 4, 3], vec![1.0, f32::NAN, 1.0], 1, 3),
                "NaN mask",
            ),
            (tokens(vec![2, 4, 3, 2], vec![1.0; 4], 2, 3), "ragged ids"),
        ] {
            assert!(
                matches!(runtime.embed_batch(&bad), Err(RuntimeError::Shape(_))),
                "{why}"
            );
        }

        // Cancellation between layers joins the workers and returns no batch.
        for stop_at in 1..=5 {
            let mut calls = 0;
            let result = runtime.embed_batch_controlled(&good, &mut || {
                calls += 1;
                if calls == stop_at {
                    Err(RuntimeError::Cpu("cancelled".to_owned()))
                } else {
                    Ok(())
                }
            });
            assert!(
                matches!(&result, Err(RuntimeError::Cpu(detail)) if detail == "cancelled"),
                "stop at checkpoint {stop_at}: {result:?}"
            );
        }
        // The next call succeeds with the same bits.
        let again = runtime.embed_batch(&good).expect("after cancellation");
        assert_eq!(again.values(), expected.values());
    }

    /// Peak resident bytes of this process, from getrusage (bytes on macOS).
    #[cfg(target_os = "macos")]
    fn peak_resident_bytes() -> u64 {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage writes one rusage into the valid, zeroed buffer.
        let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        assert_eq!(status, 0);
        // SAFETY: the call succeeded, so the buffer is initialized.
        u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).expect("maxrss")
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires WF991_BUNDLE (released FP16 bundle); run in release"]
    fn wf991_weight_storage() {
        let path = std::env::var_os("WF991_BUNDLE").expect("set WF991_BUNDLE");
        let file_bytes = std::fs::metadata(&path).expect("bundle size").len();
        let before = peak_resident_bytes();
        let bundle = Arc::new(Bundle::open(&path).expect("release bundle"));
        let mut fp16_values = 0_u64;
        for (role, text) in [
            (TowerRole::Query, "What is the capital of France?"),
            (
                TowerRole::Document,
                &"Paris is the capital and largest city of France. ".repeat(8),
            ),
        ] {
            let mut runtime = CpuRuntime::load(Arc::clone(&bundle), role).expect("tower");
            assert!(
                std::mem::size_of::<CpuRuntime>() < 1024,
                "no inline weight storage"
            );
            let tokens = match role {
                TowerRole::Query => bundle.tokenize_queries(&[text, "moon orbit"]),
                TowerRole::Document => bundle.tokenize_document_chunks(&[text.to_owned()]),
            }
            .expect("tokens");
            let output = runtime.embed_batch(&tokens).expect("batch");
            assert!(output.values().iter().all(|v| v.is_finite()));
            let (peak_tile_bytes, peak_workers) = runtime.last_workspace;
            let bound = runtime.workers() * kernels::TILE_FLOATS * 4;
            println!(
                "{role:?}: kernels={} workers={} tile={} floats, rows={} tokens/row={}, \
                 peak widened tile bytes={peak_tile_bytes} (bound {bound}), peak workers={peak_workers}",
                runtime.kernels(),
                runtime.workers(),
                kernels::TILE_FLOATS,
                tokens.rows,
                tokens.tokens_per_row,
            );
            assert!(peak_tile_bytes > 0 && peak_tile_bytes <= bound);
            fp16_values += [&runtime.weights.word, &runtime.weights.position]
                .into_iter()
                .chain(runtime.weights.layers.iter().flat_map(|layer| {
                    [
                        &layer.query.weight,
                        &layer.key.weight,
                        &layer.value.weight,
                        &layer.attention_output.weight,
                        &layer.intermediate.weight,
                        &layer.output.weight,
                    ]
                }))
                .map(|tensor| (tensor.rows * tensor.cols) as u64)
                .sum::<u64>();
        }
        let grown = peak_resident_bytes().saturating_sub(before);
        let fp32_copy = fp16_values * 4;
        println!(
            "bundle file {file_bytes} B; peak RSS growth {grown} B; an fp32 copy of the \
             matrices alone would be {fp32_copy} B"
        );
        // Mapped FP16 pages plus activations stay well under file + fp32 copy.
        assert!(
            grown < file_bytes + fp32_copy / 4,
            "the runtime kept fp32 weights"
        );
    }
}
