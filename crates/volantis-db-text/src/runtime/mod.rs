use crate::tower::TokenBatch;

/// ML runtime identity included in evidence and diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeIdentity {
    /// Stable runtime name.
    pub name: &'static str,
    /// Whether inference uses the GPU stream.
    pub gpu: bool,
}

/// One contiguous evaluated batch returned by a runtime.
#[derive(Debug)]
pub struct EmbeddingBatch {
    values: Vec<f32>,
    rows: usize,
    dims: usize,
}

impl EmbeddingBatch {
    /// Constructs a shape-checked contiguous batch.
    pub fn new(values: Vec<f32>, rows: usize, dims: usize) -> Result<Self, RuntimeError> {
        if rows == 0 || dims == 0 || rows.checked_mul(dims) != Some(values.len()) {
            return Err(RuntimeError::Shape(
                "embedding batch shape mismatch".to_owned(),
            ));
        }
        Ok(Self { values, rows, dims })
    }

    /// Returns the full contiguous batch view.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns mutable contiguous values for normalization.
    #[must_use]
    pub(crate) fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Returns the number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the coordinates in each row.
    #[must_use]
    pub const fn dims(&self) -> usize {
        self.dims
    }

    /// Moves the complete contiguous allocation to the receiver.
    #[must_use]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Typed model-runtime failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeError {
    /// Model tokenizer rejected input or bundle tables.
    Tokenization(String),
    /// An MLX operation failed.
    Mlx(String),
    /// ZE_MLX_CACHE_LIMIT_BYTES is not a byte count representable as usize.
    InvalidCacheLimit,
    /// A later open requested a different process-wide cache limit.
    CacheLimitConflict {
        /// Limit selected by the first opener, in bytes.
        configured: usize,
        /// Explicit limit requested by this opener, in bytes.
        requested: usize,
    },
    /// An MLX cache C API call returned a nonzero status.
    MlxCache {
        /// Native function that failed.
        operation: &'static str,
        /// Native return status.
        status: libc::c_int,
    },
    /// A required tensor was absent.
    MissingTensor(String),
    /// Tensor bytes or array dimensions disagreed with metadata.
    Shape(String),
    /// Persisted dtype is not implemented by the commit-1 runtime.
    UnsupportedDtype(String),
    /// The embed worker panicked while evaluating a submitted batch.
    WorkerPanicked,
    /// This build target has no model runtime for the selected backend.
    UnsupportedBackend {
        /// Stable backend name.
        backend: &'static str,
    },
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenization(detail) => write!(formatter, "tokenization failed: {detail}"),
            Self::Mlx(detail) => write!(formatter, "MLX failed: {detail}"),
            Self::InvalidCacheLimit => formatter.write_str(
                "ZE_MLX_CACHE_LIMIT_BYTES must be a nonnegative integer byte count fitting usize",
            ),
            Self::CacheLimitConflict {
                configured,
                requested,
            } => write!(
                formatter,
                "MLX cache limit is already {configured} bytes; requested {requested} bytes"
            ),
            Self::MlxCache { operation, status } => {
                write!(formatter, "{operation} failed with status {status}")
            }
            Self::MissingTensor(name) => write!(formatter, "required tensor {name} is absent"),
            Self::Shape(detail) => write!(formatter, "model shape mismatch: {detail}"),
            Self::UnsupportedDtype(name) => {
                write!(formatter, "tensor {name} has an unsupported dtype")
            }
            Self::WorkerPanicked => formatter.write_str("embed worker panicked"),
            Self::UnsupportedBackend { backend } => write!(
                formatter,
                "the {backend} model runtime is not available for this build target"
            ),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// Swappable single-tower model runtime.
pub trait ModelRuntime {
    /// Evaluates one rectangular token batch.
    fn embed_batch(&mut self, tokens: &TokenBatch) -> Result<EmbeddingBatch, RuntimeError>;

    /// Evaluates a minimal warmup graph.
    fn warm(&mut self) -> Result<(), RuntimeError>;

    /// Returns the runtime identity used by the epoch declaration.
    fn identity(&self) -> RuntimeIdentity;
}

/// Model runtime backend selected for one open.
///
/// The selection is per open, never per process, so a forced selection cannot
/// leak into another open or a shared session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    /// MLX on the GPU, with Core ML for a compiled query tower.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    Apple,
    /// Portable CPU runtime. It is not implemented yet (WF-991).
    #[cfg_attr(
        all(target_os = "macos", target_arch = "aarch64", not(test)),
        expect(
            dead_code,
            reason = "only tests select the CPU backend on Apple silicon until WF-991"
        )
    )]
    Cpu,
}

impl Backend {
    /// Returns the backend that serves this build target.
    pub(crate) const fn target_default() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Self::Apple;
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        return Self::Cpu;
    }

    /// Returns the stable name used in diagnostics.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Apple => "apple",
            Self::Cpu => "cpu",
        }
    }

    /// Fails before any model I/O when this build cannot run the backend.
    pub(crate) fn available(self) -> Result<(), RuntimeError> {
        match self {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Self::Apple => Ok(()),
            // WF-991 makes the portable CPU runtime available.
            Self::Cpu => Err(RuntimeError::UnsupportedBackend {
                backend: self.name(),
            }),
        }
    }
}

/// CoreML runtime, including the Apple Neural Engine.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod coreml;
pub(crate) mod limits;
/// MLX GPU runtime.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod mlx;
pub(crate) mod mlx_cache;

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn embedding_batches_and_runtime_errors_preserve_their_typed_contracts() {
        assert!(EmbeddingBatch::new(Vec::new(), 0, 1).is_err());
        assert!(EmbeddingBatch::new(vec![1.0], 1, 0).is_err());
        assert!(EmbeddingBatch::new(vec![1.0], 2, 1).is_err());
        let batch = EmbeddingBatch::new(vec![1.0, 2.0], 1, 2).expect("valid batch");
        assert_eq!(batch.values(), [1.0, 2.0]);
        assert_eq!(batch.rows(), 1);
        assert_eq!(batch.dims(), 2);
        for error in [
            RuntimeError::Tokenization("tokens".to_owned()),
            RuntimeError::Mlx("mlx".to_owned()),
            RuntimeError::InvalidCacheLimit,
            RuntimeError::MlxCache {
                operation: "mlx_clear_cache",
                status: 1,
            },
            RuntimeError::MissingTensor("weight".to_owned()),
            RuntimeError::Shape("shape".to_owned()),
            RuntimeError::UnsupportedDtype("weight".to_owned()),
            RuntimeError::WorkerPanicked,
            RuntimeError::UnsupportedBackend { backend: "cpu" },
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn forced_cpu_selects_unavailable_backend() {
        use crate::{NamespaceModel, TextError};
        let unsupported = |result: Result<NamespaceModel, TextError>| {
            matches!(
                result,
                Err(TextError::Runtime(RuntimeError::UnsupportedBackend {
                    backend: "cpu"
                }))
            )
        };
        let root = tempfile::tempdir().expect("temporary root");
        let missing = root.path().join("fake-990-missing.zem");
        let tiny = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ze115/gte.zem"
        ));
        // The selection fails before the bundle is opened or a session starts.
        assert!(unsupported(NamespaceModel::open_bundle_on(
            &missing,
            Backend::Cpu
        )));
        assert!(unsupported(NamespaceModel::open_bundle_on(
            tiny,
            Backend::Cpu
        )));
        // The forced selection was per open: the default is still Apple.
        assert_eq!(Backend::target_default(), Backend::Apple);
        let model = NamespaceModel::open_bundle(tiny).expect("default Apple backend");
        drop(model);
    }
}
