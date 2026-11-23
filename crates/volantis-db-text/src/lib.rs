//! Text embedding and retrieval over Volantis DB stores.

#![deny(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::unwrap_used,
    unsafe_op_in_unsafe_fn
)]
#![warn(missing_docs)]

#[cfg(test)]
extern crate self as volantis_db_text;

/// Model architecture implementations.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod arch;
/// Immutable `.zem` bundle loading and validation.
pub mod bundle;
mod epoch;
mod error;
mod ingest;
mod model;
#[cfg(target_os = "macos")]
mod model_manifest;
mod query;
/// Model runtime implementations.
pub mod runtime;
mod tokenizer;
/// Tower metadata and token batches.
pub mod tower;

pub use error::TextError;
pub use ingest::{
    ChunkPolicy, IngestControl, IngestOptions, NamespaceModel, TextDocument, TextFaultSite,
    TextIngestReport, TextOpenOptions, TextStore,
};
pub use model::{
    DEFAULT_MODEL_MANIFEST_SHA256, DEFAULT_MODEL_REPO, DEFAULT_MODEL_REVISION, ModelError,
    ModelOpenOptions, ModelRef, resolve_model,
};
pub use query::{
    Legs, ObservedComputeUnits, QueryBackend, QueryOptions, TextHit, TextQueryOutcome,
    TextQueryTimings,
};
pub use runtime::mlx_cache::DEFAULT_MLX_CACHE_LIMIT_BYTES;
pub use volantis_db::lifecycle::SearchTier;
pub use volantis_db::tier::{
    MaintenanceBudget, MaintenanceReport, MaintenanceStatus, SegmentTier,
};
