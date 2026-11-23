//! Pinned Hugging Face model resolution and verified, process-safe caching.

use std::path::PathBuf;

/// Published default model repository.
pub const DEFAULT_MODEL_REPO: &str = "volantis-labs/volantis-db-v98-e5-small";
/// Immutable commit of the published default model.
pub const DEFAULT_MODEL_REVISION: &str = "5909e5a5985f1c727766f7b50a713cfbe043256c";
/// SHA-256 of the default model's exact manifest bytes.
pub const DEFAULT_MODEL_MANIFEST_SHA256: &str =
    "bc60376c0cddb821120a70aa17bc9f3c553726843dccfbefbc5664beda8fb941";

/// Immutable model identity. No moving branch names or tags are accepted.
#[derive(Clone, Debug)]
pub struct ModelRef {
    /// Hugging Face `owner/repository`.
    pub repo: String,
    /// Full immutable 40-hex commit SHA.
    pub revision: String,
    /// SHA-256 of the exact `volantis-model.json` bytes, as 64 hex characters.
    pub manifest_sha256: String,
}

impl ModelRef {
    /// The published, immutable default model pair.
    #[must_use]
    pub fn default_model() -> Self {
        Self {
            repo: DEFAULT_MODEL_REPO.to_owned(),
            revision: DEFAULT_MODEL_REVISION.to_owned(),
            manifest_sha256: DEFAULT_MODEL_MANIFEST_SHA256.to_owned(),
        }
    }
}

/// Download/cache policy plus the existing text-store open policy.
#[derive(Default)]
pub struct ModelOpenOptions<'a> {
    /// Existing explicit-path store options.
    pub text: crate::TextOpenOptions,
    /// Cache root; wins over `ZE_MODEL_CACHE` and the macOS default.
    pub cache_dir: Option<PathBuf>,
    /// Never use the network. `ZE_MODEL_OFFLINE=1` also enables this.
    pub offline: bool,
    /// Endpoint override; otherwise `ZE_MODEL_ENDPOINT` or HF HTTPS.
    /// Plain HTTP additionally requires `allow_http` or `ZE_MODEL_ALLOW_HTTP=1`.
    pub endpoint: Option<String>,
    /// Permit plain HTTP for local test fixtures. `ZE_MODEL_ALLOW_HTTP=1` also
    /// enables this. Tokens are never sent over HTTP.
    pub allow_http: bool,
    /// Authorize HF_TOKEN on HTTPS requests to the endpoint's exact host.
    /// Default false: only HTTPS huggingface.co requests may receive the token.
    /// This opt-in never authorizes HTTP or other redirect hosts.
    pub send_token_to_endpoint: bool,
    /// Synchronous byte progress on the calling thread: downloaded bytes / total.
    /// Intermediate reports require both 64 KiB and 100 ms since the last report;
    /// each verified file also reports completion. Total is the manifest file sizes.
    /// Initially (0, 0) means downloading the manifest. Must not reenter resolution.
    pub progress: Option<&'a dyn Fn(u64, u64)>,
}

/// Typed model resolution failure; invalid bytes are never opened as a model.
#[derive(Debug)]
pub enum ModelError {
    /// No complete entry was found while offline.
    OfflineMiss,
    /// A reference, option, or manifest violates the model contract.
    Invalid(String),
    /// A file's size or SHA-256 does not match its pin.
    Verification(String),
    /// HTTP or transport failure.
    Download(String),
    /// Cache filesystem operation failed.
    Io(std::io::Error),
    /// This build target has no model resolver (macOS only).
    UnsupportedBackend,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OfflineMiss => f.write_str("model is not cached (offline)"),
            Self::Invalid(s) => write!(f, "invalid model: {s}"),
            Self::Verification(s) => write!(f, "model verification failed: {s}"),
            Self::Download(s) => write!(f, "model download failed: {s}"),
            Self::Io(e) => e.fmt(f),
            Self::UnsupportedBackend => {
                f.write_str("model resolution is not available for this build target")
            }
        }
    }
}
impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// The verified cache, its file locks and the Foundation downloader are macOS only.
#[cfg(target_os = "macos")]
mod resolve;
#[cfg(target_os = "macos")]
pub use resolve::resolve_model;
#[cfg(target_os = "macos")]
pub(crate) use resolve::{VERIFIED, valid_path, valid_sha};

/// Resolves a pinned model into the verified cache. This target has no
/// resolver, so every call fails typed before any filesystem or network work.
#[cfg(not(target_os = "macos"))]
pub fn resolve_model(
    _model: &ModelRef,
    _options: &ModelOpenOptions<'_>,
) -> Result<PathBuf, ModelError> {
    Err(ModelError::UnsupportedBackend)
}
