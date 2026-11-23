//! Bundle-backed towers for an ordinary Store.
use super::*;
use volantis_db::epoch::{EmbedderError, EmbedderFailure, EmbeddingEpoch};
use volantis_db::lifecycle::embedded::{EmbeddedChunk, NamespaceEmbedder};

use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum QueryArtifact {
    Mlx,
    PinnedSibling,
    External(PathBuf),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SessionKey(String, String, String, QueryArtifact);

#[cfg(any(test, feature = "test-support"))]
thread_local! { static STARTS: std::cell::Cell<usize> = const {std::cell::Cell::new(0)}; }

#[cfg(test)]
thread_local! {
    static AFTER_JOIN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const {std::cell::RefCell::new(None)};
}

struct Session {
    bundle: Arc<Bundle>,
    runtime: RuntimeClient,
    epoch: EmbeddingEpoch,
}
#[derive(Default)]
struct Registry {
    pinned: BTreeMap<SessionKey, Arc<Session>>,
    private: usize,
    pending: Option<TextError>,
}
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}
fn registry_error() -> TextError {
    TextError::Pipeline {
        stage: "namespace lifecycle",
        detail: "registry poisoned".into(),
    }
}

/// Lease on a verified model runtime. Snapshots share the enclosing embedder.
pub struct NamespaceModel {
    session: Option<Arc<Session>>,
    key: Option<SessionKey>,
    epoch: EmbeddingEpoch,
}

impl NamespaceModel {
    /// Number of namespace workers constructed by this thread (test observation).
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn worker_starts() -> usize {
        STARTS.with(std::cell::Cell::get)
    }

    /// Observes actual document tower loads for lifecycle tests.
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn document_load_count(&self) -> Result<usize, TextError> {
        Ok(self
            .session()?
            .runtime
            .document_loads
            .load(Ordering::Relaxed))
    }

    /// Opens an explicit local bundle with a private runtime.
    pub fn open_bundle(path: &Path) -> Result<Self, TextError> {
        Self::acquire(path, discover_query_coreml(path), None)
    }

    /// Validates the requested cache entry before looking up its session.
    pub fn open_model(
        model: &crate::ModelRef,
        options: &crate::ModelOpenOptions<'_>,
    ) -> Result<Self, TextError> {
        let path = crate::resolve_model(model, options).map_err(TextError::ModelResolution)?;
        crate::runtime::mlx_cache::configure(options.text.mlx_cache_limit_bytes)?;
        let (query, artifact) = match discover_query_coreml(&path) {
            None => (None, QueryArtifact::Mlx),
            Some(query) if query == path.with_extension("mlmodelc") => {
                (Some(query), QueryArtifact::PinnedSibling)
            }
            Some(query) => {
                let resolved = query.canonicalize().map_err(|error| TextError::Pipeline {
                    stage: "resolve query backend",
                    detail: error.to_string(),
                })?;
                (Some(resolved.clone()), QueryArtifact::External(resolved))
            }
        };
        let key = SessionKey(
            model.repo.clone(),
            model.revision.clone(),
            model.manifest_sha256.clone(),
            artifact,
        );
        Self::acquire(&path, query, Some(key))
    }

    fn acquire(
        path: &Path,
        query: Option<PathBuf>,
        key: Option<SessionKey>,
    ) -> Result<Self, TextError> {
        let mut registry = registry().lock().map_err(|_| registry_error())?;
        if let Some(error) = registry.pending.take() {
            return Err(error);
        }
        let session = if let Some(session) = key.as_ref().and_then(|k| registry.pinned.get(k)) {
            session.clone()
        } else {
            let bundle = Arc::new(Bundle::open(path)?);
            let mut epoch = embedding_epoch(&bundle);
            if let Some(SessionKey(repo, revision, manifest, _)) = &key {
                // Length framing preserves the bundle's alignment digest and prevents
                // a different repository, revision or manifest from reopening the rows.
                let mut bytes = b"ZEMREF01".to_vec();
                for field in [
                    repo.as_bytes(),
                    revision.as_bytes(),
                    manifest.as_bytes(),
                    &epoch.alignment_digest,
                ] {
                    bytes.extend_from_slice(
                        &u32::try_from(field.len())
                            .map_err(|_| TextError::InvalidInput("model identity too long"))?
                            .to_le_bytes(),
                    );
                    bytes.extend_from_slice(field);
                }

                epoch.alignment_digest = bytes;
            } else {
                crate::runtime::mlx_cache::configure(None)?;
            }
            let runtime = RuntimeClient::start(bundle.clone(), query)?;
            #[cfg(any(test, feature = "test-support"))]
            STARTS.with(|count| count.set(count.get() + 1));
            let session = Arc::new(Session {
                bundle,
                runtime,
                epoch,
            });
            if let Some(key) = &key {
                registry.pinned.insert(key.clone(), session.clone());
            } else {
                registry.private += 1;
            }
            session
        };
        Ok(Self {
            epoch: session.epoch.clone(),
            session: Some(session),
            key,
        })
    }

    fn session(&self) -> Result<&Session, TextError> {
        self.session
            .as_deref()
            .ok_or(TextError::InvalidInput("released namespace session"))
    }

    fn retire(&mut self, deferred: bool) -> Result<(), TextError> {
        if self.session.is_none() {
            return Ok(());
        }
        let mut registry = registry().lock().map_err(|_| registry_error())?;
        let Some(session) = self.session.take() else {
            return Ok(());
        };
        if let Some(key) = &self.key {
            // The registry and this lease each hold one Arc. Every other lease
            // acquires and drops its Arc under this same lifecycle lock.
            if Arc::strong_count(&session) > 2 {
                return Ok(());
            }
            registry.pinned.remove(key);
        } else {
            registry.private -= 1;
        }
        let result = session.runtime.close();
        drop(session);
        #[cfg(test)]
        AFTER_JOIN.with(|hook| {
            if let Some(observe) = hook.take() {
                observe();
            }
        });
        let clear = if registry.private == 0 && registry.pinned.is_empty() {
            crate::runtime::mlx_cache::clear().map_err(TextError::from)
        } else {
            Ok(())
        };
        let result = result.and(clear);
        if deferred && let Err(error) = result {
            registry.pending.get_or_insert(error);
            return Ok(());
        }
        result
    }
}
impl Drop for NamespaceModel {
    fn drop(&mut self) {
        let _ = self.retire(true);
    }
}
impl Session {
    fn evaluate(&self, role: TowerRole, tokens: TokenBatch) -> Result<Vec<f32>, TextError> {
        let mut batch = self.runtime.embed(role, tokens, false)?;
        let tower = match role {
            TowerRole::Document => self.bundle.document_tower(),
            TowerRole::Query => self.bundle.query_tower(),
        };
        normalize_batch(&mut batch, &tower.embedding)?;
        Ok(batch.into_values())
    }

    fn visit_chunks(
        &self,
        text: &str,
        mut accept: impl FnMut(std::ops::Range<usize>, Vec<u32>) -> Result<(), TextError>,
    ) -> Result<(), TextError> {
        let tower = &self.bundle.document_tower().embedding;
        let limit = tower.max_tokens as usize;
        let tokenizer = self.bundle.tokenizer(TowerRole::Document);
        let mut start = 0;
        loop {
            let rest = text
                .get(start..)
                .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
            let mut end = rest
                .char_indices()
                .nth(limit.saturating_mul(8))
                .map_or(rest.len(), |(i, _)| i);
            let tokens = loop {
                let body = rest
                    .get(..end)
                    .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
                let input = format!("{}{body}", tower.prompt_prefix);
                // One extra slot detects truncation, including prompt and specials.
                let tokens = tokenizer.encode_row(
                    &input,
                    limit
                        .checked_add(1)
                        .ok_or(TextError::InvalidInput("token limit overflow"))?,
                );
                if tokens.len() <= limit {
                    break tokens;
                }
                let midpoint = body.floor_char_boundary(end / 2);
                let next = body
                    .get(..midpoint)
                    .and_then(|s| s.rfind(char::is_whitespace).filter(|i| *i > 0))
                    .unwrap_or(midpoint);
                if next == 0 {
                    return Err(TextError::InvalidInput(
                        "source character and prefix exceed tower window",
                    ));
                }
                end = next;
            };
            let next = start
                .checked_add(end)
                .ok_or(TextError::InvalidInput("chunk offset overflow"))?;
            accept(start..next, tokens)?;
            start = next;
            if start == text.len() {
                break;
            }
        }
        Ok(())
    }
    fn batch_chunks(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, TextError> {
        let mut window = Vec::with_capacity(MAX_EVAL_ROWS);
        let mut output = (0..texts.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (source, text) in texts.iter().enumerate() {
            self.visit_chunks(text, |bytes, tokens| {
                let chunks = output
                    .get_mut(source)
                    .ok_or(TextError::InvalidInput("missing source"))?;
                window.push(PendingChunk {
                    source,
                    ordinal: chunks.len(),
                    tokens,
                });
                chunks.push(EmbeddedChunk {
                    bytes,
                    vector: Vec::new(),
                });
                if window.len() == MAX_EVAL_ROWS {
                    self.flush(&mut window, &mut output)?;
                }
                Ok(())
            })?;
        }
        self.flush(&mut window, &mut output)?;
        Ok(output)
    }

    fn flush(
        &self,
        window: &mut Vec<PendingChunk>,
        output: &mut [Vec<EmbeddedChunk>],
    ) -> Result<(), TextError> {
        let tokenizer = self.bundle.tokenizer(TowerRole::Document);
        submit_groups(window, |group| {
            let tokens = tokenizer.pack_rows(group.iter().map(|chunk| chunk.tokens.as_slice()))?;
            let mut batch = self.runtime.embed(TowerRole::Document, tokens, false)?;
            normalize_batch(&mut batch, &self.bundle.document_tower().embedding)?;
            for (chunk, row) in group.iter().zip(embedding_rows(&batch, group.len())?) {
                let slot = output
                    .get_mut(chunk.source)
                    .and_then(|chunks| chunks.get_mut(chunk.ordinal))
                    .filter(|slot| slot.vector.is_empty())
                    .ok_or(TextError::InvalidInput("invalid chunk position"))?;
                slot.vector = row.to_vec();
            }
            Ok(())
        })?;
        window.clear();
        Ok(())
    }
}

use crate::runtime::mlx::{MAX_EVAL_ROWS, fits_eval};

struct PendingChunk {
    source: usize,
    ordinal: usize,
    tokens: Vec<u32>,
}

fn submit_groups(
    window: &mut [PendingChunk],
    mut submit: impl FnMut(&[PendingChunk]) -> Result<(), TextError>,
) -> Result<(), TextError> {
    window.sort_unstable_by_key(|chunk| (chunk.tokens.len(), chunk.source, chunk.ordinal));
    let mut pending = &*window;
    while !pending.is_empty() {
        let count = pending
            .iter()
            .enumerate()
            .take_while(|(index, chunk)| fits_eval(index + 1, chunk.tokens.len()))
            .count();
        let (group, tail) = pending
            .split_at_checked(count)
            .ok_or(TextError::InvalidInput("invalid group size"))?;
        submit(group)?;
        pending = tail;
    }
    Ok(())
}

impl NamespaceEmbedder for NamespaceModel {
    fn release(self: Arc<Self>) -> Result<(), EmbedderError> {
        if let Some(mut model) = Arc::into_inner(self) {
            model
                .retire(false)
                .map_err(|e| EmbedderFailure::new(e.to_string()))?;
        }
        Ok(())
    }
    fn epoch(&self) -> EmbeddingEpoch {
        self.epoch.clone()
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        self.documents(&[text])?
            .pop()
            .ok_or_else(|| EmbedderFailure::new("missing document").into())
    }
    fn documents(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, EmbedderError> {
        self.session()
            .and_then(|session| session.batch_chunks(texts))
            .map_err(|e| EmbedderFailure::new(e.to_string()).into())
    }
    fn query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.session()
            .and_then(|session| {
                session.evaluate(TowerRole::Query, session.bundle.tokenize_query(text)?)
            })
            .map_err(|e| EmbedderFailure::new(e.to_string()).into())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    mod fixture {
        include!("../../tests/common/namespace_fixture.rs");
    }
    use fixture::pinned_fixture;
    #[test]
    fn namespace_session_overlapping_opens() {
        let root = tempfile::tempdir().expect("root");
        let pin = pinned_fixture(root.path());
        let barrier = std::sync::Barrier::new(2);
        let [(a, bundles_a, starts_a), (b, bundles_b, starts_b)] = std::thread::scope(|scope| {
            let workers = [0, 1].map(|_| {
                scope.spawn(|| {
                    let opts = crate::ModelOpenOptions {
                        cache_dir: Some(root.path().into()),
                        offline: true,
                        ..Default::default()
                    };
                    barrier.wait();
                    let model = NamespaceModel::open_model(&pin, &opts).expect("open");
                    (model, crate::bundle::BUNDLE_OPENS.get(), STARTS.get())
                })
            });
            workers.map(|w| w.join().expect("thread"))
        });
        a.document("fixture copper kite").expect("document");
        b.document("fixture violet spool").expect("document");
        a.query("kite").expect("query");
        let loads = [&a, &b]
            .into_iter()
            .map(|m| {
                (
                    Arc::as_ptr(&m.session().expect("session").runtime.document_loads),
                    m.session()
                        .expect("session")
                        .runtime
                        .document_loads
                        .load(Ordering::Relaxed),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            loads.values().sum::<usize>(),
            1,
            "document tower loads across workers"
        );
        assert_eq!((bundles_a + bundles_b, starts_a + starts_b), (1, 1));
        assert!(Arc::ptr_eq(
            &a.session().expect("session").bundle,
            &b.session().expect("session").bundle
        ));
    }

    fn offline(root: &Path) -> crate::ModelOpenOptions<'static> {
        crate::ModelOpenOptions {
            offline: true,
            cache_dir: Some(root.to_owned()),
            ..Default::default()
        }
    }

    fn digest(path: &Path) -> String {
        let hash = std::process::Command::new("shasum")
            .args(["-a", "256"])
            .arg(path)
            .output()
            .expect("hash fixture");
        assert!(hash.status.success());
        String::from_utf8(hash.stdout)
            .expect("output")
            .split_whitespace()
            .next()
            .expect("digest")
            .to_owned()
    }

    #[test]
    fn namespace_session_paired_lazy_load() {
        use super::lazy_document_fixtures::{FixtureTower, write_fixture_bundle};
        let root = tempfile::tempdir().expect("root");
        let mut pin = pinned_fixture(root.path());
        let entry = root.path().join("fixture%2Ffake-ze598").join(&pin.revision);
        let bundle = entry.join("tiny.zem");
        write_fixture_bundle(
            &bundle,
            &[
                FixtureTower::document("document"),
                FixtureTower::query("query"),
            ],
            b"pair",
        );
        let manifest = entry.join("volantis-model.json");
        let hash = digest(&bundle);
        let len = bundle.metadata().expect("size").len();
        std::fs::write(&manifest,format!(r#"{{"format":1,"name":"tiny","bundle":"tiny.zem","query_coreml":null,"files":[{{"path":"tiny.zem","size":{len},"sha256":"{hash}"}}]}}"#)).expect("manifest");
        pin.manifest_sha256 = digest(&manifest);
        let opts = offline(root.path());
        let a = NamespaceModel::open_model(&pin, &opts).expect("first");
        let b = NamespaceModel::open_model(&pin, &opts).expect("peer");
        a.query("bronze").expect("query");
        b.query("zeppelin").expect("query");
        let loads = &a.session().expect("session").runtime.document_loads;
        assert_eq!(loads.load(Ordering::Relaxed), 0);
        b.document("fixture violet spool").expect("first write");
        a.document("fixture copper kite").expect("peer write");
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert!(Arc::ptr_eq(
            loads,
            &b.session().expect("peer session").runtime.document_loads
        ));
    }

    #[test]
    fn namespace_session_pin_and_policy() {
        let root = tempfile::tempdir().expect("root");
        let pin = pinned_fixture(root.path());
        let opts = offline(root.path());
        let first = NamespaceModel::open_model(&pin, &opts).expect("first");
        let other = tempfile::tempdir().expect("other root");
        let mut other_opts = offline(other.path());
        let open = NamespaceModel::open_model;
        assert!(matches!(
            open(&pin, &other_opts),
            Err(TextError::ModelResolution(_))
        ));
        pinned_fixture(other.path());
        let entry = other
            .path()
            .join("fixture%2Ffake-ze598")
            .join(&pin.revision);
        std::fs::write(entry.join("tiny.zem"), b"corrupt").expect("damage copy");
        assert!(matches!(
            open(&pin, &other_opts),
            Err(TextError::ModelResolution(_))
        ));
        pinned_fixture(other.path());
        let repaired = open(&pin, &other_opts).expect("repaired");
        assert!(std::ptr::eq(
            first.session().expect("first"),
            repaired.session().expect("hit")
        ));
        other_opts.text.mlx_cache_limit_bytes = Some(213598);
        if open(&pin, &other_opts).is_ok() {
            other_opts.text.mlx_cache_limit_bytes = Some(213599);
        }
        assert!(matches!(
            open(&pin, &other_opts),
            Err(TextError::Runtime(RuntimeError::CacheLimitConflict { .. }))
        ));
        let other_opts = offline(other.path());
        for variant in 0..3 {
            let mut different = pin.clone();
            match variant {
                0 => different.repo = "fixture/other".into(),
                1 => different.revision = "2".repeat(40),
                _ => {}
            }
            let target = other
                .path()
                .join(different.repo.replace('/', "%2F"))
                .join(&different.revision);
            if target != entry {
                std::fs::create_dir_all(&target).expect("variant");
                for file in ["tiny.zem", "volantis-model.json"] {
                    std::fs::copy(entry.join(file), target.join(file)).expect("copy fixture");
                }
            } else {
                let manifest = target.join("volantis-model.json");
                let mut bytes = std::fs::read(&manifest).expect("manifest");
                bytes.push(b' ');
                std::fs::write(&manifest, bytes).expect("whitespace");
                different.manifest_sha256 = digest(&manifest);
            }
            let peer = open(&different, &other_opts).expect("other pin");
            assert_ne!(peer.epoch(), first.epoch());
            assert!(!std::ptr::eq(
                peer.session().expect("other"),
                first.session().expect("first")
            ));
        }
        let mut framed = b"ZEMREF01".to_vec();
        for part in [
            pin.repo.as_bytes(),
            pin.revision.as_bytes(),
            pin.manifest_sha256.as_bytes(),
            tiny().epoch().alignment_digest.as_slice(),
        ] {
            framed.extend_from_slice(&(part.len() as u32).to_le_bytes());
            framed.extend_from_slice(part);
        }
        assert_eq!(first.epoch().alignment_digest, framed);
    }

    #[test]
    fn namespace_session_final_release_order() {
        use crate::runtime::mlx_cache::{CLEAR_CALLS, FAIL_CLEAR};
        const CHILD: &str = "ZE598_RELEASE_TEST";
        if std::env::var_os(CHILD).is_none() {
            let name = std::thread::current().name().expect("test name").to_owned();
            let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", &name, "--nocapture"])
                .env(CHILD, "1")
                .env_remove("ZE_QUERY_COREML")
                .status()
                .expect("isolated test");
            assert!(status.success());
            return;
        }
        let root = tempfile::tempdir().expect("root");
        let pin = pinned_fixture(root.path());
        let mut other = pin.clone();
        other.revision = "2".repeat(40);
        let repo = root.path().join("fixture%2Ffake-ze598");
        let src = repo.join(&pin.revision);
        let dest = repo.join(&other.revision);
        std::fs::create_dir(&dest).expect("entry");
        for file in ["tiny.zem", "volantis-model.json"] {
            std::fs::copy(src.join(file), dest.join(file)).expect("copy");
        }
        let open = |pin| NamespaceModel::open_model(pin, &offline(root.path())).expect("model");
        let observe_next_exit = || {
            let exits = Arc::new(AtomicUsize::new(0));
            OBSERVE_WORKER_EXIT.set(Some(exits.clone()));
            exits
        };
        let first_exits = observe_next_exit();
        let a = open(&pin);
        let second_exits = observe_next_exit();
        let b = open(&other);
        let worker = |model: &NamespaceModel| {
            Arc::downgrade(&model.session().expect("session").runtime.document_loads)
        };
        let first_worker = worker(&a);
        drop(a);
        assert_eq!(first_exits.load(Ordering::Acquire), 1);
        assert_eq!(second_exits.load(Ordering::Acquire), 0);
        assert!(first_worker.upgrade().is_none());
        assert_eq!(CLEAR_CALLS.get(), 0);
        let before = worker(&b);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let at_join = barrier.clone();
        AFTER_JOIN.with(|hook| {
            hook.replace(Some(Box::new(move || {
                assert_eq!(second_exits.load(Ordering::Acquire), 1);
                assert!(before.upgrade().is_none());
                assert!(matches!(
                    registry().try_lock(),
                    Err(std::sync::TryLockError::WouldBlock)
                ));
                at_join.wait();
                assert_eq!(CLEAR_CALLS.get(), 0);
            })))
        });
        let next = std::thread::scope(|s| {
            let racer = s.spawn(|| {
                barrier.wait();
                open(&pin)
            });
            drop(b);
            racer.join().expect("reopen")
        });
        assert_eq!(CLEAR_CALLS.get(), 1);
        FAIL_CLEAR.set(true);
        drop(next);
        assert!(matches!(
            NamespaceModel::open_model(&pin, &offline(root.path())),
            Err(TextError::Runtime(RuntimeError::MlxCache {
                status: -598,
                ..
            }))
        ));
        let writer = Store::open_embedded(
            root.path().join("store"),
            OpenOptions::new(),
            Arc::new(open(&pin)),
        )
        .expect("writer");
        FAIL_CLEAR.set(true);
        assert!(matches!(writer.close(), Err(StoreError::Embedding(_))));
        writer.close().expect("repeat close");
        let private_path = root.path().join("private.zem");
        std::fs::copy(src.join("tiny.zem"), &private_path).expect("bundle");
        let invalid = private_path.with_extension("mlmodelc");
        std::fs::create_dir(&invalid).expect("invalid CoreML artifact");
        let failed_exits = observe_next_exit();
        assert!(NamespaceModel::open_bundle(&private_path).is_err());
        assert_eq!(failed_exits.load(Ordering::Acquire), 1);
        std::fs::remove_dir(invalid).expect("remove invalid artifact");
        let private = NamespaceModel::open_bundle(&private_path).expect("retry");
        let another = NamespaceModel::open_bundle(&private_path).expect("second private bundle");
        assert!(!std::ptr::eq(
            private.session().expect("one"),
            another.session().expect("two")
        ));
        let pinned = open(&pin);
        drop(another);
        drop(pinned);
        assert_eq!(CLEAR_CALLS.get(), 3);
        drop(private);
        assert_eq!(CLEAR_CALLS.get(), 4);
    }
    fn singleton_reference(
        model: &NamespaceModel,
        text: &str,
    ) -> Result<Vec<EmbeddedChunk>, TextError> {
        let tower = &model
            .session()
            .expect("session")
            .bundle
            .document_tower()
            .embedding;
        let limit = tower.max_tokens as usize;
        let tokenizer = model
            .session()
            .expect("session")
            .bundle
            .tokenizer(TowerRole::Document);
        let mut start = 0;
        let mut chunks = Vec::new();
        loop {
            let rest = text
                .get(start..)
                .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
            let mut end = rest
                .char_indices()
                .nth(limit.saturating_mul(8))
                .map_or(rest.len(), |(i, _)| i);
            let tokens = loop {
                let body = rest
                    .get(..end)
                    .ok_or(TextError::InvalidInput("invalid chunk boundary"))?;
                let input = format!("{}{body}", tower.prompt_prefix);
                // One extra slot detects truncation, including prompt and specials.
                let tokens = tokenizer.encode_batch(&[input], limit + 1)?;
                if tokens.token_ids.len() <= limit {
                    break tokens;
                }
                let midpoint = body.floor_char_boundary(end / 2);
                let next = body
                    .get(..midpoint)
                    .and_then(|s| s.rfind(char::is_whitespace).filter(|i| *i > 0))
                    .unwrap_or(midpoint);
                if next == 0 {
                    return Err(TextError::InvalidInput(
                        "source character and prefix exceed tower window",
                    ));
                }
                end = next;
            };
            chunks.push(EmbeddedChunk {
                bytes: start..start + end,
                vector: model.session()?.evaluate(TowerRole::Document, tokens)?,
            });
            start += end;
            if start == text.len() {
                break;
            }
        }
        Ok(chunks)
    }
    fn tiny() -> NamespaceModel {
        NamespaceModel::open_bundle(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/ze115/gte.zem"
        )))
        .expect("tiny bundle")
    }
    #[test]
    fn ze597_short_upsert_batches_reach_runtime() {
        let root = tempfile::tempdir().expect("root");
        let model = Arc::new(tiny());
        let store =
            Store::open_embedded(root.path(), OpenOptions::new(), model.clone()).expect("store");
        for count in [32, 33] {
            let rows = (0..count)
                .map(|i| {
                    IngestDocument::new(
                        DocumentVersion::new(DocId::new(count * 100 + i), Revision::new(1)),
                        vec![],
                    )
                    .with_text(if i % 2 == 0 { "bronze" } else { "zeppelin" })
                })
                .collect();
            store.ingest(IngestBatch::new(rows)).expect("upsert");
            let shapes = std::mem::take(
                &mut *model
                    .session()
                    .expect("session")
                    .runtime
                    .observer
                    .lock()
                    .expect("observer"),
            );
            let mut expected = vec![(32, vec![32])];
            if count == 33 {
                expected.push((1, vec![1]));
            }
            assert_eq!(shapes, expected);
            let hit = IngestDocument::new(
                DocumentVersion::new(DocId::new(count * 100), Revision::new(2)),
                vec![],
            )
            .with_text("bronze");
            store.ingest(IngestBatch::new(vec![hit])).expect("reuse");
            assert!(matches!(
                store.ingest(IngestBatch::new(vec![])),
                Err(volantis_db::ingest::IngestError::EmptyBatch)
            ));
            assert!(
                model
                    .session()
                    .expect("session")
                    .runtime
                    .observer
                    .lock()
                    .expect("observer")
                    .is_empty()
            );
        }
    }

    #[test]
    fn ze597_ragged_chunks_match_singleton_reference() {
        let model = tiny();
        let long = "the bronze zeppelin ".repeat(200);
        let texts = ["bronze", &long, "🧪 bronze 雪 zeppelin", "", "zeppelin"];
        let results = model.documents(&texts).expect("batch");
        assert_eq!(results.len(), texts.len());
        assert!(results[1].len() > 32);
        let references = texts.map(|text| singleton_reference(&model, text).expect("reference"));
        assert_ne!(references[0][0].vector, references[4][0].vector);
        for (chunks, reference) in results.iter().zip(references) {
            assert_eq!(chunks.len(), reference.len());
            for (chunk, single) in chunks.iter().zip(reference) {
                assert_eq!(chunk.bytes, single.bytes);
                assert_eq!(chunk.vector.len(), single.vector.len());
                assert!(
                    chunk
                        .vector
                        .iter()
                        .zip(single.vector)
                        .all(|(a, b)| (a - b).abs() <= 0.00002)
                );
            }
        }
    }

    #[test]
    fn ze597_length_groups_obey_eval_budget() {
        let model = tiny();
        let tokenizer = model
            .session()
            .expect("session")
            .bundle
            .tokenizer(TowerRole::Document);
        let mut rows = (0..32)
            .map(|source| PendingChunk {
                source,
                ordinal: 0,
                tokens: vec![
                    source as u32 + 1;
                    if source == 31 {
                        4097
                    } else if source % 2 == 0 {
                        4
                    } else {
                        256
                    }
                ],
            })
            .collect::<Vec<_>>();
        let mut shapes = Vec::new();
        let mut seen = Vec::new();
        submit_groups(&mut rows, |group| {
            let batch = tokenizer.pack_rows(group.iter().map(|chunk| chunk.tokens.as_slice()))?;
            shapes.push((batch.rows, batch.tokens_per_row));
            for (i, chunk) in group.iter().enumerate() {
                seen.push(chunk.source);
                for col in 0..batch.tokens_per_row {
                    let pos = i * batch.tokens_per_row + col;
                    let active = col < chunk.tokens.len();
                    assert_eq!(
                        batch.token_ids[pos],
                        if active { chunk.source as i32 + 1 } else { 0 }
                    );
                    assert_eq!(batch.attention_mask[pos], if active { 1. } else { 0. });
                }
            }
            Ok(())
        })
        .expect("groups");
        assert_eq!(shapes, [(16, 4), (15, 256), (1, 4097)]);
        seen.sort_unstable();
        assert_eq!(seen, (0..32).collect::<Vec<_>>());
        // A mixed-width group exercises the real bundle's padding and attention mask.
        let packed = tokenizer
            .pack_rows([&[3, 2][..], &[2][..]].into_iter())
            .expect("pad");
        assert_eq!(packed.token_ids, [3, 2, 2, 0]);
        assert_eq!(packed.attention_mask, [1., 1., 1., 0.]);
        let mut calls = 0;
        let error = submit_groups(&mut rows, |_| {
            calls += 1;
            if calls == 2 {
                return Err(TextError::Pipeline {
                    stage: "embed submit",
                    detail: "runtime is closed".into(),
                });
            }
            Ok(())
        })
        .expect_err("closed runtime");
        assert!(matches!(
            error,
            TextError::Pipeline {
                stage: "embed submit",
                ..
            }
        ));
        assert_eq!(calls, 2);
    }

    #[test]
    #[ignore = "requires cached release model"]
    fn ze597_cached_model_batch_parity() {
        let path = std::env::var_os("ZE597_BUNDLE").expect("ZE597_BUNDLE");
        let model = NamespaceModel::open_bundle(Path::new(&path)).expect("cached bundle");
        assert_eq!(model.epoch.document.normalization, Normalization::L2);
        let long = "the bronze zeppelin ".repeat(190);
        let texts = ["FAKE bronze", "🧪 bronze 雪 zeppelin", &long, "zeppelin"];
        let chunks = model.documents(&texts).expect("batch");
        assert_eq!(chunks.len(), texts.len());
        let mut minimum = 1_f64;
        for (parent, text) in chunks.iter().zip(texts) {
            let reference = singleton_reference(&model, text).expect("reference");
            assert_eq!(parent.len(), reference.len());
            for (chunk, single) in parent.iter().zip(reference) {
                assert_eq!(chunk.bytes, single.bytes);
                assert_eq!(chunk.vector.len(), single.vector.len());
                let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
                let a = norm(&chunk.vector);
                let b = norm(&single.vector);
                assert!((a - 1.).abs() < 1e-5 && (b - 1.).abs() < 1e-5);
                let cosine = chunk
                    .vector
                    .iter()
                    .zip(single.vector)
                    .map(|(x, y)| f64::from(*x) * f64::from(y))
                    .sum::<f64>()
                    / (a * b);
                assert!(cosine.is_finite());
                minimum = minimum.min(cosine);
            }
        }
        println!("minimum cosine {minimum:.10}");
        assert!(minimum >= 0.99999);
    }
}
