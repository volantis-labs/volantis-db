//! Background embedding of pending rows (WF-985).
//!
//! The durable pending rows are the work queue. The worker reads a bounded
//! batch at one generation, releases the read, runs the document tower with
//! no store lock held, then commits a conditional completion through the
//! normal writer path. A completion whose row changed in the meantime is
//! dropped, so the worker can lose a batch in memory without losing work.
use super::*;
use std::sync::{Condvar, Weak};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

/// Most rows in one document-tower call.
const BATCH_ROWS: usize = 32;
/// Most source bytes in one document-tower call; one longer row still runs alone.
const BATCH_BYTES: usize = 1 << 20;
/// Rows read per scan page while the worker looks for pending rows.
const PAGE_ROWS: usize = 256;
/// Attempts to read pending rows when writes keep moving the generation.
const STALE_RETRIES: usize = 16;

/// Last failure of the worker. The values are stable across the C ABI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum EmbeddingErrorCode {
    /// No failure since the last successful batch or resume.
    #[default]
    None = 0,
    /// The document tower returned an error.
    Model = 1,
    /// The document tower returned output that failed the ready-row checks.
    Output = 2,
    /// The store refused the read or the completion commit.
    Store = 3,
}

/// Point-in-time state of background embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingStatus {
    /// Admission of new batches is stopped.
    pub paused: bool,
    /// A batch is between admission and its completion commit.
    pub in_flight: bool,
    /// Live rows that wait for embedding at the current generation.
    pub pending: u64,
    /// Last failure; the worker waits for resume after a failure.
    pub last_error: EmbeddingErrorCode,
    /// A worker thread is attached to this handle.
    pub running: bool,
}

#[derive(Default)]
struct State {
    paused: bool,
    stop: bool,
    /// Incremented by each commit of pending rows and by resume.
    wake: u64,
    /// Value of `wake` at the start of the current or last batch.
    seen: u64,
    in_flight: bool,
    /// A failed batch waits for resume.
    parked: bool,
    /// The last scan found no pending row and nothing woke the worker since.
    idle: bool,
    last_error: EmbeddingErrorCode,
    thread: Option<ThreadId>,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

impl Shared {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>, StoreError> {
        self.state.lock().map_err(|_| StoreError::Synchronization {
            component: "embedding worker",
        })
    }
}

/// Worker control owned by one store handle.
pub(crate) struct EmbeddingWorker {
    shared: Arc<Shared>,
    handle: Mutex<Option<JoinHandle<()>>>,
    schedule: DocumentSchedule,
    writable: bool,
}

impl EmbeddingWorker {
    pub(crate) fn new(schedule: DocumentSchedule, writable: bool) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    paused: schedule == DocumentSchedule::Paused,
                    // Scan once at start: a reopened store may hold pending rows.
                    wake: 1,
                    ..State::default()
                }),
                changed: Condvar::new(),
            }),
            handle: Mutex::new(None),
            schedule,
            writable,
        }
    }

    pub(crate) const fn defers(&self) -> bool {
        !matches!(self.schedule, DocumentSchedule::Immediate)
    }

    /// Wakes the worker after a commit that wrote pending rows.
    pub(crate) fn notify(&self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.wake = state.wake.wrapping_add(1);
            state.idle = false;
            self.shared.changed.notify_all();
        }
    }

    /// Stops admission and joins the worker. The worker thread itself may
    /// drop the last store reference; it then exits without a self-join.
    pub(crate) fn stop(&self) -> Result<(), StoreError> {
        let worker_thread = {
            let mut state = self.shared.lock()?;
            state.stop = true;
            self.shared.changed.notify_all();
            state.thread
        };
        let handle = self
            .handle
            .lock()
            .map_err(|_| StoreError::Synchronization {
                component: "embedding worker",
            })?
            .take();
        match handle {
            Some(handle) if Some(std::thread::current().id()) != worker_thread => handle
                .join()
                .map_err(|_| StoreError::BackgroundThreadPanicked),
            Some(_) | None => Ok(()),
        }
    }
}

enum BatchOutcome {
    Idle,
    Progress,
}

fn run(store: Weak<Store>, shared: Arc<Shared>) {
    loop {
        {
            let Ok(mut state) = shared.state.lock() else {
                return;
            };
            loop {
                if state.stop {
                    return;
                }
                if !state.paused && !state.parked && state.wake != state.seen {
                    break;
                }
                let Ok(next) = shared.changed.wait(state) else {
                    return;
                };
                state = next;
            }
            // Admission: pause sets `paused` under this lock before it returns.
            state.seen = state.wake;
            state.idle = false;
            state.in_flight = true;
        }
        let outcome = match store.upgrade() {
            Some(store) => store.embed_pending_batch(),
            None => return,
        };
        let Ok(mut state) = shared.state.lock() else {
            return;
        };
        state.in_flight = false;
        match outcome {
            Ok(BatchOutcome::Idle) => state.idle = state.wake == state.seen,
            Ok(BatchOutcome::Progress) => {
                state.last_error = EmbeddingErrorCode::None;
                // More pending rows may remain: look again.
                state.wake = state.wake.wrapping_add(1);
            }
            Err(code) => {
                state.last_error = code;
                state.parked = true;
            }
        }
        shared.changed.notify_all();
    }
}

/// One pending row read for a batch.
struct PendingRow {
    version: crate::ingest::DocumentVersion,
    timestamp: i64,
    text: String,
    attributes: Vec<(ColumnId, PredicateValue)>,
    caller: Vec<u8>,
    /// Stored metadata the completion replaces; it holds the admission token.
    claim: Vec<u8>,
}

impl Store {
    /// Starts background embedding for this writable embedded namespace. The
    /// worker holds a weak reference, so dropping or closing the store stops
    /// it. A second call is a no-op.
    pub fn attach_embedding_worker(self: &Arc<Self>) -> Result<(), StoreError> {
        self.require_embedding_control()?;
        let mut handle = self
            .embedding
            .handle
            .lock()
            .map_err(|_| StoreError::Synchronization {
                component: "embedding worker",
            })?;
        if handle.is_some() {
            return Ok(());
        }
        if self.embedding.shared.lock()?.stop {
            return Err(StoreError::Closing);
        }
        let store = Arc::downgrade(self);
        let shared = Arc::clone(&self.embedding.shared);
        let spawned = std::thread::Builder::new()
            .name("ze-embedding".to_owned())
            .spawn(move || run(store, shared))
            .map_err(|source| StoreError::BackgroundStart { source })?;
        self.embedding.shared.lock()?.thread = Some(spawned.thread().id());
        *handle = Some(spawned);
        Ok(())
    }

    /// Stops admission of new document batches. A batch already admitted
    /// may still finish; `embedding_status` reports it as in flight.
    pub fn pause_embedding(&self) -> Result<(), StoreError> {
        self.require_embedding_control()?;
        self.embedding.shared.lock()?.paused = true;
        self.embedding.shared.changed.notify_all();
        Ok(())
    }

    /// Restarts admission and retries a batch that failed.
    pub fn resume_embedding(&self) -> Result<(), StoreError> {
        self.require_embedding_control()?;
        let mut state = self.embedding.shared.lock()?;
        state.paused = false;
        state.parked = false;
        state.idle = false;
        state.wake = state.wake.wrapping_add(1);
        self.embedding.shared.changed.notify_all();
        Ok(())
    }

    /// Reports worker state and the live pending rows.
    pub fn embedding_status(&self) -> Result<EmbeddingStatus, StoreError> {
        if !self.is_embedded() {
            return Err(StoreError::MissingEmbedder);
        }
        let pending = self.count_pending_rows()?;
        let running = self
            .embedding
            .handle
            .lock()
            .map_err(|_| StoreError::Synchronization {
                component: "embedding worker",
            })?
            .is_some();
        let state = self.embedding.shared.lock()?;
        Ok(EmbeddingStatus {
            paused: state.paused,
            in_flight: state.in_flight,
            pending,
            last_error: state.last_error,
            running,
        })
    }

    /// Waits until the worker has no admitted batch and either found no
    /// pending row, stopped on a failure, or is paused. Returns the status
    /// then, or after `timeout`.
    pub fn wait_embedding_idle(&self, timeout: Duration) -> Result<EmbeddingStatus, StoreError> {
        self.require_embedding_control()?;
        let deadline = Instant::now().checked_add(timeout);
        let mut state = self.embedding.shared.lock()?;
        loop {
            let settled = !state.in_flight
                && (state.parked
                    || state.paused
                    || state.stop
                    || (state.idle && state.wake == state.seen));
            let remaining =
                deadline.map_or(timeout, |end| end.saturating_duration_since(Instant::now()));
            if settled || remaining.is_zero() {
                break;
            }
            state = self
                .embedding
                .shared
                .changed
                .wait_timeout(state, remaining)
                .map_err(|_| StoreError::Synchronization {
                    component: "embedding worker",
                })?
                .0;
        }
        drop(state);
        self.embedding_status()
    }

    fn require_embedding_control(&self) -> Result<(), StoreError> {
        if !self.is_embedded() {
            return Err(StoreError::MissingEmbedder);
        }
        if !self.embedding.writable {
            return Err(StoreError::ReadOnly);
        }
        Ok(())
    }

    fn count_pending_rows(&self) -> Result<u64, StoreError> {
        let mut last = None;
        for _ in 0..STALE_RETRIES {
            match self.scan_pending(usize::MAX, usize::MAX) {
                Ok((count, _)) => return Ok(count),
                Err(QueryError::Store(error @ StoreError::ScanStale { .. })) => last = Some(error),
                Err(QueryError::Store(error)) => return Err(error),
                Err(error) => return Err(invalid(error.to_string())),
            }
        }
        Err(last.unwrap_or(StoreError::Closed))
    }

    /// Reads pending rows at one generation: the live pending count and up
    /// to `rows` rows or `bytes` source bytes of them.
    fn scan_pending(
        &self,
        rows: usize,
        bytes: usize,
    ) -> Result<(u64, Vec<PendingRow>), QueryError> {
        let control = QueryControl::Cancel(crate::lifecycle::CancelToken::new());
        let fields = DocumentFields::TEXT
            | DocumentFields::ATTRIBUTES
            | DocumentFields::METADATA
            | DocumentFields::CHUNKS;
        let mut cursor = None;
        let mut count = 0_u64;
        let mut batch = Vec::new();
        let mut used = 0_usize;
        loop {
            let mut request = DocumentScanRequest::new(PAGE_ROWS, fields, control.clone());
            if let Some(cursor) = cursor {
                request = request.with_cursor(cursor);
            }
            let page = self.scan_documents(request)?;
            for doc in page.documents {
                let row = decode_row(doc.metadata.as_deref().unwrap_or_default())
                    .map_err(QueryError::Store)?;
                if row.state != EmbeddedRowState::Pending {
                    continue;
                }
                count = count.saturating_add(1);
                let text = doc.text.clone().unwrap_or_default();
                let full = batch.len() >= rows
                    || (!batch.is_empty() && used.saturating_add(text.len()) > bytes);
                if full {
                    continue;
                }
                used = used.saturating_add(text.len());
                let caller = row.caller.to_vec();
                batch.push(PendingRow {
                    version: crate::ingest::DocumentVersion::new(doc.doc_id, doc.revision),
                    timestamp: doc.timestamp,
                    text,
                    attributes: doc.attributes.unwrap_or_default(),
                    caller,
                    claim: doc.metadata.unwrap_or_default(),
                });
            }
            cursor = page.continuation;
            if cursor.is_none() {
                return Ok((count, batch));
            }
        }
    }

    /// Embeds one bounded batch of pending rows and commits the completions
    /// that still match their live rows.
    fn embed_pending_batch(&self) -> Result<BatchOutcome, EmbeddingErrorCode> {
        let mut batch = None;
        for _ in 0..STALE_RETRIES {
            match self.scan_pending(BATCH_ROWS, BATCH_BYTES) {
                Ok((_, rows)) => {
                    batch = Some(rows);
                    break;
                }
                Err(QueryError::Store(StoreError::ScanStale { .. })) => {}
                Err(_) => return Err(EmbeddingErrorCode::Store),
            }
        }
        let rows = batch.ok_or(EmbeddingErrorCode::Store)?;
        if rows.is_empty() {
            return Ok(BatchOutcome::Idle);
        }
        let model = self
            .admit_embedder()
            .map_err(|_| EmbeddingErrorCode::Store)?;
        let dims = self
            .embedded_dims()
            .map_err(|_| EmbeddingErrorCode::Store)?;
        let epoch = self.epoch_identity().ok_or(EmbeddingErrorCode::Store)?;
        let texts = rows.iter().map(|row| row.text.as_str()).collect::<Vec<_>>();
        // No store lock, read pin or writer guard is held during inference.
        let encoded = model
            .documents(&texts)
            .map_err(|_| EmbeddingErrorCode::Model)?;
        drop(model);
        if encoded.len() != rows.len() {
            return Err(EmbeddingErrorCode::Output);
        }
        let mut documents = Vec::with_capacity(rows.len());
        let mut claims = Vec::with_capacity(rows.len());
        for (row, chunks) in rows.into_iter().zip(encoded) {
            let (vector, metadata) = ready_row(&row.text, &row.caller, &chunks, dims)
                .map_err(|_| EmbeddingErrorCode::Output)?;
            let mut document = IngestDocument::new(row.version, vector)
                .with_text(row.text)
                .with_timestamp(row.timestamp)
                .with_metadata(metadata);
            if !row.attributes.is_empty() {
                document = document.with_columns(row.attributes);
            }
            documents.push(document);
            claims.push(row.claim);
        }
        let batch = crate::ingest::IngestBatch::completion(documents, claims, epoch);
        self.ingest_prepared(&batch)
            .map_err(|_| EmbeddingErrorCode::Store)?;
        Ok(BatchOutcome::Progress)
    }
}
