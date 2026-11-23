use volantis_db::epoch::*;
use volantis_db::lifecycle::embedded::*;

pub struct Model;
impl NamespaceEmbedder for Model {
    fn epoch(&self) -> EmbeddingEpoch {
        let tower = EmbeddingTower {
            model_id: "fixture".into(),
            model_version: "1".into(),
            weights_digest: vec![1],
            dims: 2,
            normalization: Normalization::None,
            prompt_prefix: String::new(),
            max_tokens: 8,
            runtime: EmbeddingRuntime::CpuReference,
            compute_units: ComputeUnits::Cpu,
            os_build: None,
        };
        EmbeddingEpoch {
            document: tower.clone(),
            query: tower,
            alignment_digest: vec![],
        }
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        if text == "!fail" {
            return Err(EmbedderFailure::new("injected document embedding failure").into());
        }
        let mut start = 0;
        Ok(text
            .split('|')
            .map(|part| {
                let end = start + part.len();
                let chunk = EmbeddedChunk {
                    bytes: start..end,
                    vector: if part == "target" {
                        vec![1., 0.]
                    } else {
                        vec![0., 1.]
                    },
                };
                start = end + 1;
                chunk
            })
            .collect())
    }
    fn query(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(vec![1., 0.])
    }
}

#[allow(dead_code)] // Also included by test crates that only use Model.
#[derive(Default)]
pub struct CountingModel {
    pub documents: std::sync::atomic::AtomicUsize,
    pub different_epoch: bool,
    pub pause: Option<std::sync::Arc<std::sync::Barrier>>,
    pub paused: std::sync::atomic::AtomicBool,
    pub queries: std::sync::atomic::AtomicUsize,
}

impl NamespaceEmbedder for CountingModel {
    fn epoch(&self) -> EmbeddingEpoch {
        let mut epoch = Model.epoch();
        if self.different_epoch {
            epoch.document.weights_digest = vec![9];
        }
        epoch
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        self.documents
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if let Some(barrier) = &self.pause
            && text == "pause"
            && !self.paused.swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            barrier.wait();
            barrier.wait();
        }
        Model.document(text)
    }
    fn query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.queries
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Model.query(text)
    }
}

/// WF-985 batch model: two dimensions, a gate that holds `documents`, a
/// call log and one injectable failure. Fixed texts map to orthogonal
/// vectors; other texts use `Model`.
#[allow(dead_code)] // Only the deferred-embedding tests use it.
#[derive(Default)]
pub struct GateModel {
    pub calls: std::sync::Mutex<Vec<Vec<String>>>,
    pub fail: std::sync::atomic::AtomicBool,
    gate: std::sync::Mutex<GateState>,
    changed: std::sync::Condvar,
}

#[allow(dead_code)]
#[derive(Default)]
struct GateState {
    closed: bool,
    entered: usize,
}

#[allow(dead_code, clippy::expect_used)]
impl GateModel {
    pub fn close_gate(&self) {
        self.gate.lock().expect("gate").closed = true;
    }
    pub fn open_gate(&self) {
        self.gate.lock().expect("gate").closed = false;
        self.changed.notify_all();
    }
    /// Waits until `count` batch calls have entered `documents`.
    pub fn wait_entered(&self, count: usize, timeout: std::time::Duration) -> bool {
        let gate = self.gate.lock().expect("gate");
        let (gate, _) = self
            .changed
            .wait_timeout_while(gate, timeout, |state| state.entered < count)
            .expect("gate wait");
        gate.entered >= count
    }
    pub fn call_count(&self) -> usize {
        self.calls.lock().expect("calls").len()
    }
    pub fn vector(text: &str) -> Vec<f32> {
        match text {
            "fictitious mauve sprocket" => vec![1., 0.],
            "fictitious cyan thimble" => vec![0., 1.],
            _ => vec![0.6, 0.8],
        }
    }
}

#[allow(clippy::expect_used)]
impl NamespaceEmbedder for GateModel {
    fn epoch(&self) -> EmbeddingEpoch {
        Model.epoch()
    }
    fn document(&self, text: &str) -> Result<Vec<EmbeddedChunk>, EmbedderError> {
        let mut out = self.documents(&[text])?;
        out.pop()
            .ok_or_else(|| EmbedderFailure::new("missing document").into())
    }
    fn documents(&self, texts: &[&str]) -> Result<Vec<Vec<EmbeddedChunk>>, EmbedderError> {
        self.calls
            .lock()
            .expect("calls")
            .push(texts.iter().map(|text| (*text).to_owned()).collect());
        let mut gate = self.gate.lock().expect("gate");
        gate.entered += 1;
        self.changed.notify_all();
        while gate.closed {
            gate = self.changed.wait(gate).expect("gate");
        }
        drop(gate);
        if self.fail.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(EmbedderFailure::new("injected batch failure").into());
        }
        Ok(texts
            .iter()
            .map(|text| {
                vec![EmbeddedChunk {
                    bytes: 0..text.len(),
                    vector: Self::vector(text),
                }]
            })
            .collect())
    }
    fn query(&self, _: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(vec![1., 0.])
    }
}
