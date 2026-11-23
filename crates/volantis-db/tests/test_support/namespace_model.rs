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
