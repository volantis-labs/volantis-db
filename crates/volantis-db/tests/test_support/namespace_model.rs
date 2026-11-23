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
