//! Tensor names shared by every runtime backend.

use crate::bundle::Bundle;
use crate::tower::{Architecture, TowerRole, TowerSpec};

/// Name prefix of one role's tensors. A symmetric bundle serves both roles
/// from its document tensors.
pub(crate) fn tensor_prefix(bundle: &Bundle, role: TowerRole) -> &'static str {
    match role {
        TowerRole::Document => "document/",
        TowerRole::Query if bundle.is_symmetric() => "document/",
        TowerRole::Query => "query/",
    }
}

/// Tensor names one tower needs, without the role prefix.
pub(crate) fn required_tensor_names(tower: &TowerSpec) -> Vec<String> {
    let mut names = vec!["embeddings.word_embeddings.weight".to_owned()];
    if tower.architecture == Architecture::ArcticQuery {
        if tower.config.embedding_rank != tower.config.hidden {
            names.push("embeddings.embedding_projection.weight".to_owned());
        }
        if tower.config.output_dimension != tower.config.hidden {
            names.push("output_projection.weight".to_owned());
        }
    }
    if tower.config.type_vocab > 0 {
        names.push("embeddings.token_type_embeddings.weight".to_owned());
    }
    if tower.architecture == Architecture::Bert {
        names.push("embeddings.position_embeddings.weight".to_owned());
    }
    names.push("embeddings.LayerNorm.weight".to_owned());
    names.push("embeddings.LayerNorm.bias".to_owned());
    for layer in 0..tower.config.layer_count {
        match tower.architecture {
            Architecture::Bert => {
                for suffix in [
                    "attention.self.query.weight",
                    "attention.self.query.bias",
                    "attention.self.key.weight",
                    "attention.self.key.bias",
                    "attention.self.value.weight",
                    "attention.self.value.bias",
                    "attention.output.dense.weight",
                    "attention.output.dense.bias",
                    "attention.output.LayerNorm.weight",
                    "attention.output.LayerNorm.bias",
                    "intermediate.dense.weight",
                    "intermediate.dense.bias",
                    "output.dense.weight",
                    "output.dense.bias",
                    "output.LayerNorm.weight",
                    "output.LayerNorm.bias",
                ] {
                    names.push(format!("encoder.layer.{layer}.{suffix}"));
                }
            }
            Architecture::Gte | Architecture::ArcticQuery => {
                for suffix in [
                    "attention.qkv_proj.weight",
                    "attention.qkv_proj.bias",
                    "attention.o_proj.weight",
                    "attention.o_proj.bias",
                    "attn_ln.weight",
                    "attn_ln.bias",
                    "mlp.up_gate_proj.weight",
                    "mlp.down_proj.weight",
                    "mlp.down_proj.bias",
                    "mlp_ln.weight",
                    "mlp_ln.bias",
                ] {
                    names.push(format!("encoder.layer.{layer}.{suffix}"));
                }
            }
        }
    }
    if tower.config.dense_out > 0 {
        names.push("dense.weight".to_owned());
        names.push("dense.bias".to_owned());
    }
    names
}
