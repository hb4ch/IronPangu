//! Conservative lowering of a canonical Qwen mathematical plan to qualified native blocks.
use crate::{
    bound::{self, Plan},
    checkpoint::{PREFIX, sha},
};
use pangu_model::{Layer, Result, invalid};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Transform {
    Identity,
    OnePlusFp32,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub transform: Transform,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Block {
    pub layer: usize,
    pub kind: Layer,
    pub bindings: Vec<Binding>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Program {
    pub key: String,
    pub source_key: String,
    pub blocks: Vec<Block>,
    pub embedding: String,
    pub final_norm: Binding,
    pub batch: usize,
    pub context: usize,
    pub page_tokens: usize,
}
/// Native fusion only accepts the complete canonical graph, not arbitrary edited node lists.
/// Compatibility entry point for one lane. No serving readiness is implied.
pub fn lower(plan: &Plan, context: usize) -> Result<Program> {
    lower_batch(plan, context, 1)
}
/// Fixed-lane native graph with one token per selected sequence per replay.
pub fn lower_batch(plan: &Plan, context: usize, batch: usize) -> Result<Program> {
    if batch == 0 || batch > 64 {
        return Err(invalid("native batch must be in 1..=64"));
    }
    if &bound::compile(&plan.spec, plan.checkpoint.clone())? != plan {
        return Err(invalid("native lowering requires canonical typed plan"));
    }
    if !plan.spec.context_buckets.contains(&context) {
        return Err(invalid("native context must be a declared bucket"));
    }
    let mut blocks = vec![];
    for (layer, kind) in plan.spec.model.layers.iter().enumerate() {
        let mut bindings = vec![];
        let mut bind = |name: &str, transform| {
            bindings.push(Binding {
                name: format!("{PREFIX}layers.{layer}.{name}"),
                transform,
            })
        };
        bind("input_layernorm.weight", Transform::OnePlusFp32);
        bind("post_attention_layernorm.weight", Transform::OnePlusFp32);
        for name in [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            bind(name, Transform::Identity);
        }
        match kind {
            Layer::Delta => {
                for name in [
                    "linear_attn.in_proj_qkv.weight",
                    "linear_attn.in_proj_z.weight",
                    "linear_attn.in_proj_a.weight",
                    "linear_attn.in_proj_b.weight",
                    "linear_attn.conv1d.weight",
                    "linear_attn.A_log",
                    "linear_attn.dt_bias",
                    "linear_attn.norm.weight",
                    "linear_attn.out_proj.weight",
                ] {
                    bind(name, Transform::Identity);
                }
            }
            Layer::Full => {
                for name in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                ] {
                    bind(name, Transform::Identity);
                }
                bind("self_attn.q_norm.weight", Transform::OnePlusFp32);
                bind("self_attn.k_norm.weight", Transform::OnePlusFp32);
            }
        }
        blocks.push(Block {
            layer,
            kind: *kind,
            bindings,
        });
    }
    let mut result = Program {
        key: String::new(),
        source_key: plan.key.clone(),
        blocks,
        embedding: format!("{PREFIX}embed_tokens.weight"),
        final_norm: Binding {
            name: format!("{PREFIX}norm.weight"),
            transform: Transform::OnePlusFp32,
        },
        batch,
        context,
        page_tokens: plan.spec.page_tokens,
    };
    let mut bytes = serde_json::to_vec(&result).map_err(|e| invalid(e.to_string()))?;
    bytes.extend_from_slice(include_bytes!("lower.rs"));
    result.key = sha(&bytes);
    Ok(result)
}

/// Explicit runtime context specialization preserves the checkpoint's canonical math.
pub fn specialize_context(plan: &Plan, context: usize) -> Result<Plan> {
    if context == 0 || context > crate::checkpoint::MAX_MODEL_LEN {
        return Err(invalid("context exceeds checkpoint positional limit"));
    }
    if &bound::compile(&plan.spec, plan.checkpoint.clone())? != plan {
        return Err(invalid("context specialization requires canonical plan"));
    }
    let mut spec = plan.spec.clone();
    spec.context_buckets = vec![context];
    bound::compile(&spec, plan.checkpoint.clone())
}
