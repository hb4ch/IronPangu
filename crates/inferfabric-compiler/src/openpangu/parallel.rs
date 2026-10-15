//! Auditable TP/EP load slices and collective schedule. Not a native executable.
use super::{
    Manifest,
    binding::{self, WeightBinding},
};
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use serde::Serialize;
use std::collections::BTreeMap;
#[derive(Debug, Serialize)]
pub struct ParallelPlan {
    pub format: &'static str,
    pub executable: bool,
    pub checkpoint_metadata_sha256: String,
    pub topology: OpenPanguParallel,
    pub experts_per_rank: usize,
    pub ranks: Vec<Rank>,
    pub schedule: Vec<Step>,
    pub notes: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct Rank {
    pub rank: usize,
    pub device: u32,
    pub expert_begin: usize,
    pub expert_end: usize,
    pub stored_weight_bytes: u64,
    pub prepared_weight_bytes: u64,
    pub weights: BTreeMap<String, Slice>,
}
#[derive(Debug, Serialize)]
pub struct Slice {
    pub shard: String,
    pub offset: u64,
    pub bytes: u64,
    pub local_shape: Vec<u64>,
    pub dtype: String,
    pub transform: &'static str,
    pub placement: &'static str,
}
#[derive(Debug, Serialize)]
pub struct Step {
    pub layer: usize,
    pub operation: &'static str,
    pub collective: &'static str,
    pub output_layout: &'static str,
}
fn expert(name: &str) -> Option<usize> {
    name.split_once(".mlp.experts.")
        .and_then(|(_, s)| s.split('.').next())
        .and_then(|s| s.parse().ok())
}
fn tp_linear(name: &str) -> bool {
    name.starts_with("lm_head.")
        || [
            ".self_attn.q_b_proj.",
            ".self_attn.kv_b_proj.",
            ".self_attn.o_proj.",
            ".mlp.gate_proj.",
            ".mlp.up_proj.",
            ".mlp.down_proj.",
            ".mlp.shared_experts.gate_proj.",
            ".mlp.shared_experts.up_proj.",
            ".mlp.shared_experts.down_proj.",
        ]
        .iter()
        .any(|s| name.contains(s))
}
fn slice(
    b: &WeightBinding,
    rank: usize,
    tp: usize,
    sharded: bool,
    placement: &'static str,
) -> Result<Slice> {
    let t = &b.tensor;
    let mut shape = t.shape.clone();
    let mut bytes = t.bytes;
    let mut offset = t.offset;
    if sharded {
        if shape.len() != 2
            || !shape[0].is_multiple_of(tp as u64)
            || !bytes.is_multiple_of(tp as u64)
        {
            return Err(invalid("TP output rows/scale rows must divide evenly"));
        }
        shape[0] /= tp as u64;
        bytes /= tp as u64;
        offset = offset
            .checked_add(
                bytes
                    .checked_mul(rank as u64)
                    .ok_or_else(|| invalid("slice overflow"))?,
            )
            .ok_or_else(|| invalid("slice offset overflow"))?;
    }
    Ok(Slice {
        shard: t.shard.clone(),
        offset,
        bytes,
        local_shape: shape,
        dtype: t.dtype.clone(),
        transform: b.transform,
        placement,
    })
}
pub fn compile(m: &Manifest, spec: &OpenPanguParallel) -> Result<ParallelPlan> {
    spec.validate()?;
    let b = binding::bind(m)?;
    let experts = m.config["n_routed_experts"]
        .as_u64()
        .ok_or_else(|| invalid("missing expert count"))? as usize;
    if experts == 0 || !experts.is_multiple_of(spec.ep) {
        return Err(invalid("experts must divide EP group"));
    }
    if spec.max_model_len as u64 > m.config["max_position_embeddings"].as_u64().unwrap_or(0) {
        return Err(invalid("context exceeds checkpoint limit"));
    }
    let per = experts / spec.ep;
    let mut ranks = vec![];
    for (rank, &device) in spec.devices.iter().enumerate() {
        let mut r = Rank {
            rank,
            device,
            expert_begin: rank * per,
            expert_end: (rank + 1) * per,
            stored_weight_bytes: 0,
            prepared_weight_bytes: 0,
            weights: BTreeMap::new(),
        };
        for (name, weight) in &b.bindings {
            let e = expert(name);
            if e.is_some_and(|e| e / per != rank) {
                continue;
            }
            let sharded = e.is_none() && tp_linear(name);
            let x = slice(
                weight,
                rank,
                spec.tp,
                sharded,
                if e.is_some() {
                    "expert_owner"
                } else if sharded {
                    "tp_output_rows"
                } else {
                    "replicated"
                },
            )?;
            r.stored_weight_bytes = r
                .stored_weight_bytes
                .checked_add(x.bytes)
                .ok_or_else(|| invalid("weight byte overflow"))?;
            let prepared = if ["cast_fp32", "per_output_scale_to_fp32"].contains(&x.transform)
                && x.dtype == "BF16"
            {
                x.bytes
                    .checked_mul(2)
                    .ok_or_else(|| invalid("cast bytes overflow"))?
            } else {
                x.bytes
            };
            r.prepared_weight_bytes = r
                .prepared_weight_bytes
                .checked_add(prepared)
                .ok_or_else(|| invalid("prepared bytes overflow"))?;
            r.weights.insert(name.clone(), x);
        }
        ranks.push(r);
    }
    let mut schedule = vec![];
    for l in &m.layers {
        let mut push = |op, collective, layout| {
            schedule.push(Step {
                layer: l.index,
                operation: op,
                collective,
                output_layout: layout,
            })
        };
        push("mhc_pre_and_replicated_q_a_kv_a", "none", "replicated");
        push(
            "q_b_kv_b_output_row_matmuls",
            "all_gather_each_projection",
            "replicated_canonical_rows",
        );
        push("mla_swa_or_sparse_attention_and_mome", "none", "replicated");
        push(
            "o_proj_output_row_matmul",
            "all_gather",
            "replicated_canonical_rows",
        );
        push("mhc_post_and_ffn_pre", "none", "replicated");
        if l.feed_forward == "moe" {
            push(
                "sigmoid_router_bias_topk_normalize_scale",
                "none",
                "identical_routes_all_ranks",
            );
            push(
                "owner_expert_gate_up_silu_down_weighted_sum",
                "all_reduce_sum_fp32",
                "replicated_expert_output",
            );
            push(
                "shared_expert_gate_up_silu_down",
                "all_gather_each_projection",
                "replicated_shared_output",
            );
            push("add_shared_and_routed", "none", "replicated");
        } else {
            push(
                "dense_gate_up_silu_down",
                "all_gather_each_projection",
                "replicated",
            );
        }
        push("mhc_post_and_optional_block_norm", "none", "replicated");
    }
    Ok(ParallelPlan {format:"inferfabric.openpangu-parallel.v1",executable:false,checkpoint_metadata_sha256:b.checkpoint_metadata_sha256,topology:spec.clone(),experts_per_rank:per,ranks,schedule,notes:vec![
        "TP and EP use the SAME ranks, not a TP*EP Cartesian mesh; pipeline parallelism is 1".into(),
        "Correctness-first output-row TP gathers full activations after each sharded projection; attention/cache and token batch remain replicated".into(),
        "Each expert has one owner and is NOT additionally tensor-sharded; zero-work ranks still participate in every collective".into(),
        "EP uses replicated tokens, local owner masking, FP32 weighted accumulation and all-reduce; no all-to-all dispatch is claimed".into(),
        "Prepared bytes cover declared weight casts only; exclude staging, kernel packing, KV/MoME/index state, workspace, graph and communication buffers".into(),
        "Placement/schedule contract only; full OpenPangu native execution and serving remain unimplemented".into(),
    ]})
}
