//! CPU-only architecture audit. This is not a model compiler or serving qualification.
mod inventory;
mod json;
use crate::checkpoint::sha;
use inferfabric_model::{Result, invalid};
pub use inventory::{Inventory, TensorDescriptor};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
pub const MAX_METADATA: u64 = 16 * 1024 * 1024;
#[derive(Debug, Serialize)]
pub struct Layer {
    pub index: usize,
    pub attention: String,
    pub sliding_window: Option<u64>,
    pub feed_forward: String,
    pub cla_source_layer: Option<u64>,
    pub indexer_mode: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct Manifest {
    pub format: &'static str,
    pub executable: bool,
    pub hardware_qualified: bool,
    pub config_sha256: String,
    pub config: Value,
    pub layers: Vec<Layer>,
    pub features: BTreeMap<String, bool>,
    pub required_operations: Vec<String>,
    pub inventory: Option<Inventory>,
    pub warnings: Vec<String>,
    pub unresolved: Vec<String>,
}
pub(super) fn number(c: &Value, key: &str, default: Option<u64>) -> Result<u64> {
    match c.get(key) {
        Some(v) => v
            .as_u64()
            .ok_or_else(|| invalid(format!("{key} must be an unsigned integer"))),
        None => default.ok_or_else(|| invalid(format!("missing {key}"))),
    }
}
fn positive(c: &Value, key: &str) -> Result<u64> {
    let n = number(c, key, None)?;
    if n == 0 || n > 1 << 30 {
        return Err(invalid(format!("invalid {key}")));
    }
    Ok(n)
}
fn boolean(c: &Value, key: &str, default: bool) -> Result<bool> {
    match c.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid(format!("{key} must be boolean"))),
    }
}
fn indices(c: &Value, key: &str, bound: u64) -> Result<Vec<u64>> {
    let Some(value) = c.get(key) else {
        return Ok(vec![]);
    };
    let values = value
        .as_array()
        .ok_or_else(|| invalid(format!("{key} must be an array")))?;
    let mut seen = BTreeSet::new();
    values
        .iter()
        .map(|v| {
            let n = v
                .as_u64()
                .ok_or_else(|| invalid(format!("invalid {key} entry")))?;
            if n >= bound || !seen.insert(n) {
                return Err(invalid(format!("duplicate/out-of-range {key}")));
            }
            Ok(n)
        })
        .collect()
}
pub fn inspect_config(path: &Path) -> Result<Manifest> {
    from_config(&json::read(path)?)
}
pub fn from_config(bytes: &[u8]) -> Result<Manifest> {
    let c = json::parse(bytes)?;
    if c["model_type"] != "pangu_v2_moe" {
        return Err(invalid(
            "expected model_type pangu_v2_moe; V3/VL/other models require separate contracts",
        ));
    }
    for key in [
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "max_position_embeddings",
    ] {
        positive(&c, key)?;
    }
    let layers = positive(&c, "num_hidden_layers")?;
    let mtp = number(&c, "num_nextn_predict_layers", Some(0))?;
    if layers > 4096 || mtp > 256 {
        return Err(invalid("layer count exceeds inspector limits"));
    }
    let total = layers + mtp;
    let dense = number(&c, "first_k_dense_replace", Some(layers))?;
    if dense > layers {
        return Err(invalid("dense prefix exceeds layer count"));
    }
    let moe = dense < layers;
    if moe {
        let experts = positive(&c, "n_routed_experts")?;
        let topk = positive(&c, "num_experts_per_tok")?;
        positive(&c, "moe_intermediate_size")?;
        number(&c, "n_shared_experts", None)?;
        boolean(&c, "norm_topk_prob", false)?;
        if topk > experts {
            return Err(invalid("expert top-k exceeds expert count"));
        }
    }
    let swa = indices(&c, "swa_layers", total)?;
    let windows = match c.get("sliding_window_list") {
        Some(v) => v
            .as_array()
            .ok_or_else(|| invalid("sliding_window_list must be array"))?
            .clone(),
        None => vec![],
    };
    if swa.len() != windows.len() {
        return Err(invalid("SWA layers/windows length mismatch"));
    }
    let mut window_by_layer = BTreeMap::new();
    for (i, v) in swa.iter().zip(windows) {
        let n = v
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("invalid sliding window"))?;
        window_by_layer.insert(*i, n);
    }
    indices(&c, "dsa_layers", layers)?;
    indices(&c, "block_post_layernorm_idx", layers)?;
    let index_topk = number(&c, "index_topk", Some(0))?;
    if index_topk > 0 {
        positive(&c, "index_n_heads")?;
        positive(&c, "index_head_dim")?;
        positive(&c, "q_lora_rank")?;
    }
    let mome = boolean(&c, "use_mome", false)?;
    if mome {
        positive(&c, "router_sliding_window")?;
        positive(&c, "q_lora_rank")?;
    }
    let streams = number(&c, "mhc_num_stream", Some(1))?;
    if streams == 0 || streams > 1024 {
        return Err(invalid("invalid mHC stream count"));
    }
    let mhc = streams > 1;
    let mut warnings = vec![];
    if c.get("use_mhc").is_some() && boolean(&c, "use_mhc", false)? != mhc {
        warnings.push("use_mhc differs from mhc_num_stream; reference decoder selects mHC by stream count; resolve before compilation".into());
    }
    if mhc {
        positive(&c, "mhc_recur_norm")?;
        if !boolean(&c, "mhc_use_gamma", false)? {
            return Err(invalid("reference mHC requires mhc_use_gamma"));
        }
    }
    let mut cla = BTreeMap::new();
    if let Some(v) = c.get("cla_explicit_mapping").filter(|v| !v.is_null()) {
        let pairs: Vec<(u64, u64)> = if let Some(m) = v.as_object() {
            m.iter()
                .map(|(k, v)| {
                    Ok((
                        k.parse::<u64>()
                            .map_err(|_| invalid("invalid CLA layer key"))?,
                        v.as_u64().ok_or_else(|| invalid("invalid CLA source"))?,
                    ))
                })
                .collect::<Result<_>>()?
        } else if let Some(a) = v.as_array() {
            a.iter()
                .map(|v| {
                    let pair: [u64; 2] = serde_json::from_value(v.clone())
                        .map_err(|_| invalid("CLA entries must be integer pairs"))?;
                    Ok((pair[0], pair[1]))
                })
                .collect::<Result<_>>()?
        } else {
            return Err(invalid("invalid CLA mapping"));
        };
        for (reuse, source) in pairs {
            if reuse >= layers || source >= reuse || cla.insert(reuse, source).is_some() {
                return Err(invalid(
                    "CLA source must precede a unique in-range reuse layer",
                ));
            }
        }
    }
    let indexer_modes: Vec<String> = match c.get("indexer_types").filter(|v| !v.is_null()) {
        None => vec![],
        Some(v) => {
            let modes: Vec<String> = serde_json::from_value(v.clone())
                .map_err(|_| invalid("indexer_types must contain strings"))?;
            if modes.len() < layers as usize
                || modes.len() > total as usize
                || modes.iter().any(String::is_empty)
            {
                return Err(invalid("invalid indexer_types length or empty mode"));
            }
            modes
        }
    };
    let sink = number(&c, "param_sink_number", Some(0))? > 0;
    let mut summary = vec![];
    for i in 0..layers {
        let window = window_by_layer.get(&i).copied();
        summary.push(Layer {
            index: i as usize,
            attention: if window.is_some() {
                "sliding_window_mla"
            } else if index_topk > 0 {
                "sparse_mla"
            } else {
                "full_mla"
            }
            .into(),
            sliding_window: window,
            feed_forward: if i < dense { "dense" } else { "moe" }.into(),
            cla_source_layer: cla.get(&i).copied(),
            indexer_mode: indexer_modes.get(i as usize).cloned(),
        });
    }
    let features = BTreeMap::from([
        ("cross_layer_cache_reuse_declared".into(), !cla.is_empty()),
        (
            "index_sharing_declared".into(),
            indexer_modes.iter().any(|v| v == "shared"),
        ),
        ("moe".into(), moe),
        ("mome".into(), mome),
        ("mhc".into(), mhc),
        ("learned_sinks".into(), sink),
        ("sparse_indexer".into(), index_topk > 0),
        ("mtp_declared".into(), mtp > 0),
        (
            "quantization_declared".into(),
            c.get("quantization_config").is_some_and(|v| !v.is_null()),
        ),
    ]);
    let mut ops = vec![
        "embedding",
        "rms_norm",
        "linear",
        "rotary",
        "mla_cache_append",
        "mla_attention",
        "gated_ffn",
        "residual",
        "logits",
        "sampling",
    ];
    if !cla.is_empty() {
        ops.push("shared_layer_cache_views")
    }
    if indexer_modes.iter().any(|v| v == "shared") {
        ops.push("shared_sparse_indices")
    }
    if moe {
        ops.extend([
            "sigmoid_router",
            "routing_bias_topk",
            "expert_permutation",
            "grouped_matmul",
            "weighted_combine",
            "shared_experts",
        ])
    }
    if mome {
        ops.extend(["causal_channel_convolution", "history_gather_update"])
    }
    if mhc {
        ops.extend(["mhc_pre", "sinkhorn", "mhc_post"])
    }
    if index_topk > 0 {
        ops.extend(["indexer_cache_append", "sparse_topk", "sparse_attention"])
    }
    if sink {
        ops.push("attention_sinks")
    }
    if !swa.is_empty() {
        ops.push("sliding_window_attention")
    }
    warnings.push("Layer summary follows the default NPUPanguSparseAttention selection: SWA first, then positive index_topk, else full MLA. dsa_layers is retained but the alternative high-throughput backend must be qualified separately.".into());
    warnings.push("Raw config is preserved, including unknown fields; this audit does not certify complete model semantics, index sharing, CLA mapping, quantization or MTP execution.".into());
    Ok(Manifest {
        format: "inferfabric.openpangu-audit.v1",
        executable: false,
        hardware_qualified: false,
        config_sha256: sha(bytes),
        config: c,
        layers: summary,
        features,
        required_operations: ops.into_iter().map(str::to_owned).collect(),
        inventory: None,
        warnings,
        unresolved: vec![
            "Exact checkpoint tensor binding/shape contract and payload integrity".into(),
            "Tokenizer/chat-template and generation-default compatibility".into(),
            "Rank placement, replicated/transformed weights and load staging".into(),
            "KV/indexer/MoME state, activation/workspace/communication and graph memory".into(),
            "Native ABI, numerical and serving qualification on both full checkpoints".into(),
        ],
    })
}
pub fn inspect_checkpoint(root: &Path) -> Result<Manifest> {
    let root = root.canonicalize()?;
    let mut result = inspect_config(&json::local_file(&root, "config.json")?)?;
    result.inventory = Some(inventory::inspect(&root)?);
    Ok(result)
}
