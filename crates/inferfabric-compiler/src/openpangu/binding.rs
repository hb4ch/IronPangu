//! Checkpoint binding for the observed OpenPangu V2 MLA + mHC + MoME contract.
//! This is a weight contract, not an executable native plan.
use super::{Inventory, Manifest, TensorDescriptor, number};
use inferfabric_model::{Result, invalid};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize)]
pub struct WeightContract {
    pub format: &'static str,
    pub executable: bool,
    pub checkpoint_metadata_sha256: String,
    pub main_layers: usize,
    pub bound_bytes: u64,
    pub mtp_bytes: u64,
    pub bindings: BTreeMap<String, WeightBinding>,
    pub excluded_mtp_tensors: Vec<String>,
    pub shared_index_sources: BTreeMap<usize, usize>,
    pub limitations: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct WeightBinding {
    pub tensor: TensorDescriptor,
    pub transform: &'static str,
}
struct Binder<'a> {
    inventory: &'a Inventory,
    bindings: BTreeMap<String, WeightBinding>,
    quantized: bool,
    ignore: BTreeSet<String>,
}
impl Binder<'_> {
    fn tensor(
        &mut self,
        name: &str,
        shape: &[u64],
        dtypes: &[&str],
        transform: &'static str,
    ) -> Result<()> {
        let t = self
            .inventory
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing OpenPangu tensor {name}")))?;
        if t.shape != shape || !dtypes.contains(&t.dtype.as_str()) {
            return Err(invalid(format!(
                "{name}: expected {shape:?} {dtypes:?}, got {:?} {}",
                t.shape, t.dtype
            )));
        }
        if self
            .bindings
            .insert(
                name.into(),
                WeightBinding {
                    tensor: t.clone(),
                    transform,
                },
            )
            .is_some()
        {
            return Err(invalid(format!("duplicate binding {name}")));
        }
        Ok(())
    }
    fn float(&mut self, name: &str, shape: &[u64], transform: &'static str) -> Result<()> {
        self.tensor(name, shape, &["BF16", "F32"], transform)
    }
    fn linear(&mut self, prefix: &str, output: u64, input: u64) -> Result<()> {
        if self.quantized && !self.ignore.contains(prefix) {
            self.tensor(
                &format!("{prefix}.weight"),
                &[output, input],
                &["I8"],
                "dynamic_w8a8_weight_out_in",
            )?;
            self.float(
                &format!("{prefix}.weight_scale"),
                &[output, 1],
                "per_output_scale_to_fp32",
            )
        } else {
            self.float(
                &format!("{prefix}.weight"),
                &[output, input],
                "linear_weight_out_in",
            )
        }
    }
    fn mlp(&mut self, prefix: &str, h: u64, intermediate: u64) -> Result<()> {
        self.linear(&format!("{prefix}.gate_proj"), intermediate, h)?;
        self.linear(&format!("{prefix}.up_proj"), intermediate, h)?;
        self.linear(&format!("{prefix}.down_proj"), h, intermediate)
    }
    fn mhc(&mut self, prefix: &str, h: u64, streams: u64, merge: bool) -> Result<()> {
        let out = if merge {
            streams
        } else {
            streams * (streams + 2)
        };
        let suffix = if merge { "_pre" } else { "" };
        self.float(
            &format!("{prefix}.branch_alpha{suffix}"),
            &[if merge { 1 } else { 3 }],
            "cast_fp32",
        )?;
        self.float(
            &format!("{prefix}.branch_beta{suffix}"),
            &[out],
            "cast_fp32",
        )?;
        self.float(&format!("{prefix}.norm_gamma"), &[streams * h], "cast_fp32")?;
        self.float(
            &format!("{prefix}.phi.weight"),
            &[out, streams * h],
            "cast_fp32",
        )
    }
}
fn require(c: &Value, key: &str, expected: Value) -> Result<()> {
    if c[key] != expected {
        return Err(invalid(format!(
            "OpenPangu weight contract requires {key}={expected}"
        )));
    }
    Ok(())
}
fn quantization(c: &Value) -> Result<(bool, BTreeSet<String>)> {
    let Some(q) = c.get("quantization_config").filter(|q| !q.is_null()) else {
        return Ok((false, BTreeSet::new()));
    };
    for (key, value) in [
        ("quant_method", "compressed-tensors"),
        ("quantize", "w8a8_dynamic"),
        ("format", "int-quantized"),
        ("quantization_status", "compressed"),
    ] {
        require(q, key, Value::from(value))?;
    }
    let groups = q["config_groups"]
        .as_object()
        .ok_or_else(|| invalid("missing quantization groups"))?;
    if groups.len() != 1 {
        return Err(invalid(
            "only a single W8A8 Linear quantization group is supported",
        ));
    }
    let g = groups.values().next().unwrap();
    require(g, "targets", serde_json::json!(["Linear"]))?;
    for (key, value) in [
        ("num_bits", Value::from(8)),
        ("type", Value::from("int")),
        ("symmetric", Value::from(true)),
    ] {
        require(&g["weights"], key, value.clone())?;
        require(&g["input_activations"], key, value)?;
    }
    require(&g["weights"], "strategy", Value::from("channel"))?;
    require(&g["weights"], "dynamic", Value::from(false))?;
    require(&g["input_activations"], "strategy", Value::from("token"))?;
    require(&g["input_activations"], "dynamic", Value::from(true))?;
    for obj in [&g["weights"], &g["input_activations"]] {
        for key in ["actorder", "group_size", "block_structure"] {
            if !obj[key].is_null() {
                return Err(invalid(format!("unsupported quantization {key}")));
            }
        }
    }
    if !g["output_activations"].is_null() || !q["kv_cache_scheme"].is_null() {
        return Err(invalid(
            "output/cache quantization requires a separate contract",
        ));
    }
    let ignore: Vec<String> = serde_json::from_value(q["ignore"].clone())
        .map_err(|_| invalid("quantization ignore must be explicit module names"))?;
    if ignore.iter().any(|s| s.contains(['*', '?', '['])) {
        return Err(invalid("quantization ignore patterns are not supported"));
    }
    Ok((true, ignore.into_iter().collect()))
}
/// Bind every main-model tensor. MTP weights are explicitly excluded, not silently loaded.
pub fn bind(m: &Manifest) -> Result<WeightContract> {
    let c = &m.config;
    let i = m
        .inventory
        .as_ref()
        .ok_or_else(|| invalid("checkpoint inventory required"))?;
    for key in [
        "use_mhc",
        "mhc_use_gamma",
        "use_mome",
        "sandwich_norm",
        "param_sink_with_value",
        "router_enable_expert_bias",
    ] {
        require(c, key, Value::from(true))?;
    }
    require(c, "hidden_act", Value::from("silu"))?;
    require(c, "tie_word_embeddings", Value::from(false))?;
    for key in ["rope_interleaved", "rope_interleave"] {
        if c.get(key).is_some() {
            require(c, key, Value::from(false))?;
        }
    }
    if m.features["cross_layer_cache_reuse_declared"] {
        return Err(invalid("CLA weight binding is not implemented"));
    }
    let n = |key| {
        let v = number(c, key, None)?;
        if v == 0 || v > (1 << 20) {
            return Err(invalid(format!("unsupported binding dimension {key}")));
        }
        Ok(v)
    };
    let (h, heads, q, kv, nope, rope, v) = (
        n("hidden_size")?,
        n("num_attention_heads")?,
        n("q_lora_rank")?,
        n("kv_lora_rank")?,
        n("qk_nope_head_dim")?,
        n("qk_rope_head_dim")?,
        n("v_head_dim")?,
    );
    let streams = n("mhc_num_stream")?;
    if !(2..=16).contains(&streams) || n("router_sliding_window")? != 3 {
        return Err(invalid("unsupported mHC stream count or MoME window"));
    }
    let (quantized, ignore) = quantization(c)?;
    let mut b = Binder {
        inventory: i,
        bindings: BTreeMap::new(),
        quantized,
        ignore,
    };
    b.float(
        "model.embed_tokens.weight",
        &[n("vocab_size")?, h],
        "embedding",
    )?;
    b.linear("lm_head", n("vocab_size")?, h)?;
    b.float("model.norm.weight", &[h], "rms_norm")?;
    b.mhc("model.merge_mhc_module", h, streams, true)?;
    let mut last_index = None;
    let mut shared = BTreeMap::new();
    for l in &m.layers {
        let p = format!("model.layers.{}", l.index);
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_mlp_layernorm",
            "post_mlp_layernorm",
        ] {
            b.float(&format!("{p}.{name}.weight"), &[h], "rms_norm")?;
        }
        if c["block_post_layernorm_idx"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x.as_u64() == Some(l.index as u64)))
        {
            b.float(
                &format!("{p}.block_post_layernorm.weight"),
                &[streams * h],
                "rms_norm",
            )?;
        }
        b.mhc(&format!("{p}.attn_mhc_module"), h, streams, false)?;
        b.mhc(&format!("{p}.mlp_mhc_module"), h, streams, false)?;
        let a = format!("{p}.self_attn");
        for (name, out, input) in [
            ("q_a_proj", q, h),
            ("q_b_proj", heads * (nope + rope), q),
            ("kv_a_proj_with_mqa", kv + rope, h),
            ("kv_b_proj", heads * (nope + v), kv),
            ("o_proj", h, heads * v),
        ] {
            b.linear(&format!("{a}.{name}"), out, input)?;
        }
        b.float(&format!("{a}.q_a_layernorm.weight"), &[q], "rms_norm")?;
        b.float(&format!("{a}.kv_a_layernorm.weight"), &[kv], "rms_norm")?;
        for (name, dim) in [
            ("qa_conv", q),
            ("compresskv_conv", kv),
            ("o_conv", heads * v),
        ] {
            b.float(
                &format!("{a}.{name}.weight"),
                &[dim, 1, 3],
                "mome_channels_1_width_to_width_channels",
            )?;
        }
        b.float(
            &format!("{a}.param_sink_compressed_kv"),
            &[n("param_sink_number")?, kv],
            "latent_sink",
        )?;
        b.float(
            &format!("{a}.param_sink_k_pe"),
            &[n("param_sink_number")?, rope],
            "rotary_sink",
        )?;
        if l.attention == "sparse_mla" {
            let mode = l.indexer_mode.as_deref().unwrap_or("full");
            match mode {
                "full" => last_index = Some(l.index),
                "shared" => {
                    shared.insert(
                        l.index,
                        last_index.ok_or_else(|| {
                            invalid("shared indexer has no earlier full producer")
                        })?,
                    );
                }
                _ => return Err(invalid(format!("unsupported sparse indexer mode {mode}"))),
            }
            // Released checkpoint retains indexer parameters even on shared consumers.
            b.linear(
                &format!("{a}.indexer.wq_b"),
                n("index_head_dim")? * n("index_n_heads")?,
                q,
            )?;
            b.linear(&format!("{a}.indexer.wk"), n("index_head_dim")?, h)?;
            b.linear(&format!("{a}.indexer.weights_proj"), n("index_n_heads")?, h)?;
            b.float(
                &format!("{a}.indexer.k_norm.weight"),
                &[n("index_head_dim")?],
                "rms_norm",
            )?;
        } else if l.indexer_mode.as_deref().is_some_and(|s| s != "swa") {
            return Err(invalid("non-SWA index mode on window layer"));
        }
        let p = format!("{p}.mlp");
        if l.feed_forward == "dense" {
            b.mlp(&p, h, n("intermediate_size")?)?;
        } else {
            let experts = n("n_routed_experts")?;
            b.linear(&format!("{p}.gate"), experts, h)?;
            b.float(
                &format!("{p}.e_score_correction_bias"),
                &[experts],
                "cast_fp32",
            )?;
            b.mlp(
                &format!("{p}.shared_experts"),
                h,
                n("n_shared_experts")? * n("moe_intermediate_size")?,
            )?;
            for e in 0..experts {
                b.mlp(&format!("{p}.experts.{e}"), h, n("moe_intermediate_size")?)?;
            }
        }
    }
    let mut excluded = vec![];
    let mut mtp_bytes = 0;
    let total = m.layers.len() as u64 + number(c, "num_nextn_predict_layers", Some(0))?;
    for (name, t) in &i.tensors {
        if b.bindings.contains_key(name) {
            continue;
        }
        let layer = name
            .strip_prefix("model.layers.")
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<u64>().ok());
        if layer.is_some_and(|l| l >= m.layers.len() as u64 && l < total) {
            excluded.push(name.clone());
            mtp_bytes += t.bytes;
        } else {
            return Err(invalid(format!("unbound main-model tensor {name}")));
        }
    }
    Ok(WeightContract {
        format:"inferfabric.openpangu-weights.v1",executable:false,
        checkpoint_metadata_sha256:i.metadata_sha256.clone(),main_layers:m.layers.len(),
        bound_bytes:b.bindings.values().map(|b| b.tensor.bytes).sum(),mtp_bytes,
        bindings:b.bindings,excluded_mtp_tensors:excluded,shared_index_sources:shared,
        limitations:vec!["Weight/header contract only; no native execution DAG or payload-integrity qualification".into(),"MTP explicitly excluded; shared indexer parameters retained, consumer index dependency recorded".into(),"Quantized scale positivity/finiteness must be verified when payloads are loaded".into(),"No rank placement or runtime HBM estimate; FP32 casts and convolution transforms require separate allocation".into()],
    })
}
