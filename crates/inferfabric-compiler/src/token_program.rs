//! Executable storage lowering. Numerical rounding boundaries are preserved;
//! identical conversions of an immutable SSA value are shared.
use crate::openpangu::Inventory;
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Instruction {
    pub logical_node: usize,
    pub name: String,
    pub op: String,
    pub inputs: Vec<usize>,
    pub attrs: BTreeMap<String, String>,
    pub kind: usize,
    pub width: usize,
    pub dtype: String,
    pub kernel: String,
    pub reason: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Program {
    pub format: String,
    pub max_context: usize,
    pub eos: usize,
    pub output: usize,
    pub instructions: Vec<Instruction>,
    pub conversions_shared: usize,
    pub copies_eliminated: usize,
    pub limitations: Vec<String>,
}

fn exclusive_region(graph: &Graph, types: &[Type], input: usize, owner: usize) -> bool {
    fn region(g: &Graph, t: &[Type], mut id: usize) -> Option<(usize, usize, usize)> {
        let Type::Tensor(width) = t[id] else {
            return None;
        };
        let mut offset = 0;
        while g.nodes[id].op == "slice" {
            offset += g.nodes[id].integer("start").ok()?;
            id = g.nodes[id].inputs[0];
        }
        matches!(g.nodes[id].op.as_str(), "linear" | "norm" | "mla_value")
            .then_some((id, offset, width))
    }
    let Some((root, start, width)) = region(graph, types, input) else {
        return false;
    };
    let overlap = |id| {
        region(graph, types, id)
            .is_some_and(|(r, s, w)| r == root && s < start + width && start < s + w)
    };
    if overlap(graph.output) {
        return false;
    }
    !graph
        .nodes
        .iter()
        .enumerate()
        .any(|(id, n)| id != owner && n.op != "slice" && n.inputs.iter().any(|&i| overlap(i)))
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use inferfabric_authoring::execution::Node;
    #[test]
    fn disjoint_views_can_mutate_but_overlapping_readers_cannot() {
        let node = |op: &str, inputs: Vec<usize>, start: usize| Node {
            source: None,
            name: op.into(),
            op: op.into(),
            inputs,
            attrs: BTreeMap::from([("start".into(), start.to_string())]),
        };
        let mut g = Graph {
            name: "ownership".into(),
            max_context: 1,
            eos: 0,
            output: 4,
            nodes: vec![
                node("linear", vec![], 0),
                node("slice", vec![0], 0),
                node("slice", vec![0], 4),
                node("mome", vec![1], 0),
                node("norm", vec![2], 0),
            ],
        };
        let t = vec![
            Type::Tensor(8),
            Type::Tensor(4),
            Type::Tensor(4),
            Type::Tensor(4),
            Type::Tensor(4),
        ];
        assert!(exclusive_region(&g, &t, 1, 3));
        g.nodes[2].attrs.insert("start".into(), "2".into());
        assert!(!exclusive_region(&g, &t, 1, 3));
        g.nodes[2].attrs.insert("start".into(), "4".into());
        g.output = 1;
        assert!(!exclusive_region(&g, &t, 1, 3));
        g.output = 4;
        g.nodes[0].op = "mhc_input".into();
        assert!(!exclusive_region(&g, &t, 1, 3));
    }
}

pub fn lower(graph: &Graph, inventory: Option<&Inventory>) -> Result<Program> {
    let types = graph.validate()?;
    let mut p = Program {
        format: "inferfabric.token-program.v2".into(), max_context: graph.max_context,
        eos: graph.eos, output: 0, instructions: vec![], conversions_shared: 0, copies_eliminated: 0,
        limitations: vec!["Composite vendor recipes still own internal temporaries and launches; only inter-operation storage conversions are executable instructions in this revision. No claim of a complete kernel DAG or static arena.".into()],
    };
    let mut outputs = vec![];
    let mut casts = BTreeMap::new();
    for (logical, n) in graph.nodes.iter().enumerate() {
        let mut inputs: Vec<usize> = n.inputs.iter().map(|&i| outputs[i]).collect();
        for (slot, input) in inputs.iter_mut().enumerate() {
            if p.instructions[*input].kind != 0 {
                continue;
            }
            let wanted = match n.op.as_str() {
                "slice" | "repeat" | "concat" => None,
                "optional_norm"
                    if inventory.is_some_and(|v| {
                        !v.tensors
                            .contains_key(&format!("{}.weight", n.attrs["weight"]))
                    }) =>
                {
                    None
                }
                "allreduce" => Some(n.attrs.get("dtype").map(String::as_str).unwrap_or("f32")),
                "router" | "rope" | "add" => Some("f32"),
                "mhc_post" if slot == 0 => None,
                _ => Some("bf16"),
            };
            if let Some(dtype) = wanted {
                if p.instructions[*input].dtype != dtype {
                    let key = (*input, dtype.to_string());
                    if let Some(&id) = casts.get(&key) {
                        *input = id;
                        p.conversions_shared += 1;
                    } else {
                        let id = p.instructions.len();
                        let width = p.instructions[*input].width;
                        p.instructions.push(Instruction {
                            logical_node: logical,
                            name: format!("__cast_{id}"),
                            op: "cast".into(),
                            inputs: vec![*input],
                            attrs: BTreeMap::from([("dtype".into(), dtype.into())]),
                            kind: 0,
                            width,
                            dtype: dtype.into(),
                            kernel: "aclnnCast".into(),
                            reason: format!(
                                "{} input {slot}: {} storage required by numerical contract",
                                n.name, dtype
                            ),
                        });
                        casts.insert(key, id);
                        *input = id;
                    }
                }
            }
        }
        if n.op == "mome" {
            let source = &p.instructions[inputs[0]];
            let exclusive = exclusive_region(graph, &types, n.inputs[0], logical);
            if exclusive {
                p.copies_eliminated += 1;
            } else {
                let id = p.instructions.len();
                let width = source.width;
                p.instructions.push(Instruction { logical_node: logical, name: format!("__copy_{id}"), op: "copy".into(), inputs: vec![inputs[0]], attrs: BTreeMap::new(), kind: 0, width, dtype: "bf16".into(), kernel: "copy".into(), reason: "Private writable input for in-place convolution; exclusive ownership not proven".into() });
                inputs[0] = id;
            }
        }
        let (kind, width) = match types[logical] {
            Type::Tensor(w) => (0, w),
            Type::HyperConnection => (1, 0),
            Type::Routes => (2, 0),
            Type::Experts(w) => (3, w),
            Type::Cache => (4, 0),
        };
        let dtype = if kind != 0 {
            "opaque".into()
        } else {
            match n.op.as_str() {
                "slice" | "repeat" | "concat" => p.instructions[inputs[0]].dtype.clone(),
                "optional_norm" => {
                    let exists = inventory.map(|v| {
                        v.tensors
                            .contains_key(&format!("{}.weight", n.attrs["weight"]))
                    });
                    if exists == Some(false) {
                        p.instructions[inputs[0]].dtype.clone()
                    } else {
                        "bf16".into()
                    }
                }
                "allreduce" => n.attrs.get("dtype").cloned().unwrap_or("f32".into()),
                "expert_reduce" | "add" => "f32".into(),
                _ => "bf16".into(),
            }
        };
        let kernel = if n.op == "linear" {
            if n.attrs.contains_key("paired_weight") {
                "paired_bf16"
            } else {
                match inventory
                    .and_then(|v| v.tensors.get(&format!("{}.weight", n.attrs["weight"])))
                    .map(|v| v.dtype.as_str())
                {
                    Some("BF16") => "bf16",
                    Some("I8") => "int8",
                    None => "unbound",
                    Some(_) => return Err(invalid("unsupported physical linear weight dtype")),
                }
            }
        } else if n.op == "optional_norm" {
            match inventory {
                Some(v)
                    if !v
                        .tensors
                        .contains_key(&format!("{}.weight", n.attrs["weight"])) =>
                {
                    "alias"
                }
                Some(_) => "norm",
                None => "unbound",
            }
        } else {
            n.op.as_str()
        };
        let id = p.instructions.len();
        p.instructions.push(Instruction { logical_node: logical, name: n.name.clone(),
            op: match n.op.as_str() { "add" => "add_f32".into(), "mome" => "mome_inplace".into(), _=>n.op.clone() }, inputs,
            attrs: n.attrs.clone(), kind, width, dtype, kernel: if n.op == "mome" { "mome_inplace".into() } else { kernel.into() },
            reason: "Registered backend recipe; checkpoint-bound selection, no runtime alternative search".into(),
        });
        let output = if n.op == "add" {
            let out = p.instructions.len();
            p.instructions.push(Instruction { logical_node: logical, name: format!("__round_{out}"), op: "cast".into(), inputs: vec![id],
                attrs: BTreeMap::from([("dtype".into(), "bf16".into())]), kind: 0, width, dtype: "bf16".into(), kernel: "aclnnCast".into(),
                reason: "Preserve authored residual-add BF16 rounding boundary; cannot cancel through an FP32 consumer".into() });
            out
        } else {
            id
        };
        outputs.push(output);
    }
    p.output = outputs[graph.output];
    Ok(p)
}

impl Program {
    pub fn validate(&self) -> Result<()> {
        if self.format != "inferfabric.token-program.v2"
            || self.instructions.is_empty()
            || self.instructions.len() > 10000
            || self.output >= self.instructions.len()
            || self.max_context == 0
            || self.max_context > 512
        {
            return Err(invalid("physical program bounds"));
        }
        let mut names = std::collections::BTreeSet::new();
        for (id, i) in self.instructions.iter().enumerate() {
            if i.kind > 4
                || i.width > 200000
                || i.inputs.len() > 8
                || i.attrs.len() > 16
                || !names.insert(&i.name)
                || i.inputs.iter().any(|&j| j >= id)
                || (i.kind == 0 && !matches!(i.dtype.as_str(), "bf16" | "f32"))
                || (i.kind != 0 && i.dtype != "opaque")
            {
                return Err(invalid("physical instruction contract"));
            }
            for s in [&i.name, &i.op, &i.dtype, &i.kernel]
                .into_iter()
                .chain(i.attrs.keys())
                .chain(i.attrs.values())
            {
                if s.is_empty() || s.len() > 1024 || s.contains('\0') {
                    return Err(invalid("physical instruction string"));
                }
            }
            if i.op == "cast"
                && (i.inputs.len() != 1
                    || i.attrs.get("dtype") != Some(&i.dtype)
                    || self.instructions[i.inputs[0]].kind != 0
                    || self.instructions[i.inputs[0]].dtype == i.dtype
                    || self.instructions[i.inputs[0]].width != i.width)
            {
                return Err(invalid("physical cast contract"));
            }
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        fn num(b: &mut Vec<u8>, n: usize) {
            b.extend_from_slice(&(n as u32).to_le_bytes());
        }
        fn text(b: &mut Vec<u8>, s: &str) {
            num(b, s.len());
            b.extend_from_slice(s.as_bytes());
        }
        if self.instructions.iter().any(|i| i.kernel == "unbound") {
            return Err(invalid("cannot execute unbound kernel selection"));
        }
        let mut b = b"IFTOKN02".to_vec();
        for n in [
            self.max_context,
            self.eos,
            self.output,
            self.instructions.len(),
        ] {
            num(&mut b, n);
        }
        for i in &self.instructions {
            num(&mut b, i.kind);
            num(&mut b, i.width);
            text(&mut b, &i.name);
            text(&mut b, &i.op);
            num(&mut b, i.inputs.len());
            for &j in &i.inputs {
                num(&mut b, j);
            }
            num(&mut b, i.attrs.len());
            for (k, v) in &i.attrs {
                text(&mut b, k);
                text(&mut b, v);
            }
            text(&mut b, &i.dtype);
            text(&mut b, &i.kernel);
            num(&mut b, i.logical_node);
        }
        Ok(b)
    }
}

/// Inspection metadata for backend recipes, separate from executable instructions.
/// A vendor API may enqueue several device kernels; do not label it a measured launch.
pub fn recipes(p: &Program, graph: &Graph) -> Vec<serde_json::Value> {
    p.instructions.iter().map(|i| {
        let n = &graph.nodes[i.logical_node];
        let (apis, internal, materialization): (&[&str], &str, &str) = match i.op.as_str() {
            "cast" => (&["aclnnCast"], "", "new device output; shared by identical consumers"),
            "copy" => (&["aclrtMemcpyAsync"], "private writable input; source preserved", "new device output"),
            "mome_inplace" => (&["aclnnAiInfraFusedCausalConv1d"], "mutates exclusively owned input; copy decision is an explicit preceding instruction", "output aliases input; persistent recurrent state"),
            "add_f32" => (&["aclnnAdd"], "FP32 arithmetic; BF16 rounding is a separate instruction", "new device output"),
            "embedding" => (&["aclnnIndexSelect"], "device token index", "new device output"),
            "linear" => match i.kernel.as_str() {
                "bf16" => (&["aclnnMatmul"], "BF16 input/output", "new device output"),
                "paired_bf16" => (&["aclnnMatmul"], "BF16 paired rows; two parameter D2D copies once before capture", "persistent packed weights + new output"),
                "int8" => (&["aclnnDynamicQuant", "aclnnQuantMatmulV5"], "BF16 → INT8 quantization with FP32 scale; BF16 result", "quantized input + scale + output"),
                _ => (&[], "checkpoint inventory required to select BF16 or INT8", "unresolved"),
            },
            "allreduce" => (&["HcclAllReduce"], "sum; dtype is explicit; runtime chooses HCCL transport algorithm", "new device output"),
            "gather" => (&["HcclAllGather"], "BF16 rank-order concatenation", "new device output"),
            "slice" | "mhc_input" => (&[], "shared allocation lease", "alias; no copy"),
            "repeat" | "concat" => (&["aclrtMemcpyAsync"], "device-to-device materialization", "new contiguous output"),
            "norm" | "optional_norm" if i.kernel != "alias" => (&["aclnnRmsNorm"], "BF16 operands; vendor accumulation contract", "new output + inverse RMS"),
            "optional_norm" => (&[], "checkpoint binding absent", "alias; no copy"),
            "mome" => (&["aclrtMemcpyAsync", "aclnnAiInfraFusedCausalConv1d"], "input staging protects source from vendor in-place mutation", "persistent mutable input/state"),
            "mhc_pre" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre", "aclnnManifoldConstrainedHyperConnectionSinkhornEnhance"], "BF16 residual; FP32 phi, mixing and Sinkhorn; residual borrowed", "persistent state; source alias"),
            "mhc_merge" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre"], "BF16 residual; FP32 phi and mixing", "persistent state; source alias"),
            "mhc_post" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPost"], "BF16 residual and branch; FP32 mixing", "persistent output"),
            "rope" => (&["aclnnRotaryPositionEmbedding", "aclnnCast"], "FP32 rotation → BF16 rounding", "FP32 temporary + BF16 result"),
            "mla_query" => (&["aclnnBatchMatMul", "aclnnCast", "aclnnRotaryPositionEmbedding", "aclnnCast", "aclnnCat"], "strided BF16 → FP32 rotary extraction → BF16 rounding; packed with latent query", "immutable weights packed once; rotary and concatenation temporaries"),
            "mla_value" => (&["aclnnBatchMatMul"], "BF16 heads", "immutable weights packed once; new output"),
            "sink_normalize" => (&["aclnnRmsNorm", "aclrtMemcpyAsync"], "initialization only after first preparation", "persistent normalized sink"),
            "cache_append" => (&["aclnnInplaceIndexCopy"], "cache mutation; latent and rotary packing", "persistent paged cache"),
            "cache_norm_rope_append" => (&["aclrtMemcpyAsync", "aclnnAiInfraKvRmsNormRopeCache"], "pack latent/rotary; fused BF16 normalization, rotation and cache write", "packed input + persistent paged cache"),
            "latent_attention" => (&["aclnnAiInfraFusedInferAttentionSinkMetadata", "aclnnAiInfraFusedInferAttentionSink"], "device metadata once per token; BF16 attention with learned sinks", "persistent metadata/cache; new output"),
            "swiglu" => (&["aclnnSwiGlu (compiled ABI)"], "compiled CANN recipe; packed equal halves; two-input form first concatenates", "new output; borrowed packed input"),
            "router" => (&["aclnnMatmul", "aclnnMoeGatingTopK"], "FP32 projection and device top-k selection; selection recipe remains composite", "persistent logits/ids/scores"),
            "expert_linear" => (&["aclnnMoeInitRoutingV3", "aclnnGroupedMatmulV5"], "filter/sort/quantize only owner-local routes; INT8 grouped matmul", "packed expert rows, counts and inverse indices"),
            "expert_swiglu_down" => (&["aclnnDequantSwigluQuant", "aclnnGroupedMatmulV5"], "fused activation/quantization; INT8 grouped down projection", "packed expert output + scales"),
            "expert_reduce" => (&["aclnnCast", "aclnnMoeFinalizeRoutingV2"], "BF16 expert output → FP32; weighted local route reduction", "FP32 local partial output"),
            _ => (&[], "unresolved backend recipe", "unresolved"),
        };
        let placement = if matches!(i.op.as_str(), "cast"|"copy") { "inherited_input" } else { n.attrs.get("placement").map(String::as_str).unwrap_or(match n.op.as_str() {
            "expert_linear" | "expert_swiglu_down" | "expert_reduce" => "ep_owner_local",
            "mla_query" | "mla_value" | "latent_attention" => "tp_local_heads",
            "router" => "replicated_routing", "allreduce"|"gather" => "replicated_output", _ => "inherited",
        }) };
        let communication = match i.op.as_str() {
            "allreduce" => "AllReduce(sum), ranks 0–7",
            "gather" => "AllGather, ranks 0–7",
            "expert_linear" | "expert_swiglu_down" | "expert_reduce" => "EP owner-local compute; explicit downstream AllReduce combines partials",
            _ => "none in this instruction",
        };
        serde_json::json!({"apis":apis,"internal":internal,"materialization":materialization,"placement":placement,"communication":communication,
            "granularity":if matches!(i.op.as_str(), "cast"|"add_f32"|"slice"|"gather"|"allreduce") {"instruction"} else {"backend recipe"}})
    }).collect()
}
