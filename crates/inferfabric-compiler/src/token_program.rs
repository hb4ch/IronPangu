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

fn emit(
    p: &mut Program,
    logical: usize,
    name: &str,
    op: &str,
    inputs: Vec<usize>,
    attrs: BTreeMap<String, String>,
    kind: usize,
    width: usize,
    dtype: &str,
) -> usize {
    let id = p.instructions.len();
    p.instructions.push(Instruction {
        logical_node: logical,
        name: format!("{name}__{op}_{id}"),
        op: op.into(),
        inputs,
        attrs,
        kind,
        width,
        dtype: dtype.into(),
        kernel: if op == "cast" {
            "aclnnCast".into()
        } else {
            op.into()
        },
        reason: "Expanded backend invocation; storage and dependencies are explicit".into(),
    });
    id
}
pub fn lower(graph: &Graph, inventory: Option<&Inventory>) -> Result<Program> {
    let types = graph.validate()?;
    let mut p = Program {
        format: "inferfabric.token-program.v2".into(), max_context: graph.max_context,
        eos: graph.eos, output: 0, instructions: vec![], conversions_shared: 0, copies_eliminated: 0,
        limitations: vec!["Per-token vendor composites are expanded into declared API invocations, explicit casts, packing and initialization of outputs. Parameter/state preparation remains a declared initialization recipe; vendor-internal kernel launches are opaque. Hardware preparation binds fixed arena slots; this is not a cost-optimized arena packing.".into()],
    };
    let mut outputs = vec![];
    let mut casts = BTreeMap::new();
    let mut attention_metadata = None;
    let mut route_quantizations = BTreeMap::new();
    for (logical, n) in graph.nodes.iter().enumerate() {
        if n.attrs.get("cache_dtype").is_some_and(|v| v != "bf16") {
            return Err(invalid(format!(
                "{}: requested {} cache has no qualified Ascend attention/cache kernel; refusing BF16 substitution",
                n.name, n.attrs["cache_dtype"]
            )));
        }
        let mut inputs: Vec<usize> = n.inputs.iter().map(|&i| outputs[i]).collect();
        for (slot, input) in inputs.iter_mut().enumerate() {
            if p.instructions[*input].kind != 0 {
                continue;
            }
            let wanted = match n.op.as_str() {
                "slice" | "repeat" | "concat" | "cast" => None,
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
            Type::Quantized(w) => (5, w),
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
                "cast" => n.attrs["dtype"].clone(),
                _ => "bf16".into(),
            }
        };
        if n.op == "cast" && p.instructions[inputs[0]].dtype == dtype {
            outputs.push(inputs[0]);
            continue;
        }
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
        } else if n.op == "cast" {
            "aclnnCast"
        } else {
            n.op.as_str()
        };
        if n.op == "linear"
            && p.instructions[inputs[0]].kind == 5
            && !matches!(kernel, "int8" | "unbound")
        {
            return Err(invalid("quantized input requires INT8 projection weights"));
        }
        if n.op == "linear" && kernel == "int8" && p.instructions[inputs[0]].kind == 0 {
            let id = p.instructions.len();
            let width = p.instructions[inputs[0]].width;
            p.instructions.push(Instruction {
                logical_node: logical,
                name: format!("__quant_{id}"),
                op: "quantize".into(),
                inputs: vec![inputs[0]],
                attrs: BTreeMap::from([
                    ("dtype".into(), "int8".into()),
                    ("scheme".into(), "per_token_symmetric".into()),
                    ("scale_dtype".into(), "f32".into()),
                ]),
                kind: 5,
                width,
                dtype: "opaque".into(),
                kernel: "quantize".into(),
                reason: "Explicit activation quantization for selected INT8 projection".into(),
            });
            inputs[0] = id;
        }
        if n.op == "swiglu" && inputs.len() == 2 {
            inputs = vec![emit(
                &mut p,
                logical,
                &n.name,
                "concat",
                inputs,
                BTreeMap::new(),
                0,
                width * 2,
                "bf16",
            )];
        }
        if n.op == "cache_norm_rope_append" {
            let packed = emit(
                &mut p,
                logical,
                &n.name,
                "concat",
                inputs[..2].to_vec(),
                BTreeMap::new(),
                0,
                576,
                "bf16",
            );
            inputs = vec![packed, inputs[2]];
        }
        if n.op == "expert_reduce" {
            let rows = emit(
                &mut p,
                logical,
                &n.name,
                "expert_values",
                vec![inputs[0]],
                BTreeMap::new(),
                0,
                width * 8,
                "bf16",
            );
            let f32rows = emit(
                &mut p,
                logical,
                &n.name,
                "cast",
                vec![rows],
                BTreeMap::from([("dtype".into(), "f32".into())]),
                0,
                width * 8,
                "f32",
            );
            inputs.insert(0, f32rows);
        }
        if n.op == "mhc_pre" {
            let pre = emit(
                &mut p,
                logical,
                &n.name,
                "mhc_pre_project",
                inputs,
                n.attrs.clone(),
                1,
                0,
                "opaque",
            );
            let sink = emit(
                &mut p,
                logical,
                &n.name,
                "mhc_sinkhorn",
                vec![pre],
                BTreeMap::new(),
                1,
                0,
                "opaque",
            );
            outputs.push(sink);
            continue;
        }
        if n.op == "router" {
            let logits = emit(
                &mut p,
                logical,
                &n.name,
                "router_projection",
                inputs,
                n.attrs.clone(),
                0,
                256,
                "f32",
            );
            let selected = emit(
                &mut p,
                logical,
                &n.name,
                "router_select",
                vec![logits],
                BTreeMap::new(),
                2,
                0,
                "opaque",
            );
            outputs.push(selected);
            continue;
        }
        if n.op == "expert_linear" || n.op == "expert_swiglu_down" {
            let q = if n.op == "expert_linear" {
                let key = (inputs[0], inputs[1]);
                if let Some(&q) = route_quantizations.get(&key) {
                    q
                } else {
                    let q = emit(
                        &mut p,
                        logical,
                        &n.name,
                        "route_quantize",
                        inputs,
                        BTreeMap::new(),
                        6,
                        2560,
                        "opaque",
                    );
                    route_quantizations.insert(key, q);
                    q
                }
            } else {
                let packed = if inputs.len() == 3 {
                    emit(
                        &mut p,
                        logical,
                        &n.name,
                        "expert_pack",
                        inputs[..2].to_vec(),
                        BTreeMap::new(),
                        3,
                        2048,
                        "opaque",
                    )
                } else {
                    inputs[0]
                };
                emit(
                    &mut p,
                    logical,
                    &n.name,
                    "expert_act_quantize",
                    vec![packed],
                    BTreeMap::new(),
                    6,
                    1024,
                    "opaque",
                )
            };
            let zero = emit(
                &mut p,
                logical,
                &n.name,
                "expert_zero",
                vec![q],
                BTreeMap::from([("width".into(), width.to_string())]),
                3,
                width,
                "opaque",
            );
            let result = emit(
                &mut p,
                logical,
                &n.name,
                "expert_gmm",
                vec![q, zero],
                n.attrs.clone(),
                3,
                width,
                "opaque",
            );
            outputs.push(result);
            continue;
        }
        if n.op == "latent_attention" {
            let meta = if let Some(id) = attention_metadata {
                id
            } else {
                let zero = emit(
                    &mut p,
                    logical,
                    &n.name,
                    "attention_metadata_zero",
                    vec![],
                    BTreeMap::new(),
                    7,
                    0,
                    "opaque",
                );
                let meta = emit(
                    &mut p,
                    logical,
                    &n.name,
                    "attention_metadata",
                    vec![zero],
                    BTreeMap::new(),
                    7,
                    0,
                    "opaque",
                );
                attention_metadata = Some(meta);
                meta
            };
            inputs.push(meta);
        }
        // Expand the query recipe before serialization. Every conversion and
        // materialization has its own SSA value and scheduling position.
        if n.op == "mla_query" || n.op == "rope" {
            let mut emit = |op: &str,
                            ins: Vec<usize>,
                            attrs: BTreeMap<String, String>,
                            width: usize,
                            dtype: &str| {
                let id = p.instructions.len();
                p.instructions.push(Instruction {
                    logical_node: logical,
                    name: format!("{}__{}_{}", n.name, op, id),
                    op: op.into(),
                    inputs: ins,
                    attrs,
                    kind: 0,
                    width,
                    dtype: dtype.into(),
                    kernel: if op == "cast" {
                        "aclnnCast".into()
                    } else {
                        op.into()
                    },
                    reason: "Expanded composite; single declared operation with explicit storage"
                        .into(),
                });
                id
            };
            let mut projection = None;
            let rotary = if n.op == "mla_query" {
                let layout = n
                    .attrs
                    .get("input_layout")
                    .cloned()
                    .unwrap_or("replicated".into());
                let attrs = BTreeMap::from([("input_layout".into(), layout)]);
                let mut project_attrs = attrs.clone();
                project_attrs.insert("weight".into(), n.attrs["weight"].clone());
                projection = Some(emit(
                    "query_project",
                    inputs.clone(),
                    project_attrs,
                    3072,
                    "bf16",
                ));
                emit("query_rotary_extract", inputs, attrs, 384, "f32")
            } else {
                inputs[0]
            };
            let rw = if projection.is_some() { 384 } else { width };
            let rotated = emit(
                "rope_f32",
                vec![rotary],
                BTreeMap::from([("theta".into(), n.attrs["theta"].clone())]),
                rw,
                "f32",
            );
            let rounded = emit(
                "cast",
                vec![rotated],
                BTreeMap::from([("dtype".into(), "bf16".into())]),
                rw,
                "bf16",
            );
            let output = if let Some(projected) = projection {
                emit(
                    "concat_columns",
                    vec![projected, rounded],
                    BTreeMap::from([("rows".into(), "6".into())]),
                    width,
                    "bf16",
                )
            } else {
                rounded
            };
            outputs.push(output);
            continue;
        }
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
            if i.kind > 7
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
            "mhc_pre_project" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre"], "BF16 residual; FP32 mixing; Sinkhorn is separate", "persistent mHC outputs and residual alias"),
            "mhc_sinkhorn" => (&["aclnnManifoldConstrainedHyperConnectionSinkhornEnhance"], "20 FP32 normalization iterations", "persistent Sinkhorn output; preserves mHC handle"),
            "router_projection" => (&["aclnnMatmul"], "FP32 gate projection; top-k is separate", "persistent FP32 logits"),
            "router_select" => (&["aclnnMoeGatingTopK"], "sigmoid/bias group top-k; ids remain on device", "INT32 ids + FP32 scores and normalized logits"),
            "route_quantize" => (&["aclnnMoeInitRoutingV3"], "owner-local route filtering/sorting and INT8 quantization", "INT8 [8,width], FP32 [8] scales, INT64 [32] groups, INT32 [8] inverse indices"),
            "expert_act_quantize" => (&["aclnnDequantSwigluQuant"], "packed BF16 SwiGLU and INT8 quantization", "INT8 [8,width] + FP32 [8] scales; borrowed group metadata"),
            "expert_zero" => (&["aclrtMemsetAsync"], "define unselected expert rows as zero", "fresh BF16 [8,width]; exclusive input to grouped matmul"),
            "expert_gmm" => (&["aclnnGroupedMatmulV5"], "INT8 grouped matmul; quantization and zeroing are separate", "writes exclusively owned initialized output; packed weights prepared once"),
            "expert_values" => (&[], "view packed expert rows as a tensor", "alias; no copy"),
            "expert_pack" => (&["aclrtMemcpyAsync"], "interleave gate/up rows for the same routing decision", "new BF16 packed rows; borrowed routing metadata"),
            "attention_metadata_zero" => (&["aclrtMemsetAsync"], "initialize metadata output once per token", "persistent metadata buffer"),
            "attention_metadata" => (&["aclnnAiInfraFusedInferAttentionSinkMetadata"], "device sequence length; shared by attention instructions", "persistent device metadata"),
            "cast" => (&["aclnnCast"], "", "new device output; shared by identical consumers"),
            "quantize" => (&["aclnnDynamicQuant"], "BF16 → signed INT8; per-token symmetric FP32 scale", "explicit INT8 payload + FP32 scale"),
            "copy" => (&["aclrtMemcpyAsync"], "private writable input; source preserved", "new device output"),
            "mome_inplace" => (&["aclnnAiInfraFusedCausalConv1d"], "mutates exclusively owned input; copy decision is an explicit preceding instruction", "output aliases input; persistent recurrent state"),
            "add_f32" => (&["aclnnAdd"], "FP32 arithmetic; BF16 rounding is a separate instruction", "new device output"),
            "embedding" => (&["aclnnIndexSelect"], "device token index", "new device output"),
            "linear" => match i.kernel.as_str() {
                "bf16" => (&["aclnnMatmul"], "BF16 input/output", "new device output"),
                "paired_bf16" => (&["aclnnMatmul"], "BF16 paired rows; two parameter D2D copies once before capture", "persistent packed weights + new output"),
                "int8" => (&["aclnnQuantMatmulV5"], "consumes explicit INT8 payload and FP32 scale; BF16 result", "new BF16 output"),
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
            "query_project" => (&["aclnnBatchMatMul"], "strided local query/weight views; BF16 input/output", "new BF16 latent query"),
            "query_rotary_extract" => (&["aclnnCast"], "strided BF16 rotary columns to contiguous FP32", "new FP32 rotary input"),
            "rope_f32" => (&["aclnnRotaryPositionEmbedding"], "FP32 rotation; rounding is a separate instruction", "new FP32 output"),
            "concat_columns" => (&["aclnnCat"], "concatenate columns of declared row matrices", "new contiguous output"),
            "rope" => (&["aclnnRotaryPositionEmbedding", "aclnnCast"], "FP32 rotation → BF16 rounding", "FP32 temporary + BF16 result"),
            "mla_query" => (&["aclnnBatchMatMul", "aclnnCast", "aclnnRotaryPositionEmbedding", "aclnnCast", "aclnnCat"], "strided BF16 → FP32 rotary extraction → BF16 rounding; packed with latent query", "immutable weights packed once; rotary and concatenation temporaries"),
            "mla_value" => (&["aclnnBatchMatMul"], "BF16 heads", "immutable weights packed once; new output"),
            "sink_normalize" => (&["aclnnRmsNorm", "aclrtMemcpyAsync"], "initialization only after first preparation", "persistent normalized sink"),
            "cache_append" => (&["aclnnInplaceIndexCopy"], "cache mutation; latent and rotary packing", "persistent paged cache"),
            "cache_norm_rope_append" => (&["aclnnAiInfraKvRmsNormRopeCache"], "explicit packed input; fused BF16 normalization, rotation and cache write", "persistent paged cache + required vendor outputs; sink copy during preparation only"),
            "latent_attention" => (&["aclnnAiInfraFusedInferAttentionSink"], "consumes explicit metadata dependency; BF16 attention with learned sinks", "persistent metadata/cache; new output"),
            "swiglu" => (&["aclnnSwiGlu (compiled ABI)"], "compiled CANN recipe; packed equal halves; two-input form first concatenates", "new output; borrowed packed input"),
            "router" => (&["aclnnMatmul", "aclnnMoeGatingTopK"], "FP32 projection and device top-k selection; selection recipe remains composite", "persistent logits/ids/scores"),
            "expert_linear" => (&["aclnnMoeInitRoutingV3", "aclnnGroupedMatmulV5"], "filter/sort/quantize only owner-local routes; INT8 grouped matmul", "packed expert rows, counts and inverse indices"),
            "expert_swiglu_down" => (&["aclnnDequantSwigluQuant", "aclnnGroupedMatmulV5"], "fused activation/quantization; INT8 grouped down projection", "packed expert output + scales"),
            "expert_reduce" => (&["aclnnMoeFinalizeRoutingV2"], "consumes explicitly converted FP32 expert rows; weighted local reduction", "FP32 local partial output"),
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
            "route_quantize" | "expert_act_quantize" | "expert_gmm" | "expert_zero" | "expert_pack" | "expert_linear" | "expert_swiglu_down" | "expert_reduce" => "EP owner-local compute; explicit downstream AllReduce combines partials",
            _ => "none in this instruction",
        };
        let preparation = match i.op.as_str() {
            "linear" if i.kernel=="paired_bf16" => "Once before replay: two BF16 parameter copies pack the selected projection rows.",
            "expert_gmm" => "Once before replay: copy owner-local INT8 weights and lossless BF16 scales into grouped banks. No FP32 scale round-trip.",
            "sink_normalize" => "Once before replay: normalize checkpoint sink rows and pack BF16 latent/rotary constants; subsequent tokens borrow the prepared result.",
            "cache_append" | "cache_norm_rope_append" => "Once before replay: copy prepared constant sinks into persistent cache state; initialize metadata from control inputs.",
            "mhc_pre_project" | "mhc_merge" => "Once before replay: fold scalar checkpoint parameters and upload prepared mHC coefficients.",
            "router_projection" => "Once before replay: transpose/convert checkpoint gate parameters and upload FP32 weights and bias.",
            "mome_inplace" => "Once before replay: transpose convolution weights and initialize recurrent state and control indices.",
            _ => "No parameter/state copy recipe in this instruction.",
        };
        let fields:Vec<serde_json::Value>=match i.kind {
            0=>vec![serde_json::json!({"name":"value","dtype":i.dtype,"shape":[i.width]})],
            2=>vec![serde_json::json!({"name":"expert_ids","dtype":"int32","shape":[8]}),serde_json::json!({"name":"scores","dtype":"f32","shape":[8]})],
            3=>vec![serde_json::json!({"name":"packed_rows","dtype":"bf16","shape":[8,i.width]}),serde_json::json!({"name":"groups","dtype":"int64","shape":[32]}),serde_json::json!({"name":"inverse_indices","dtype":"int32","shape":[8]})],
            5=>vec![serde_json::json!({"name":"quantized","dtype":"int8","shape":[1,i.width]}),serde_json::json!({"name":"scales","dtype":"f32","shape":[1]})],
            6=>vec![serde_json::json!({"name":"quantized_rows","dtype":"int8","shape":[8,i.width]}),serde_json::json!({"name":"scales","dtype":"f32","shape":[8]}),serde_json::json!({"name":"groups","dtype":"int64","shape":[32]}),serde_json::json!({"name":"inverse_indices","dtype":"int32","shape":[8]})],
            7=>vec![serde_json::json!({"name":"metadata_effect","dtype":"int32","shape":[1024]})],
            _=>vec![],
        };
        serde_json::json!({"preparation":preparation,"value_fields":fields,"apis":apis,"internal":internal,"materialization":materialization,"placement":placement,"communication":communication,
            "granularity":if i.op=="sink_normalize" {"initialization recipe"} else {"instruction"}})
    }).collect()
}
