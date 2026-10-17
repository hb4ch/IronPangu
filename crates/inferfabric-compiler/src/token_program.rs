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
    /// Resident histories, independent of the one-token invocation batch size.
    pub resident_sequences: usize,
    pub prefill_tokens: usize,
    /// A bounded variable-query launch envelope, never an ACLGraph family.
    #[serde(default)]
    pub uncaptured_prefill: bool,
    /// Explicit exact shapes. Empty preserves the legacy single-size contract.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefill_variants: Vec<usize>,
    #[serde(default)]
    pub shared_pages: bool,
    pub eos: usize,
    pub output: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_rows: Option<crate::output_rows::OutputRows>,
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
        kernel: if op == "expert_zero" {
            "aclnnInplaceZero".into()
        } else if op == "cast" {
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
        format: "inferfabric.token-program.v2".into(), max_context: graph.max_context, resident_sequences: 1, prefill_tokens: 1, uncaptured_prefill: false, prefill_variants: vec![], shared_pages: false,
        eos: graph.eos, output: 0, output_rows: None, instructions: vec![], conversions_shared: 0, copies_eliminated: 0,
        limitations: vec!["Per-token vendor composites are expanded into declared API invocations, explicit casts, packing and initialization of outputs. Parameter/state preparation remains a declared initialization recipe; vendor-internal kernel launches are opaque. Hardware preparation binds fixed arena slots; this is not a cost-optimized arena packing.".into()],
    };
    let mut outputs = vec![];
    let mut casts = BTreeMap::new();
    let mut attention_metadata = BTreeMap::new();
    let mut route_quantizations = BTreeMap::new();
    for (logical, n) in graph.nodes.iter().enumerate() {
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
            Type::Indices(w) => (8, w),
            Type::IndexCache(w) => (9, w),
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
        if n.op == "cache_append" && n.attrs.get("cache_dtype").is_some_and(|v|v!="bf16") {
            let id=emit(&mut p,logical,&n.name,"cache_quantize_append",inputs,n.attrs.clone(),4,0,"opaque");
            outputs.push(id);continue;
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
            if n.attrs.get("cache_dtype").is_some_and(|v|v=="int8_ds_mla") {
                let cache=emit(&mut p,logical,&n.name,"cache_c8_norm_rope_append",inputs,n.attrs.clone(),4,0,"opaque");
                outputs.push(cache);continue;
            }
            if n.attrs.get("cache_dtype").is_some_and(|v| v != "bf16") {
                let normalized = emit(&mut p, logical, &n.name, "cache_normalize",
                    vec![packed], n.attrs.iter().filter(|(k,_)| matches!(k.as_str(), "weight"|"eps"|"theta")).map(|(k,v)|(k.clone(),v.clone())).collect(),
                    0, 576, "bf16");
                let attrs = n.attrs.iter().filter(|(k,_)| k.starts_with("cache_")).map(|(k,v)|(k.clone(),v.clone())).collect();
                let cache = emit(&mut p, logical, &n.name, "cache_quantize_append",
                    vec![normalized, inputs[1]], attrs, 4, 0, "opaque");
                outputs.push(cache);
                continue;
            }

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
        if n.op == "index_cache_append" && n.attrs.get("cache_dtype").is_some_and(|v|v=="int8") {
            let q=emit(&mut p,logical,&n.name,"index_quantize",inputs,
                BTreeMap::from([("block_width".into(),"128".into())]),5,128,"opaque");
            let q=emit(&mut p,logical,&n.name,"quant_scales_f16",vec![q],BTreeMap::new(),5,128,"opaque");
            let out=emit(&mut p,logical,&n.name,"index_cache_append",vec![q],n.attrs.clone(),9,128,"opaque");
            outputs.push(out);continue;
        }
        if n.op == "sparse_index" {
            if p.instructions[inputs[2]].attrs.get("cache_dtype").is_some_and(|v|v=="int8") {
                let q=emit(&mut p,logical,&n.name,"index_quantize",vec![inputs[0]],
                    BTreeMap::from([("block_width".into(),"128".into())]),5,3072,"opaque");
                inputs[0]=emit(&mut p,logical,&n.name,"quant_scales_f16",vec![q],BTreeMap::new(),5,3072,"opaque");
                inputs[1]=emit(&mut p,logical,&n.name,"cast",vec![inputs[1]],
                    BTreeMap::from([("dtype".into(),"f16".into())]),0,24,"f16");
            }
            let meta=if let Some(id)=p.instructions.iter().position(|i|i.op=="packed_attention_metadata") { id } else {
                emit(&mut p,logical,&n.name,"packed_attention_metadata",vec![],BTreeMap::new(),7,0,"opaque")
            };
            inputs.push(meta);
            let out=emit(&mut p,logical,&n.name,"sparse_index",inputs,n.attrs.clone(),8,2048,"opaque");
            outputs.push(out);continue;
        }
        if n.op == "latent_attention" && p.instructions[inputs[1]].op=="cache_c8_norm_rope_append" {
            let meta=if let Some(id)=p.instructions.iter().position(|i|i.op=="packed_attention_metadata") { id } else {
                emit(&mut p,logical,&n.name,"packed_attention_metadata",vec![],BTreeMap::new(),7,0,"opaque")
            };
            inputs.push(meta);
            let op=if n.attrs["selection"]=="learned_indices" {"indexed_packed_attention"} else {"packed_latent_attention"};
            let out=emit(&mut p,logical,&n.name,op,inputs,n.attrs.clone(),0,3072,"bf16");
            outputs.push(out);continue;
        }
        if n.op == "latent_attention" {
            if n.attrs["selection"]=="learned_indices" {return Err(invalid("learned-index attention currently requires packed C8 cache"));}
            let window = n.attrs["window"].clone();
            let meta = if let Some(&id) = attention_metadata.get(&window) {
                id
            } else {
                let zero = emit(
                    &mut p,
                    logical,
                    &n.name,
                    "attention_metadata_zero",
                    vec![],
                    BTreeMap::from([("window".into(),window.clone())]),
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
                    BTreeMap::from([("window".into(),window.clone())]),
                    7,
                    0,
                    "opaque",
                );
                attention_metadata.insert(window,meta);
                meta
            };
            inputs.push(meta);
            if p.instructions[inputs[1]].op == "cache_quantize_append" {
                let attrs = p.instructions[inputs[1]].attrs.clone();
                let decoded = emit(&mut p, logical, &n.name, "cache_dequantize",
                    vec![inputs[1]], attrs, 0, graph.max_context.div_ceil(16) * 16 * 576, "bf16");
                inputs.push(decoded);
            }
        }
        if n.op == "rotary_prefix" {
            let rows = n.integer("heads")?;
            let columns = n.integer("head_dim")?;
            let rotary = n.integer("rotary_dim")?;
            let mut parts = vec![];
            for (start, count) in [(0, rotary), (rotary, columns - rotary)] {
                parts.push(emit(&mut p, logical, &n.name, "column_slice", inputs.clone(),
                    BTreeMap::from([("rows".into(), rows.to_string()), ("start".into(), start.to_string()),
                        ("width".into(), count.to_string())]), 0, rows * count, "bf16"));
            }
            let rotated = emit(&mut p, logical, &n.name, "rope_bf16", vec![parts[0]],
                BTreeMap::from([("theta".into(), n.attrs["theta"].clone())]), 0, rows * rotary, "bf16");
            let output = emit(&mut p, logical, &n.name, "concat_columns", vec![rotated, parts[1]],
                BTreeMap::from([("rows".into(), rows.to_string())]), 0, width, "bf16");
            outputs.push(output);
            continue;
        }
        // Expand the query recipe before serialization. Every conversion and
        // materialization has its own SSA value and scheduling position.
        if n.op == "mla_query" || n.op == "rope" {
            let bf16_rotary = n.attrs.get("rotary_dtype").is_some_and(|v| v == "bf16");
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
                if bf16_rotary {
                    emit("column_slice", inputs, BTreeMap::from([
                        ("rows".into(), "6".into()), ("start".into(), "128".into()),
                        ("width".into(), "64".into()),
                    ]), 384, "bf16")
                } else {
                    emit("query_rotary_extract", inputs, attrs, 384, "f32")
                }
            } else {
                inputs[0]
            };
            let rw = if projection.is_some() { 384 } else { width };
            let rotated = emit(
                if bf16_rotary { "rope_bf16" } else { "rope_f32" },
                vec![rotary],
                BTreeMap::from([("theta".into(), n.attrs["theta"].clone())]),
                rw,
                if bf16_rotary { "bf16" } else { "f32" },
            );
            let rounded = if bf16_rotary {
                rotated
            } else {
                emit("cast", vec![rotated],
                    BTreeMap::from([("dtype".into(), "bf16".into())]), rw, "bf16")
            };
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
    pub fn with_output_rows(mut self) -> Result<Self> {
        if !self.uncaptured_prefill { return Err(invalid("compact output rows require uncaptured prefill")); }
        self.output_rows = Some(crate::output_rows::plan(&self.instructions, self.output)?.ok_or_else(|| invalid("no safe output suffix"))?);
        let residents = self.resident_sequences;
        self.with_resident_sequences(residents)
    }
    pub fn with_uncaptured_prefill(mut self) -> Result<Self> {
        self.uncaptured_prefill = true;
        self.prefill_tokens = 4096;
        self.prefill_variants.clear();
        self.batch_head_projections();
        let residents = self.resident_sequences;
        self.with_resident_sequences(residents)
    }
    fn batch_head_projections(&mut self) {
        let previous = std::mem::take(&mut self.instructions);
        let mut outputs = Vec::with_capacity(previous.len());
        for mut instruction in previous {
            instruction.inputs = instruction.inputs.iter().map(|&id| outputs[id]).collect();
            if matches!(instruction.op.as_str(), "query_project" | "mla_value") {
                let query = instruction.op == "query_project";
                let replicated = query && instruction.attrs.get("input_layout").is_some_and(|v|v=="replicated");
                let weight = instruction.attrs["weight"].clone();
                let (k,n) = if query {(128,512)} else {(512,128)};
                let original_name = instruction.name.clone();
                instruction.op = "head_matmul".into();
                instruction.kernel = "aclnnBatchMatMul".into();
                instruction.reason = "Batch all query rows by head; explicit head-major result and separate token-layout materialization".into();
                instruction.attrs = BTreeMap::from([
                    ("weight".into(),weight), ("heads".into(),"6".into()),
                    ("k".into(),k.to_string()), ("n".into(),n.to_string()),
                    ("input_head_stride".into(),if query {"192"} else {"512"}.into()),
                    ("input_rank_stride".into(),if replicated {"1152"} else {"0"}.into()),
                    ("weight_rows".into(),"1536".into()), ("weight_cols".into(),"512".into()),
                    ("weight_offset".into(),if query {"0"} else {"65536"}.into()),
                    ("weight_head_stride".into(),"131072".into()),
                    ("weight_k_stride".into(),if query {"512"} else {"1"}.into()),
                    ("weight_n_stride".into(),if query {"1"} else {"512"}.into()),
                ]);
                let id = self.instructions.len();
                let mut layout = instruction.clone();
                layout.name = format!("{original_name}__token_layout");
                layout.op = "head_to_token".into();
                layout.kernel = "aclnnCast".into();
                layout.inputs = vec![id];
                layout.attrs = BTreeMap::from([("heads".into(),"6".into()),("width".into(),n.to_string())]);
                layout.reason = "Explicit BF16 [heads,tokens,width] to [tokens,heads,width]; one-token decode is an alias".into();
                self.instructions.push(instruction);
                self.instructions.push(layout);
            } else if matches!(instruction.op.as_str(),"gather"|"repeat") && instruction.dtype=="bf16" {
                let gather=instruction.op=="gather";
                let heads=if gather {8} else {instruction.attrs["count"].parse::<usize>().unwrap()};
                let width=instruction.width/heads;
                instruction.op=if gather {"gather_heads"} else {"repeat_heads"}.into();
                instruction.kernel=if gather {"HcclAllGather"} else {"aclrtMemcpyAsync"}.into();
                instruction.attrs=BTreeMap::from([("heads".into(),heads.to_string())]);
                instruction.reason="Whole token matrix per rank/repetition; separate explicit token-layout conversion".into();
                let id=self.instructions.len();
                let mut layout=instruction.clone();
                layout.name=format!("{}__token_layout",instruction.name);
                layout.op="head_to_token".into();layout.kernel="aclnnCast".into();
                layout.inputs=vec![id];
                layout.attrs=BTreeMap::from([("heads".into(),heads.to_string()),("width".into(),width.to_string())]);
                layout.reason="Explicit head-major to token-major materialization; decode aliases input".into();
                self.instructions.push(instruction);self.instructions.push(layout);
            } else {
                self.instructions.push(instruction);
            }
            outputs.push(self.instructions.len()-1);
        }
        self.output = outputs[self.output];
    }
    pub fn prefill_counts(&self) -> Vec<usize> {
        if !self.prefill_variants.is_empty() {self.prefill_variants.clone()}
        else if self.prefill_tokens>1 {vec![self.prefill_tokens]} else {vec![]}
    }
    pub fn with_prefill_variants(mut self, counts: Vec<usize>) -> Result<Self> {
        self.uncaptured_prefill = false;
        self.prefill_tokens=*counts.last().ok_or_else(||invalid("empty prefill variant family"))?;
        self.prefill_variants=counts;
        let residents=self.resident_sequences;
        self.with_resident_sequences(residents)
    }
    pub fn with_shared_pages(mut self, enabled: bool) -> Result<Self> {
        self.shared_pages=enabled;
        let count=self.resident_sequences;
        self.with_resident_sequences(count)
    }
    pub fn with_prefill_tokens(mut self, count: usize) -> Result<Self> {
        self.uncaptured_prefill = false;
        self.prefill_tokens = count;
        self.prefill_variants.clear();
        let residents = self.resident_sequences;
        self.with_resident_sequences(residents)
    }
    pub fn with_resident_sequences(mut self, count: usize) -> Result<Self> {
        self.resident_sequences = count;
        self.format = if self.output_rows.is_some() { "inferfabric.token-program.v8" } else if self.uncaptured_prefill { "inferfabric.token-program.v7" } else if !self.prefill_variants.is_empty() { "inferfabric.token-program.v6" } else if self.shared_pages { "inferfabric.token-program.v5" } else if self.prefill_tokens > 1 { "inferfabric.token-program.v4" } else if count == 1 { "inferfabric.token-program.v2" } else { "inferfabric.token-program.v3" }.into();
        self.validate()?;
        Ok(self)
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(rows) = &self.output_rows {
            if !self.uncaptured_prefill || crate::output_rows::plan(&self.instructions, self.output)?.as_ref() != Some(rows) {
                return Err(invalid("output row plan does not match the proven suffix"));
            }
        }
        if self.uncaptured_prefill && (self.prefill_tokens != 4096 || !self.prefill_variants.is_empty()) {
            return Err(invalid("uncaptured prefill requires a 4096-token envelope without captured variants"));
        }
        if self.prefill_variants.len()>5 || self.prefill_variants.iter().any(|n| !matches!(n,2|4|8|16|32)) ||
            self.prefill_variants.windows(2).any(|v|v[0]>=v[1]) ||
            self.prefill_variants.last().is_some_and(|&n|n!=self.prefill_tokens) {
            return Err(invalid("invalid exact prefill variant family"));
        }
        if self.format != if self.output_rows.is_some() { "inferfabric.token-program.v8" } else if self.uncaptured_prefill { "inferfabric.token-program.v7" } else if !self.prefill_variants.is_empty() { "inferfabric.token-program.v6" } else if self.shared_pages { "inferfabric.token-program.v5" } else if self.prefill_tokens > 1 { "inferfabric.token-program.v4" } else if self.resident_sequences == 1 { "inferfabric.token-program.v2" } else { "inferfabric.token-program.v3" }
            || (!self.uncaptured_prefill && !matches!(self.prefill_tokens, 1 | 2 | 4 | 8 | 16 | 32))
            || self.prefill_tokens > self.max_context
            || ((self.prefill_tokens > 1 || self.shared_pages) && self.instructions.iter().any(|i| i.op == "cache_quantize_append"))
            || self.resident_sequences == 0 || self.resident_sequences > 64
            || (self.resident_sequences > 1 && self.instructions.iter().any(|i| i.op == "cache_quantize_append"))
            || self.instructions.is_empty()
            || self.instructions.len() > 10000
            || self.output >= self.instructions.len()
            || self.max_context == 0
            || self.max_context > 65_536
            || (self.max_context>512 && self.instructions.iter().any(|i|matches!(i.op.as_str(),"cache_quantize_append"|"packed_latent_attention")))
        {
            return Err(invalid("physical program bounds"));
        }
        let mut names = std::collections::BTreeSet::new();
        for (id, i) in self.instructions.iter().enumerate() {
            if self.uncaptured_prefill && (matches!(i.op.as_str(),
                "mome"|"query_project"|"query_rotary_extract"|"rope"|"rope_f32"|"mla_query"|"mla_value"|"repeat"|"gather") ||
                (i.op=="sparse_index" && i.inputs.first().and_then(|&input|self.instructions.get(input)).is_none_or(|node|node.kind!=5))) {
                return Err(invalid(format!("uncaptured prefill has no batched lowering for {}",i.op)));
            }
            if i.kind > 9
                || (i.width > 200000 && !(i.op == "cache_dequantize" && i.width == self.max_context.div_ceil(16)*16*576))
                || i.inputs.len() > 8
                || i.attrs.len() > 16
                || !names.insert(&i.name)
                || i.inputs.iter().any(|&j| j >= id)
                || (i.kind == 0 && !matches!(i.dtype.as_str(), "bf16" | "f16" | "f32"))
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
        let mut b = if self.output_rows.is_some() { b"IFTOKN08" } else if self.uncaptured_prefill { b"IFTOKN07" } else if !self.prefill_variants.is_empty() { b"IFTOKN06" } else if self.shared_pages { b"IFTOKN05" } else if self.prefill_tokens > 1 { b"IFTOKN04" } else if self.resident_sequences == 1 { b"IFTOKN02" } else { b"IFTOKN03" }.to_vec();
        for n in [
            self.max_context,
            self.eos,
            self.output,
            self.instructions.len(),
        ] {
            num(&mut b, n);
        }
        if self.resident_sequences != 1 || self.prefill_tokens > 1 || self.shared_pages { num(&mut b, self.resident_sequences); }
        if self.prefill_tokens > 1 || self.shared_pages { num(&mut b, self.prefill_tokens); }
        if self.uncaptured_prefill { num(&mut b,usize::from(self.shared_pages)); }
        else if !self.prefill_variants.is_empty() {
            num(&mut b,usize::from(self.shared_pages));
            num(&mut b,self.prefill_variants.len());
            for &count in &self.prefill_variants {num(&mut b,count);}
        }
        if let Some(rows) = &self.output_rows {
            for value in [rows.source, rows.source_width, rows.first, rows.last] { num(&mut b, value); }
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
    let mut result = recipes_for_query(p, graph, p.prefill_tokens);
    for (id, recipe) in result.iter_mut().enumerate() {
        if let Some(rows) = &p.output_rows {
            if id >= rows.first && id <= rows.last {
                recipe["output_rows"] = serde_json::json!({
                    "domain":"request_sample_rows", "max_rows":p.resident_sequences,
                    "empty":"skip this proven output suffix; state prefix still executes",
                    "selection":if id==rows.first {Some(rows)} else {None},
                    "decode":"unchanged fixed sequence rows; compact selection only for uncaptured prefill",
                });
                if let Some(fields)=recipe["value_fields"].as_array_mut() {
                    for field in fields {
                        if let Some(shape)=field["prefill_shape"].as_array_mut() {
                            let axis=usize::from(p.instructions[id].op=="gather_heads"); if axis<shape.len() {shape[axis]=serde_json::json!("sample_rows");}
                        }
                    }
                }
            }
        }
        recipe["prefill_execution"] = serde_json::json!({
            "mode": if p.uncaptured_prefill {"uncaptured"} else {"exact_graph"},
            "max_query_tokens": p.prefill_tokens,
            "variable_query_tokens": p.uncaptured_prefill,
            "capture_allowed": !p.uncaptured_prefill,
            "invocation_lifetime": if p.uncaptured_prefill {"until_owning_stream_completion"} else {"prepared_or_graph_owned"},
        });
    }
    if !p.prefill_variants.is_empty() {
        for recipe in &mut result { recipe["prefill_variants"] = serde_json::json!([]); }
        for &count in &p.prefill_variants {
            for (recipe, variant) in result.iter_mut().zip(recipes_for_query(p, graph, count)) {
                recipe["prefill_variants"].as_array_mut().unwrap().push(serde_json::json!({
                    "packed_prefill":variant["packed_prefill"], "value_fields":variant["value_fields"]
                }));
            }
        }
    }
    result
}

// Storage remains sized for the maximum prepared shape, independently of the
// invocation's query count. This includes cache retention and control offsets.
fn recipes_for_query(p: &Program, graph: &Graph, query_tokens: usize) -> Vec<serde_json::Value> {
    let cache_controls: BTreeMap<_, _> = p.instructions.iter()
        .filter(|i| matches!(i.op.as_str(), "cache_c8_norm_rope_append" | "cache_norm_rope_append" | "cache_append" | "index_cache_append"))
        .enumerate().map(|(slot, i)| (i.name.as_str(), slot * 16 * p.prefill_tokens)).collect();
    p.instructions.iter().map(|i| {
        let n = &graph.nodes[i.logical_node];
        let (apis, internal, materialization): (&[&str], &str, &str) = match i.op.as_str() {
            "mhc_pre_project" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre"], "BF16 residual; FP32 mixing; Sinkhorn is separate", "persistent mHC outputs and residual alias"),
            "mhc_sinkhorn" => (&["aclnnManifoldConstrainedHyperConnectionSinkhornEnhance"], "20 FP32 normalization iterations", "persistent Sinkhorn output; preserves mHC handle"),
            "router_projection" => (&["aclnnMatmul"], "FP32 gate projection; top-k is separate", "persistent FP32 logits"),
            "router_select" => (&["aclnnMoeGatingTopK"], "sigmoid/bias group top-k; ids remain on device", "INT32 ids + FP32 scores and normalized logits"),
            "route_quantize" => (&["aclnnMoeInitRoutingV3"], "owner-local route filtering/sorting and INT8 quantization", "INT8 [8,width], FP32 [8] scales, INT64 [32] groups, INT32 [8] inverse indices"),
            "expert_act_quantize" => (&["aclnnDequantSwigluQuant"], "packed BF16 SwiGLU and INT8 quantization", "INT8 [8,width] + FP32 [8] scales; borrowed group metadata"),
            "expert_zero" if i.kernel == "aclnnInplaceZero" => (&["aclnnInplaceZero"], "device byte-zero of packed BF16 storage; vendor ZerosLike/ViewCopy implementation", "fresh BF16 routed rows; INT8 byte alias for zeroing; exclusive input to grouped matmul"),
            "expert_zero" => (&["aclrtMemsetAsync"], "define unselected expert rows as zero", "fresh BF16 [8,width]; exclusive input to grouped matmul"),
            "expert_gmm" => (&["aclnnGroupedMatmulV5"], "INT8 grouped matmul; quantization and zeroing are separate", "writes exclusively owned initialized output; packed weights prepared once"),
            "expert_values" => (&[], "view packed expert rows as a tensor", "alias; no copy"),
            "expert_pack" => (&["aclrtMemcpyAsync"], "interleave gate/up rows for the same routing decision", "new BF16 packed rows; borrowed routing metadata"),
            "attention_metadata_zero" => (&["aclrtMemsetAsync"], "initialize metadata output once per token", "persistent metadata buffer"),
            "attention_metadata" => (&["aclnnAiInfraFusedInferAttentionSinkMetadata"], "total query count and final device KV length; pre_tokens = declared window - 1; shared only within the same window", "one persistent 4096-byte device metadata buffer per window for the whole chunk"),
            "cast" => (&["aclnnCast"], "", "new device output; shared by identical consumers"),
            "quantize" => (&["aclnnDynamicQuant"], "BF16 → signed INT8; per-token symmetric FP32 scale", "explicit INT8 payload + FP32 scale"),
            "index_quantize" => (&["aclnnDynamicQuant"], "BF16 → signed INT8; independent 128-column heads/keys with FP32 scale", "INT8 payload + FP32 scale per 128 columns"),
            "quant_scales_f16" => (&["aclnnCast"], "Explicit FP32 → FP16 index scales; INT8 payload aliases input", "borrowed INT8 payload + newly allocated FP16 scales"),
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
            "gather_heads" => (&["HcclAllGather"], "BF16 whole token matrix per rank", "new rank-major output; explicit head_to_token consumer"),
            "repeat_heads" => (&["aclrtMemcpyAsync"], "one whole token matrix copy per declared repetition", "new repetition-major output; explicit head_to_token consumer"),
            "slice" => (&["aclnnCast"], "single-token alias; multi-token same-dtype strided column materialization", "one contiguous token matrix for prefill; no per-row assembly"),
            "mhc_input" => (&[], "shared allocation lease", "alias; no copy"),
            "repeat" | "concat" => (&["aclrtMemcpyAsync"], "device-to-device materialization", "new contiguous output"),
            "norm" | "optional_norm" if i.kernel != "alias" => (&["aclnnRmsNorm"], "BF16 operands; vendor accumulation contract", "new output + inverse RMS"),
            "optional_norm" => (&[], "checkpoint binding absent", "alias; no copy"),
            "mome" => (&["aclrtMemcpyAsync", "aclnnAiInfraFusedCausalConv1d"], "input staging protects source from vendor in-place mutation", "persistent mutable input/state"),
            "mhc_pre" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre", "aclnnManifoldConstrainedHyperConnectionSinkhornEnhance"], "BF16 residual; FP32 phi, mixing and Sinkhorn; residual borrowed", "persistent state; source alias"),
            "mhc_merge" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPre"], "BF16 residual; FP32 phi and mixing", "persistent state; source alias"),
            "mhc_post" => (&["aclnnAiInfraManifoldConstrainedHyperConnectionPost"], "BF16 residual and branch; FP32 mixing", "persistent output"),
            "query_project" => (&["aclnnBatchMatMul"], "strided local query/weight views; BF16 input/output", "new BF16 latent query"),
            "head_matmul" => (&["aclnnBatchMatMul"], "declared BF16 strided input/weight views; one 3D batch matmul over all query rows per head", "new contiguous BF16 [heads,tokens,columns]; layout conversion is a separate instruction"),
            "head_to_token" => (&["aclnnCast"], "same-dtype BF16 strided materialization, no numerical cast; one-token decode aliases input", "explicit [heads,tokens,width] to [tokens,heads,width] device layout conversion"),
            "query_rotary_extract" => (&["aclnnCast"], "strided BF16 rotary columns to contiguous FP32", "new FP32 rotary input"),
            "column_slice" => (&["aclnnCast"], "materialize declared strided BF16 columns without dtype conversion", "new contiguous BF16 output; source is a borrowed strided view"),
            "rope_bf16" => (&["aclnnRotaryPositionEmbedding"], "BF16 half rotation with BF16 position-derived coefficients", "new BF16 output"),
            "rope_f32" => (&["aclnnRotaryPositionEmbedding"], "FP32 rotation; rounding is a separate instruction", "new FP32 output"),
            "concat_columns" => (&["aclnnCat"], "concatenate columns of declared row matrices", "new contiguous output"),
            "rope" => (&["aclnnRotaryPositionEmbedding", "aclnnCast"], "FP32 rotation → BF16 rounding", "FP32 temporary + BF16 result"),
            "mla_query" => (&["aclnnBatchMatMul", "aclnnCast", "aclnnRotaryPositionEmbedding", "aclnnCast", "aclnnCat"], "strided BF16 → FP32 rotary extraction → BF16 rounding; packed with latent query", "immutable weights packed once; rotary and concatenation temporaries"),
            "mla_value" => (&["aclnnBatchMatMul"], "BF16 heads", "immutable weights packed once; new output"),
            "sink_normalize" => (&["aclnnRmsNorm", "aclrtMemcpyAsync"], "initialization only after first preparation", "persistent normalized sink"),
            "packed_attention_metadata" => (&["if_packed_attention_metadata_batch"], "all device positions -> causal INT32 indices and sequence lengths in one launch; scalar entry for decode", "shared persistent indices and lengths; no activation transfer"),
            "cache_c8_norm_rope_append" => (&["aclnnAiInfraKvRmsNormRopeCache"], "fused RMSNorm + BF16 rotary + pertile128 INT8 latent quantization; 4 FP32 scales", "packed PA_BSND rows: 512 INT8 bytes + 64 BF16 rotary + 4 FP32 scales = 656 bytes; no BF16 mirror"),
            "index_cache_append" if i.attrs.get("cache_dtype").is_some_and(|v|v=="int8") => (&["aclnnInplaceIndexCopy", "aclnnInplaceIndexCopy"], "INT8 key and FP16 scale append at same physical row; page size 64", "separate key/scale arrays share exclusive page ownership; no implicit quantization"),
            "index_cache_append" => (&["aclnnInplaceIndexCopy"], "BF16 index key append at device physical row; page size 64", "persistent paged BF16 keys; exclusive request ownership"),
            "sparse_index" if p.instructions[i.inputs[0]].kind==5 => (&["aclnnAiInfraQuantLightningIndexer"], "24-head INT8 queries/keys, FP16 scales/head weights; weighted-ReLU top-2048; causal device lengths", "INT32 logical indices; all operands already have the planned dtype"),
            "sparse_index" => (&["aclnnLightningIndexerEnhance"], "24-head BF16 weighted-ReLU top-2048; causal device lengths", "INT32 logical indices; shared consumers retain the SSA value"),
            "indexed_packed_attention" | "packed_latent_attention" => (&["aclnnAiInfraKvQuantSparseFlashAttention"], "direct packed C8 attention: quant modes 2, scale repo 1, tile 128, rope 64; BF16 learned sinks", "reads packed cache directly; no full-cache dequantization buffer"),
            "cache_normalize" => (&["aclnnAiInfraKvRmsNormRopeCache"], "BF16 RMS normalization and rotary encoding into temporary row", "temporary 16-row vendor output; only first row consumed"),
            "cache_quantize_append" => (&["if_cache_quantize"], "explicit symmetric per-token-head quantization; ties away from zero; signed INT4 low nibble first", "persistent packed payload + FP32 scales; BF16 learned sinks remain constants"),
            "cache_dequantize" => (&["if_cache_dequantize"], "explicit packed cache decode to BF16; future rows zeroed; all device resident", "leased BF16 attention scratch, reusable across layers; no persistent BF16 cache mirror"),
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
            "gather" | "gather_heads" => "AllGather, ranks 0–7",
            "route_quantize" | "expert_act_quantize" | "expert_gmm" | "expert_zero" | "expert_pack" | "expert_linear" | "expert_swiglu_down" | "expert_reduce" => "EP owner-local compute; explicit downstream AllReduce combines partials",
            _ => "none in this instruction",
        };
        let preparation = match i.op.as_str() {
            "linear" if i.kernel=="paired_bf16" => "Once before replay: two BF16 parameter copies pack the selected projection rows.",
            "expert_gmm" => "Once before replay: copy owner-local INT8 weights and lossless BF16 scales into grouped banks. No FP32 scale round-trip.",
            "sink_normalize" => "Once before replay: normalize checkpoint sink rows and pack BF16 latent/rotary constants; subsequent tokens borrow the prepared result.",
            "cache_c8_norm_rope_append" | "cache_quantize_append" | "cache_append" | "cache_norm_rope_append" => "Once before replay: copy prepared constant sinks into persistent cache state; initialize metadata from control inputs.",
            "mhc_pre_project" | "mhc_merge" => "Once before replay: fold scalar checkpoint parameters and upload prepared mHC coefficients.",
            "router_projection" => "Once before replay: transpose/convert checkpoint gate parameters and upload FP32 weights and bias.",
            "mome_inplace" => "Once before replay: transpose convolution weights and initialize recurrent state and control indices.",
            "packed_attention_metadata" => "Once before replay: upload query counts 1..max_query_tokens as control constants; per-query KV lengths and candidate indices are generated on device.",
            "attention_metadata" => "Once before replay: upload INT64 query counts 1..max_query_tokens; one metadata buffer per declared window serves the whole chunk.",
            _ => "No parameter/state copy recipe in this instruction.",
        };
        let mut fields:Vec<serde_json::Value>=match i.kind {
            0=>vec![serde_json::json!({"name":"value","dtype":i.dtype,"shape":[i.width]})],
            2=>vec![serde_json::json!({"name":"expert_ids","dtype":"int32","shape":[8]}),serde_json::json!({"name":"scores","dtype":"f32","shape":[8]})],
            3=>vec![serde_json::json!({"name":"packed_rows","dtype":"bf16","shape":[8,i.width]}),serde_json::json!({"name":"groups","dtype":"int64","shape":[32]}),serde_json::json!({"name":"inverse_indices","dtype":"int32","shape":[8]})],
            4 if i.op=="cache_c8_norm_rope_append"=>vec![serde_json::json!({"name":"packed_cache","dtype":"uint8","shape":[p.max_context.div_ceil(32),32,1,656]}),serde_json::json!({"name":"latent_view","dtype":"int8","shape":[p.max_context.div_ceil(32)*32,512],"byte_offset":0,"row_stride_bytes":656}),serde_json::json!({"name":"rotary_view","dtype":"bf16","shape":[p.max_context.div_ceil(32)*32,64],"byte_offset":512,"row_stride_bytes":656}),serde_json::json!({"name":"scale_view","dtype":"f32","shape":[p.max_context.div_ceil(32)*32,4],"byte_offset":640,"row_stride_bytes":656})],
            7 if i.op=="packed_attention_metadata"=>vec![serde_json::json!({"name":"causal_indices","dtype":"int32","shape":[1,1,512]}),serde_json::json!({"name":"kv_length","dtype":"int32","shape":[1]})],
            4 if i.op=="cache_quantize_append"=>vec![serde_json::json!({"name":"cache_payload","dtype":i.attrs["cache_dtype"],"shape":[p.max_context.div_ceil(16)*16,1,576]}),serde_json::json!({"name":"cache_scales","dtype":"f32","shape":[p.max_context.div_ceil(16)*16,1]})],
            5=>vec![serde_json::json!({"name":"quantized","dtype":"int8","shape":[1,i.width]}),serde_json::json!({"name":"scales","dtype":if i.op=="quant_scales_f16" {"f16"} else {"f32"},"shape":[if matches!(i.op.as_str(),"index_quantize"|"quant_scales_f16") {i.width/128} else {1}]})],
            6=>vec![serde_json::json!({"name":"quantized_rows","dtype":"int8","shape":[8,i.width]}),serde_json::json!({"name":"scales","dtype":"f32","shape":[8]}),serde_json::json!({"name":"groups","dtype":"int64","shape":[32]}),serde_json::json!({"name":"inverse_indices","dtype":"int32","shape":[8]})],
            7=>vec![serde_json::json!({"name":"metadata_effect","dtype":"int32","shape":[1024]})],
            8=>vec![serde_json::json!({"name":"selected_indices","dtype":"int32","shape":[1,1,i.width],"addressing":"logical_token","sharing":"SSA consumers retain the device value"})],
            9=>vec![serde_json::json!({"name":"index_cache","dtype":if i.attrs.get("cache_dtype").is_some_and(|v|v=="int8") {"int8"} else {"bf16"},"shape":[p.resident_sequences*p.max_context.div_ceil(64),64,1,i.width]}),
                serde_json::json!({"name":"device_block_table","dtype":"int32","shape":[1,p.max_context.div_ceil(64)],"role":"control_input","ownership":if p.shared_pages {"shared exclusive page reservations"} else {"fixed resident partitions"}}),
                serde_json::json!({"name":"physical_append_position","dtype":"int32","shape":[1],"page_tokens":64,"role":"control_input","control_bank_bytes":cache_controls.len()*16*p.prefill_tokens,"byte_offset":cache_controls[i.name.as_str()]+8})],
            _=>vec![],
        };
        if i.kind==9 && i.attrs.get("cache_dtype").is_some_and(|v|v=="int8") {
            fields.push(serde_json::json!({"name":"index_cache_scales","dtype":"f16","shape":[p.resident_sequences*p.max_context.div_ceil(64),64,1],"ownership":"same page IDs as index_cache"}));
        }
        if i.kind==9 {
            fields.push(serde_json::json!({"name":"index_query_lengths","dtype":"int32","shape":[p.prefill_tokens],"role":"initialization_constant","values":"1..max_query_tokens; select count-1 for TND causal alignment"}));
        }
        if i.op=="packed_attention_metadata" {
            fields.push(serde_json::json!({"name":"packed_query_lengths","dtype":"int32","shape":[p.prefill_tokens],"role":"initialization_constant","values":"1..max_query_tokens; select count-1 for TND causal alignment"}));
        }
        if i.op=="attention_metadata" {
            fields.push(serde_json::json!({"name":"window_query_lengths","dtype":"int64","shape":[p.prefill_tokens],"role":"shared_initialization_constant","values":"1..max_query_tokens; select count-1; KV length is the final chunk row"}));
        }
        if matches!(i.op.as_str(), "cache_c8_norm_rope_append" | "cache_norm_rope_append" | "cache_append") {
            for field in &mut fields {
                if let Some(shape) = field.get_mut("shape").and_then(serde_json::Value::as_array_mut) {
                    if let Some(first) = shape.first_mut() {
                        if let Some(n) = first.as_u64() { *first = serde_json::json!(n * p.resident_sequences as u64); }
                    }
                }
            }
            if i.op != "cache_c8_norm_rope_append" {
                let window=graph.nodes.iter().filter(|node|node.op=="latent_attention" && node.inputs.get(1)==Some(&i.logical_node))
                    .filter_map(|node|node.integer("window").ok()).max().unwrap_or(p.max_context);
                let capacity=inferfabric_ir::serving::sliding_capacity(p.max_context as u32,window as u32,p.prefill_tokens as u32,16).expect("validated cache dimensions") as usize;
                fields.push(serde_json::json!({"name":"bf16_cache","dtype":"bf16","shape":[p.resident_sequences*capacity/16,16,1,576],"retention":"sliding_window","window_tokens":window,"retained_capacity_tokens":capacity,"max_query_tokens":p.prefill_tokens,"addressing":"absolute logical page modulo retained page count, then owned physical page ID"}));
            }
            let retained=if i.op=="cache_c8_norm_rope_append" {serde_json::json!(p.max_context.div_ceil(32)*32)} else {fields.iter().find(|f|f["name"]=="bf16_cache").unwrap()["retained_capacity_tokens"].clone()};
            fields.push(serde_json::json!({"name":"resident_partition","resident_sequences":p.resident_sequences,"logical_capacity_tokens":p.max_context,"owned_capacity_tokens":retained,"ownership":if p.shared_pages {"shared physical page pool; partial request reservations; exclusive ownership checked at bind; release after rank completion"} else {"disjoint fixed page partitions; mappings persist across selection; cross-partition binding rejected"}}));
            fields.push(serde_json::json!({"name":"device_block_table","dtype":"int32","shape":[1,if i.op=="cache_c8_norm_rope_append" {p.max_context.div_ceil(32)} else {p.max_context.div_ceil(16)}],"role":"control_input","binding":"absolute logical pages map to owned physical page IDs; BF16 ring repeats owned IDs; stable device address"}));
            fields.push(serde_json::json!({"name":"physical_append_position","dtype":if i.op=="cache_append" {"int32"} else {"int64"},"shape":[1],"role":"control_input","binding":"block_table[logical_position / page_tokens] * page_tokens + logical_position % page_tokens","page_tokens":if i.op=="cache_c8_norm_rope_append" {32} else {16},"staging":"one packed control upload per rank per invocation","control_bank_bytes":cache_controls.len()*16*p.prefill_tokens,"byte_offset":cache_controls[i.name.as_str()]+if i.op=="cache_append" {8} else {0}}));
        }
        if matches!(i.op.as_str(), "mome" | "mome_inplace") {
            fields.push(serde_json::json!({"name":"recurrent_history","dtype":"bf16","shape":[p.resident_sequences,2,i.width],"binding":"shared device resident-slot control selects one history; other slots remain untouched"}));
            fields.push(serde_json::json!({"name":"mome_y_scratch","dtype":"bf16","shape":[if p.uncaptured_prefill && i.op=="mome_inplace" {query_tokens} else {1},i.width],"role":"device_scratch","aliases_input":false,"binding":"distinct yOut; vendor writes this buffer before its cross-core barrier, then copies it back to x for inplace execution; lease remains ordered on the owning stream"}));
            if p.uncaptured_prefill && i.op=="mome_inplace" {
                fields.push(serde_json::json!({"name":"mome_query_starts","dtype":"int32","shape":[2],"role":"shared_control_input","values":"[0, actual_query_tokens]","binding":"one packed sequence, num_computed_tokens is its absolute start position"}));
            }
        }
        if query_tokens>1 {
            for field in &mut fields {
                let name=field["name"].as_str().unwrap_or("");
                let shape=field.get("shape").and_then(serde_json::Value::as_array).cloned();
                if let Some(mut shape)=shape {
                    if i.kind==0 && name=="value" && i.op!="sink_normalize" {
                        shape.insert(0,serde_json::json!(query_tokens));
                    } else if matches!(name,"selected_indices"|"expert_ids"|"scores"|"packed_rows"|"inverse_indices"|"quantized"|"scales"|"quantized_rows"|"causal_indices"|"kv_length"|"physical_append_position")
                        && let Some(n)=shape.first().and_then(serde_json::Value::as_u64) {
                        shape[0]=serde_json::json!(n*query_tokens as u64);
                    }
                    field["prefill_shape"]=serde_json::json!(shape);
                }
            }
        }
        if matches!(i.op.as_str(),"head_matmul"|"head_to_token"|"gather_heads"|"repeat_heads") {
            let heads=i.attrs["heads"].parse::<usize>().unwrap();
            let width=if matches!(i.op.as_str(),"gather_heads"|"repeat_heads") {i.width/heads}
                else {i.attrs[if i.op=="head_matmul" {"n"} else {"width"}].parse::<usize>().unwrap()};
            fields[0]=serde_json::json!({"name":"value","dtype":"bf16","shape":[heads,1,width],
                "prefill_shape":if i.op!="head_to_token" {vec![heads,query_tokens,width]} else {vec![query_tokens,heads,width]},
                "layout":if i.op!="head_to_token" {"head,token,column"} else {"token,head,column"}});
            if i.op=="head_matmul" {
                let k=i.attrs["k"].parse::<usize>().unwrap();
                fields.push(serde_json::json!({"name":"input_view","dtype":"bf16","shape":[heads,query_tokens,k],
                    "role":"borrowed_strided_view","strides_elements":[i.attrs["input_head_stride"].parse::<usize>().unwrap(),p.instructions[i.inputs[0]].width,1],
                    "rank_offset_elements":i.attrs["input_rank_stride"],"materialization":"vendor 3D BatchMatMul strided-input contract"}));
                fields.push(serde_json::json!({"name":"weight_view","dtype":"bf16","shape":[heads,k,width],
                    "role":"borrowed_parameter_view","binding":i.attrs["weight"],"offset_elements":i.attrs["weight_offset"],
                    "strides_elements":[i.attrs["weight_head_stride"].parse::<usize>().unwrap(),i.attrs["weight_k_stride"].parse::<usize>().unwrap(),i.attrs["weight_n_stride"].parse::<usize>().unwrap()]}));
            }
        }
        let batched_index = i.op=="sparse_index" && p.instructions[i.inputs[0]].kind==5;
        let batched_attention = matches!(i.op.as_str(),"indexed_packed_attention"|"packed_latent_attention"|"latent_attention");
        let batched_window = matches!(i.op.as_str(),"latent_attention"|"attention_metadata");
        let batched_rotary = i.op=="rope_bf16";
        let batched_mome = p.uncaptured_prefill && i.op=="mome_inplace";
        let batched_cache = matches!(i.op.as_str(),"cache_append"|"cache_norm_rope_append"|"cache_c8_norm_rope_append"|"index_cache_append");
        let batched = batched_index || batched_attention || batched_window || batched_rotary || batched_mome || batched_cache || i.op=="packed_attention_metadata";
        let per_query = !batched && matches!(i.op.as_str(), "mome"|"mome_inplace"|"query_project"|"query_rotary_extract"|"rope"|"rope_bf16"|"rope_f32"|"mla_query"|"mla_value"|"index_cache_append"|"cache_append"|"cache_norm_rope_append"|"cache_c8_norm_rope_append"|"latent_attention"|"indexed_packed_attention"|"sparse_index"|"packed_latent_attention"|"attention_metadata"|"packed_attention_metadata");
        let assembly = !batched && matches!(i.op.as_str(), "repeat"|"query_project"|"query_rotary_extract"|"rope"|"rope_bf16"|"rope_f32"|"mla_query"|"mla_value"|"mome"|"latent_attention"|"indexed_packed_attention"|"sparse_index"|"packed_latent_attention");
        let packed = if query_tokens > 1 { serde_json::json!({
            "tokens":query_tokens,
            "schedule":if i.op=="sink_normalize" {"shared initialization constant"} else if per_query {"ordered per-query calls"} else {"token matrix"},
            "api_repetitions":if i.op=="repeat_heads" {i.attrs["heads"].parse::<usize>().unwrap()} else if per_query {query_tokens} else {1},
            "causal_alignment":if batched_mome {Some("one causal sequence per resident; query starts [0,T], computed tokens = absolute chunk start; update two-row recurrent history once after the sequence")} else if batched_window {Some("one TND sequence; total query count and final chunk KV length; per-row causal window from declared pre_tokens; one shared metadata buffer per window")} else if batched_index || batched_attention {Some("one TND sequence; total query count and final chunk KV length; all chunk keys appended before selection/attention; each attention row consumes its own causal indices")} else {None},
            "coefficient_layout":if batched_rotary {Some("BF16 [tokens,1,1,64] cosine/sine; one absolute position per token; broadcast across heads")} else {None},
            "cache_append_layout":if batched_cache {Some("one batch of independent token rows; device physical positions are a strided vector in the packed control bank (16 bytes per row); BF16/C8 norm-rope input [tokens,1,1,576], cosine/sine [tokens,1,1,64]; index copy source [tokens,width]")} else {None},
            "output_elements":if i.kind==0 {Some(i.width*if i.op=="sink_normalize" {1} else {query_tokens})} else {None},
            "assembly_copies":if assembly {query_tokens} else if i.op=="gather" {8*query_tokens} else {0},
            "assembly_api":"aclrtMemcpyAsync device-to-device",
            "assembly_layout":if i.op=="gather" {"rank-major to token-major"} else {"contiguous token rows"},
            "control_row_stride_bytes":16,
            "control_bank_bytes":cache_controls.len()*16*p.prefill_tokens
        }) } else {serde_json::Value::Null};
        serde_json::json!({"packed_prefill":packed,"preparation":preparation,"value_fields":fields,"apis":apis,"internal":internal,"materialization":materialization,"placement":placement,"communication":communication,
            "granularity":if i.op=="sink_normalize" {"initialization recipe"} else {"instruction"}})
    }).collect()
}
