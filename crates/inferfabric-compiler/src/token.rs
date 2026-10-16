//! Lower token-kernel parameter requirements to the permanent initialization IR.
//! No layer traversal, model-name dispatch, or checkpoint-name placement rules.
use crate::openpangu::{Inventory, TensorDescriptor};
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_ir::initialization::{
    DType, InitializationPlan, Instruction, Parameter, SourceRows,
};
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq)]
struct Request {
    shape: Vec<u64>,
    dtype: DType,
    shard_rows: bool,
    shard_columns: bool,
}
struct Bindings<'a> {
    inventory: &'a Inventory,
    requests: BTreeMap<String, Request>,
}
impl Bindings<'_> {
    fn tensor(&self, name: &str) -> Result<&TensorDescriptor> {
        self.inventory
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("missing graph parameter {name}")))
    }
    fn add(&mut self, name: String, shape: Vec<u64>, dtype: DType, shard_rows: bool) -> Result<()> {
        let r = Request {
            shape,
            dtype,
            shard_rows,
            shard_columns: false,
        };
        self.request(name, r)
    }
    fn request(&mut self, name: String, r: Request) -> Result<()> {
        if let Some(old) = self.requests.insert(name.clone(), r.clone())
            && old != r
        {
            return Err(invalid(format!("conflicting graph parameter views {name}")));
        }
        Ok(())
    }
    fn linear(
        &mut self,
        name: &str,
        input: u64,
        output: u64,
        sharded: bool,
        quant_only: bool,
        columns: bool,
    ) -> Result<()> {
        let dtype = match self.tensor(&format!("{name}.weight"))?.dtype.as_str() {
            "I8" => DType::I8,
            "BF16" if !quant_only => DType::Bf16,
            _ => return Err(invalid(format!("unsupported projection dtype {name}"))),
        };
        if columns && sharded {
            return Err(invalid("projection cannot shard both axes"));
        }
        self.request(
            format!("{name}.weight"),
            Request {
                shape: vec![output, input],
                dtype,
                shard_rows: sharded,
                shard_columns: columns,
            },
        )?;
        if dtype == DType::I8 {
            if quant_only && self.tensor(&format!("{name}.weight_scale"))?.dtype != "BF16" {
                return Err(invalid(
                    "grouped INT8 matmul requires lossless BF16 checkpoint scales",
                ));
            }
            self.add(
                format!("{name}.weight_scale"),
                vec![output, 1],
                if quant_only { DType::Bf16 } else { DType::F32 },
                sharded,
            )?;
        }
        Ok(())
    }
}
pub fn initialization_plans(
    graph: &Graph,
    topology: &OpenPanguParallel,
    inventory: &Inventory,
) -> Result<Vec<InitializationPlan>> {
    let types = graph.validate()?;
    topology.validate()?;
    if topology.tp != 8
        || topology.max_model_len != graph.max_context
        || topology.max_num_seqs != 1
        || topology.max_num_batched_tokens != 1
    {
        return Err(invalid("unsupported token executor topology/capacity"));
    }
    let mut plans = vec![];
    for (rank, &device) in topology.devices.iter().enumerate() {
        let mut b = Bindings {
            inventory,
            requests: BTreeMap::new(),
        };
        for n in &graph.nodes {
            let Some(name) = n.attrs.get("weight") else {
                continue;
            };
            let width = n.inputs.first().map_or(0, |&j| match types[j] {
                Type::Tensor(w) | Type::Experts(w) | Type::Quantized(w) => w as u64,
                _ => 0,
            });
            match n.op.as_str() {
                "embedding" => b.add(
                    format!("{name}.weight"),
                    vec![n.integer("vocab")? as u64, n.integer("width")? as u64],
                    DType::Bf16,
                    false,
                )?,
                "linear" if n.attrs.contains_key("paired_weight") => {
                    b.add(
                        format!("{name}.weight"),
                        vec![n.integer("width")? as u64, width],
                        DType::Bf16,
                        false,
                    )?;
                    b.add(
                        format!("{}.weight", n.attrs["paired_weight"]),
                        vec![n.integer("paired_width")? as u64, width],
                        DType::Bf16,
                        false,
                    )?;
                }
                "linear" => b.linear(
                    name,
                    width,
                    n.integer("width")? as u64,
                    n.attrs["placement"] == "tp_rows",
                    false,
                    n.attrs["placement"] == "tp_columns",
                )?,
                "norm" | "optional_norm" | "cache_norm_rope_append" => {
                    let key = format!("{name}.weight");
                    if n.op != "optional_norm" || inventory.tensors.contains_key(&key) {
                        b.add(key, vec![width], DType::Bf16, false)?;
                    }
                }
                "mome" => b.add(
                    format!("{name}.weight"),
                    vec![width, 1, 3],
                    DType::Bf16,
                    n.attrs.get("placement").is_some_and(|p| p == "tp_channels"),
                )?,
                "mhc_pre" | "mhc_merge" => {
                    let merge = n.op == "mhc_merge";
                    let rows = if merge { 4 } else { 24 };
                    let suffix = if merge { "_pre" } else { "" };
                    b.add(
                        format!("{name}.phi.weight"),
                        vec![rows, 10240],
                        DType::F32,
                        false,
                    )?;
                    b.add(format!("{name}.norm_gamma"), vec![10240], DType::F32, false)?;
                    b.add(
                        format!("{name}.branch_alpha{suffix}"),
                        vec![if merge { 1 } else { 3 }],
                        DType::F32,
                        false,
                    )?;
                    b.add(
                        format!("{name}.branch_beta{suffix}"),
                        vec![rows],
                        DType::F32,
                        false,
                    )?;
                }
                "mla_query" | "mla_value" => {
                    b.add(format!("{name}.weight"), vec![1536, 512], DType::Bf16, true)?
                }
                "sink_normalize" => {
                    b.add(
                        format!("{name}.param_sink_compressed_kv"),
                        vec![128, 512],
                        DType::Bf16,
                        false,
                    )?;
                    b.add(
                        format!("{name}.param_sink_k_pe"),
                        vec![128, 64],
                        DType::Bf16,
                        false,
                    )?;
                    b.add(
                        format!("{}.weight", n.attrs["norm"]),
                        vec![512],
                        DType::Bf16,
                        false,
                    )?;
                }
                "router" => {
                    b.add(
                        format!("{name}.gate.weight"),
                        vec![256, 2560],
                        DType::Bf16,
                        false,
                    )?;
                    b.add(
                        format!("{name}.e_score_correction_bias"),
                        vec![256],
                        DType::F32,
                        false,
                    )?;
                }
                "expert_linear" | "expert_swiglu_down" => {
                    for e in rank * 32..(rank + 1) * 32 {
                        b.linear(
                            &name.replace("{expert}", &e.to_string()),
                            if n.op == "expert_swiglu_down" && n.inputs.len() == 2 {
                                width / 2
                            } else {
                                width
                            },
                            n.integer("width")? as u64,
                            false,
                            true,
                            false,
                        )?;
                        if let Some(other) = n.attrs.get("paired_weight") {
                            b.linear(
                                &other.replace("{expert}", &e.to_string()),
                                width,
                                n.integer("width")? as u64,
                                false,
                                true,
                                false,
                            )?;
                        }
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "missing parameter requirement for {}",
                        n.op
                    )));
                }
            }
        }
        let mut plan = InitializationPlan {
            rank: rank as u32,
            device,
            checkpoint_metadata_sha256: inventory.metadata_sha256.clone(),
            parameters: vec![],
            instructions: vec![],
        };
        for (name, r) in &b.requests {
            let t = b.tensor(name)?;
            let source_dtype = match t.dtype.as_str() {
                "I8" => DType::I8,
                "BF16" => DType::Bf16,
                "F32" => DType::F32,
                _ => return Err(invalid("unsupported parameter source dtype")),
            };
            if source_dtype != r.dtype && !(source_dtype == DType::Bf16 && r.dtype == DType::F32) {
                return Err(invalid(format!("parameter cast mismatch {name}")));
            }
            let mut expected = r.shape.clone();
            if r.shard_rows {
                expected[0] *= topology.tp as u64;
            }
            if r.shard_columns {
                expected[1] *= topology.tp as u64;
            }
            if t.shape != expected {
                return Err(invalid(format!(
                    "graph binding {name}: expected {expected:?}, got {:?}",
                    t.shape
                )));
            }
            let source_bytes = r.shape.iter().try_fold(source_dtype.bytes(), |a, &d| {
                a.checked_mul(d)
                    .ok_or_else(|| invalid("parameter size overflow"))
            })?;
            let source_offset = t
                .offset
                .checked_add(if r.shard_rows {
                    source_bytes
                        .checked_mul(rank as u64)
                        .ok_or_else(|| invalid("slice overflow"))?
                } else if r.shard_columns {
                    r.shape[1]
                        .checked_mul(source_dtype.bytes())
                        .and_then(|n| n.checked_mul(rank as u64))
                        .ok_or_else(|| invalid("column slice overflow"))?
                } else {
                    0
                })
                .ok_or_else(|| invalid("slice offset overflow"))?;
            let expected_bytes = source_bytes
                * if r.shard_rows || r.shard_columns {
                    topology.tp as u64
                } else {
                    1
                };
            if expected_bytes != t.bytes {
                return Err(invalid("checkpoint byte extent mismatch"));
            }
            let id = plan.parameters.len() as u32;
            plan.parameters.push(Parameter {
                id,
                name: name.clone(),
                dtype: r.dtype,
                shape: r.shape.clone(),
            });
            plan.instructions.push(Instruction::LoadSlice {
                destination: id,
                shard: t.shard.clone(),
                offset: source_offset,
                bytes: source_bytes,
                source_dtype,
                source_rows: if r.shard_columns {
                    Some(SourceRows {
                        row_bytes: r.shape[1] * source_dtype.bytes(),
                        stride_bytes: expected[1] * source_dtype.bytes(),
                    })
                } else {
                    None
                },
            });
        }
        plan.validate()?;
        plans.push(plan);
    }
    Ok(plans)
}
/// Publish a new bundle directory. Refuse to overwrite an existing bundle so
/// readers cannot observe a mixture of old execution and new initialization IR.
pub fn write_bundle(
    graph: &Graph,
    topology: &OpenPanguParallel,
    inventory: &Inventory,
    output: &std::path::Path,
) -> Result<()> {
    let plans = initialization_plans(graph, topology, inventory)?;
    let program = crate::token_program::lower(graph, Some(inventory))?;
    let binary = program.encode()?;
    let physical = crate::token_physical::lower(graph, &topology.devices)?;
    if output.exists() {
        return Err(invalid(
            "token bundle output already exists; choose a new directory",
        ));
    }
    let parent = output.parent().unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    // Reserve a unique sibling, then rename only after all writes succeed.
    let mut counter = 0u32;
    let staging = loop {
        let candidate = parent.join(format!(
            ".inferfabric-token-{}-{counter}",
            std::process::id()
        ));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                counter = counter
                    .checked_add(1)
                    .ok_or_else(|| invalid("staging counter overflow"))?;
            }
            Err(e) => return Err(e.into()),
        }
    };
    let publish = (|| {
        for initialization in &plans {
            crate::write_atomic(
                &staging.join(format!("rank-{}.bin", initialization.rank)),
                &crate::initialization::encode(initialization)?,
            )?;
            crate::write_atomic(
                &staging.join(format!("rank-{}.init.json", initialization.rank)),
                &serde_json::to_vec_pretty(initialization).map_err(|e| invalid(e.to_string()))?,
            )?;
        }
        crate::write_atomic(&staging.join("execution.bin"), &binary)?;
        crate::write_atomic(
            &staging.join("device-program.json"),
            &serde_json::to_vec_pretty(&program).map_err(|e| invalid(e.to_string()))?,
        )?;
        crate::write_atomic(
            &staging.join("execution.json"),
            &serde_json::to_vec_pretty(graph).map_err(|e| invalid(e.to_string()))?,
        )?;
        crate::write_atomic(&staging.join("execution.dot"), graph.dot()?.as_bytes())?;
        crate::write_atomic(
            &staging.join("execution.html"),
            crate::token_view::render_with_program(graph, &physical, &program)?.as_bytes(),
        )?;
        crate::write_atomic(
            &staging.join("physical.json"),
            &serde_json::to_vec_pretty(&physical).map_err(|e| invalid(e.to_string()))?,
        )?;
        crate::write_atomic(&staging.join("plan.json"), &serde_json::to_vec_pretty(&serde_json::json!({"format":"inferfabric.token-bundle.v1","executable":true,"topology":topology,"nodes":graph.nodes.len(),"execution_format":program.format,"instructions":program.instructions.len(),"execution_sha256":crate::token_physical::digest(&binary),"checkpoint_metadata_sha256":inventory.metadata_sha256})).map_err(|e|invalid(e.to_string()))?)?;
        std::fs::rename(&staging, output)?;
        Ok(())
    })();
    if publish.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    publish
}
