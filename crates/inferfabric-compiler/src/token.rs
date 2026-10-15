//! Lower token-kernel parameter requirements to the permanent initialization IR.
//! No layer traversal, model-name dispatch, or checkpoint-name placement rules.
use crate::openpangu::{Inventory, TensorDescriptor};
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_ir::initialization::{DType, InitializationPlan, Instruction, Parameter};
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq)]
struct Request {
    shape: Vec<u64>,
    dtype: DType,
    shard_rows: bool,
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
        };
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
    ) -> Result<()> {
        let dtype = match self.tensor(&format!("{name}.weight"))?.dtype.as_str() {
            "I8" => DType::I8,
            "BF16" if !quant_only => DType::Bf16,
            _ => return Err(invalid(format!("unsupported projection dtype {name}"))),
        };
        self.add(
            format!("{name}.weight"),
            vec![output, input],
            dtype,
            sharded,
        )?;
        if dtype == DType::I8 {
            self.add(
                format!("{name}.weight_scale"),
                vec![output, 1],
                DType::F32,
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
                Type::Tensor(w) | Type::Experts(w) => w as u64,
                _ => 0,
            });
            match n.op.as_str() {
                "embedding" => b.add(
                    format!("{name}.weight"),
                    vec![n.integer("vocab")? as u64, n.integer("width")? as u64],
                    DType::Bf16,
                    false,
                )?,
                "linear" => b.linear(
                    name,
                    width,
                    n.integer("width")? as u64,
                    n.attrs["placement"] == "tp_rows",
                    false,
                )?,
                "norm" | "optional_norm" => {
                    let key = format!("{name}.weight");
                    if n.op == "norm" || inventory.tensors.contains_key(&key) {
                        b.add(key, vec![width], DType::Bf16, false)?;
                    }
                }
                "mome" => b.add(
                    format!("{name}.weight"),
                    vec![width, 1, 3],
                    DType::Bf16,
                    false,
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
                            width,
                            n.integer("width")? as u64,
                            false,
                            true,
                        )?;
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
                } else {
                    0
                })
                .ok_or_else(|| invalid("slice offset overflow"))?;
            let expected_bytes = source_bytes * if r.shard_rows { topology.tp as u64 } else { 1 };
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
            });
        }
        plan.validate()?;
        plans.push(plan);
    }
    Ok(plans)
}
