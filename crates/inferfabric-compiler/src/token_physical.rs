//! Ascend token backend adapter. Model topology comes from the graph, while
//! implementation boundaries come from the existing backend's operator contracts.
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_model::{Result, invalid};
use inferfabric_plan::physical::*;
use sha2::{Digest, Sha256};

pub(crate) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn lower(graph: &Graph, devices: &[u32]) -> Result<Plan> {
    let types = graph.validate()?;
    if devices.len() != 8 {
        return Err(invalid("ascend_token_v1 requires eight rank bindings"));
    }
    let logical_digest = digest(&serde_json::to_vec(graph).map_err(|e| invalid(e.to_string()))?);
    let mut source = Vec::new();
    source.extend_from_slice(include_bytes!("../../../native/src/openpangu_decode.cpp"));
    source.extend_from_slice(include_bytes!("../../../native/src/token_dispatch.inc"));
    source.extend_from_slice(include_bytes!(
        "../../../native/include/ascend_token_contract.h"
    ));
    source.extend_from_slice(include_bytes!("../../../native/CMakeLists.txt"));
    source.extend_from_slice(include_bytes!("../../../native/src/openpangu_ops.h"));
    source.extend_from_slice(include_bytes!("token_physical.rs"));
    source.extend_from_slice(include_bytes!("../../../native/src/resident_tensor.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/resident_transfer.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/execution_profile.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/execution_capture.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/prepared_execution.h"));
    let revision = digest(&source);
    let mut values: Vec<Value> = vec![];
    let mut operations = vec![];
    let mut collective_sequence = 0;
    for (i, n) in graph.nodes.iter().enumerate() {
        let (dtype, shape, bytes, layout, location) = match types[i] {
            Type::Tensor(w) => {
                // Storage precision follows backend boundaries, not the logical
                // tensor type. Slices/repeats preserve their producer's storage.
                let dtype = match n.op.as_str() {
                    "expert_reduce" | "allreduce" => Some("f32".to_string()),
                    "slice" | "repeat" | "concat" => values[n.inputs[0]].dtype.clone(),
                    "optional_norm" if values[n.inputs[0]].dtype.as_deref() != Some("bf16") => None,
                    _ => Some("bf16".to_string()),
                };
                let bytes = dtype
                    .as_deref()
                    .map(|t| w as u64 * if t == "bf16" { 2 } else { 4 });
                let layout = if n.op == "slice" {
                    "device_contiguous_shared_lease_view"
                } else {
                    "device_contiguous_shared_lease_payload"
                };
                (dtype, vec![w as u64], bytes, layout, Location::Device)
            }
            Type::Experts(_) => (
                None,
                vec![],
                None,
                "opaque_owner_local_expert_rows",
                Location::Device,
            ),
            Type::HyperConnection => (
                None,
                vec![],
                None,
                "device_state_and_values",
                Location::Device,
            ),
            Type::Routes => (
                None,
                vec![],
                None,
                "opaque_routing_ids_and_scores",
                Location::Device,
            ),
            Type::Cache => (
                None,
                vec![],
                None,
                "opaque_device_cache_reference",
                Location::Device,
            ),
        };
        values.push(Value {
            logical_node: i,
            dtype,
            shape,
            layout: layout.into(),
            location,
            payload_bytes: bytes,
            alignment: None,
            arena_offset: None,
            lifetime: [i, i],
        });
        let mut boundaries = vec![Boundary {
            kind: BoundaryKind::InternalLaunches,
            detail: "BF16 results retain BF16 storage; FP32 arithmetic/reduction boundaries remain explicit. Slices are shared-allocation device views, and operator outputs transfer ownership without copies. Device-only single-stream execution: leases recycle after last enqueue, descriptors persist; qualified Cast/Add/Sub/Mul/Matmul invocations are repeatable, other vendor invocations remain explicitly one-shot. No activation/routing/cache readback or implicit fallback".into(),
        }, Boundary {
            kind: BoundaryKind::DynamicAllocation,
            detail: "Per-rank reusable device pool grows on a new size/liveness requirement during preparation; warmed decode reuses buffers and workspace. Runtime audit records allocations and reuses. Static arena offsets and full workspace bounds remain unresolved".into(),
        }];
        if matches!(
            n.op.as_str(),
            "rope" | "mla_query" | "mome" | "cache_append"
        ) {
            boundaries.push(Boundary {kind: BoundaryKind::ControlUpload,
                detail: "Position/length/index metadata or position-derived rotary constants only; never activation contents".into()});
        }
        if matches!(n.op.as_str(), "mome" | "mhc_pre" | "mhc_merge" | "router") {
            boundaries.push(Boundary {kind: BoundaryKind::ParameterReadback,
                detail: "One-time immutable parameter transformation from resident checkpoint weights; no graph value readback".into()});
            boundaries.push(Boundary {
                kind: BoundaryKind::ParameterUpload,
                detail: "Upload transformed immutable parameters once per state binding".into(),
            });
        }
        if matches!(n.op.as_str(), "expert_linear" | "expert_swiglu_down") {
            boundaries.push(Boundary {kind: BoundaryKind::PersistentState,
                detail: "Device-packed owner-local expert weight banks; eight device-indexed slots per rank, including masked foreign routes. Additional bank memory and compute are not yet optimized".into()});
        }
        if i == graph.output {
            boundaries.push(Boundary {kind: BoundaryKind::Synchronize,
                detail: "Request output only: enqueue pinned D2H of permitted IDs/diagnostics, then complete this rank's stream once. No per-operator or device-global fence; teardown also completes the owning stream".into()});
            boundaries.push(Boundary {kind: BoundaryKind::RequestOutput,
                detail: "After decode only: device argmax/top-k/log-softmax; 64 bytes of consensus token IDs per rank, plus 80 bytes of top-five output diagnostics on leader. Prefill and intermediate values are never read back".into()});
        }
        let collective = if matches!(n.op.as_str(), "gather" | "allreduce") {
            let Type::Tensor(width) = types[n.inputs[0]] else {
                return Err(invalid("collective input type"));
            };
            let c = Collective {
                group: "all_ranks".into(),
                operation: if n.op == "gather" {
                    "all_gather"
                } else {
                    "all_reduce_sum"
                }
                .into(),
                dtype: if n.op == "gather" { "bf16" } else { "f32" }.into(),
                elements_per_rank: width as u64,
                sequence: collective_sequence,
            };
            collective_sequence += 1;
            Some(c)
        } else {
            None
        };
        operations.push(Operation {
            name: n.name.clone(), logical_nodes: vec![i], inputs: n.inputs.clone(), collective,
            candidates: vec![Candidate {
                id: format!("ascend.token.device_resident.{}.v1", n.op),
                contract_revision: revision.clone(), numerical_contract: format!("ascend_token_v1/{}", n.op),
                location: Location::Device, eligible: true, rejection: None,
                workspace_bound_bytes: None, capture_qualified: false, boundaries,
                unresolved: vec!["Runtime-prepared descriptors/pool are not a frozen physical arena; internal device DAG and static full memory bound remain unresolved".into(),
                    "Native binary, vendor runtime and custom kernel identities require deployment binding".into()],
                cost: Cost::default(),
            }],
        });
    }
    let mut plan = serial_plan(
        graph.name.clone(),
        "ascend_token_v1".into(),
        logical_digest,
        devices,
        operations,
        values,
        vec![graph.output],
    )?;
    plan.limitations.push("Request boundary: NPU top-k/log-softmax; export 64-byte rank-consensus token IDs per rank, plus 80-byte top-five output diagnostics on leader only. No vocabulary logits or intermediate activations exported".into());
    Ok(plan)
}
