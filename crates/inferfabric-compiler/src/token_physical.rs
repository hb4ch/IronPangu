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
    source.extend_from_slice(include_bytes!("../../../native/src/openpangu_mhc.h"));
    source.extend_from_slice(include_bytes!(
        "../../../native/src/openpangu_decoder_state.h"
    ));
    source.extend_from_slice(include_bytes!("token_physical.rs"));
    source.extend_from_slice(include_bytes!("token_program.rs"));
    source.extend_from_slice(include_bytes!("../../../native/src/resident_tensor.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/resident_transfer.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/execution_profile.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/execution_capture.h"));
    source.extend_from_slice(include_bytes!("../../../native/src/prepared_execution.h"));
    source.extend_from_slice(include_bytes!(
        "../../../native/src/prepared_weight_views.h"
    ));
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
                    "expert_reduce" => Some("f32".to_string()),
                    "allreduce" => Some(n.attrs.get("dtype").cloned().unwrap_or_else(|| "f32".into())),
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
        if matches!(n.op.as_str(), "mhc_pre" | "mhc_merge" | "mhc_post" | "latent_attention" | "router" | "swiglu") {
            boundaries.push(Boundary { kind: BoundaryKind::InternalLaunches,
                detail: "Read-only device operands borrow shared allocation leases instead of copying into staging tensors. mHC retains the residual lease through post; writes and recycling remain ordered on the owning stream".into() });
        }
        if n.op == "cache_norm_rope_append" {
            boundaries.push(Boundary { kind: BoundaryKind::InternalLaunches,
                detail: "Fused KV RMS normalization, half-rotary and paged cache write; BF16 input, gamma and rotary coefficients, epsilon 1e-5. Position is a device control input. Combined latent/rotary cache remains device resident; no implicit readback or fallback. Numerical contract differs from separate FP32 rotary".into() });
        }
        if n.op == "latent_attention" {
            boundaries.push(Boundary { kind: BoundaryKind::InternalLaunches,
                detail: "Dense FusedInferAttentionSink in TND sliding-window mode implements the validated all-causal-within-window contract (context <=512 and <=window). Bounded noncontiguous paged KV views stay on device. One AICPU metadata operation per rank/token is shared by attention nodes and captured for replay; no host length readback. This is a distinct numerical kernel from sparse Pioneer".into() });
            boundaries.push(Boundary { kind: BoundaryKind::PersistentState,
                detail: "Immutable sink latent/rotary views are packed once before capture (147456 bytes per distinct cache); a shared 4 MiB compressed causal mask and 4096-byte metadata buffer remain device resident".into() });
            boundaries.push(Boundary { kind: BoundaryKind::ControlUpload,
                detail: "Upload only sequence length each token; query length and causal mask are initialization constants. No per-token candidate-index upload".into() });
        }
        if n.op == "linear" && n.attrs.get("placement").is_some_and(|p| p == "tp_columns") {
            boundaries.push(Boundary { kind: BoundaryKind::InternalLaunches,
                detail: "Input-column TP slice loaded by generic strided initialization; local dynamic quantization and BF16 partial output. Cross-rank sum and its declared precision belong to a separate explicit allreduce node; no implicit gather".into() });
        }
        if n.op == "linear" && n.attrs.contains_key("paired_weight") {
            let Type::Tensor(input_width) = types[n.inputs[0]] else { unreachable!() };
            let packed_bytes = 2 * input_width as u64 * (n.integer("width")? + n.integer("paired_width")?) as u64;
            boundaries.push(Boundary { kind: BoundaryKind::PersistentState,
                detail: format!("Two explicitly bound replicated BF16 projections share an input. Pack their rows once on device before capture, then execute one matmul; {packed_bytes} additional bytes per distinct pair. Loaded source parameters are retained; no per-token packing or host transfer") });
        }
        if n.op == "swiglu" && n.inputs.len() == 1 {
            boundaries.push(Boundary { kind: BoundaryKind::InternalLaunches,
                detail: "Packed equal gate/up halves are borrowed directly by the compiled SwiGLU kernel; no activation concatenation copy".into() });
        }
        if matches!(
            n.op.as_str(),
            "mhc_pre" | "mhc_merge" | "expert_reduce" | "rope" | "mla_query" | "mla_value"
        ) {
            boundaries.push(Boundary {kind: BoundaryKind::InternalLaunches, detail: match n.op.as_str() {
                "mhc_pre" | "mhc_merge" => "mHC inference-only output flag permits the vendor small-token tiling; no training auxiliary values are consumed",
                "expert_reduce" => "Fused FP32 MoeFinalizeRoutingV2 weighted reduction uses original router scores and device inverse indices; drop-pad row mapping skips -1 foreign routes without a host count or score reorder",
                "mla_query" => "One batched BF16 head matmul; immutable strided weight view packed once on device before capture (786432 bytes per rank per distinct view); bounded input view, strided FP32 rotary extraction and one column-concatenation kernel; batched FP32 rotary followed by BF16 rounding",
                "mla_value" => "One batched BF16 head matmul; immutable transposed weight view packed once on device before capture (786432 bytes per rank per distinct view); no per-token weight repacking or parameter host transfer",
                _ => "Fused rotary operation in FP32 followed by BF16 rounding; shared position-derived controls remain device-resident",
            }.into()});
        }
        if matches!(
            n.op.as_str(),
            "rope" | "mla_query" | "mome" | "cache_append" | "cache_norm_rope_append"
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
                detail: "Device-packed owner-local expert banks; MoeInitRoutingV3 filters/sorts/quantizes on device into 32 per-expert counts and inverse row indices. Count-list grouped INT8 matmul skips foreign routes and zero-local ranks. BF16 checkpoint scales are required for lossless scale storage; grouped dequantization is a distinct numerical implementation. Bank memory remains additional to loaded checkpoint storage".into()});
        }
        if n.op == "expert_linear" && n.attrs.contains_key("paired_weight") {
            boundaries.push(Boundary {kind: BoundaryKind::InternalLaunches,
                detail: "Two explicitly bound projections are concatenated per owner-local expert during preparation; one input quantization and one sparse grouped matmul produce packed BF16 rows for fused activation/down. No intermediate re-packing or additional expert bank duplication".into()});
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
                dtype: if n.op == "gather" { "bf16".into() } else {
                    n.attrs.get("dtype").cloned().unwrap_or_else(|| "f32".into())
                },
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
                contract_revision: revision.clone(), numerical_contract: match n.op.as_str() {
                    "cache_norm_rope_append" => "ascend_token_v1/cache_norm_rope_append/fused_bf16_rotary_v1".into(),
                    "latent_attention" => "ascend_token_v1/latent_attention/dense_sink_bf16_v1".into(),
                    "allreduce" => format!("ascend_token_v1/allreduce/{}", n.attrs.get("dtype").map(String::as_str).unwrap_or("f32")),
                    "expert_linear" if n.attrs.contains_key("paired_weight") => "ascend_token_v1/expert_linear/paired_fused_routing_gmm_bf16_scales_v1".into(),
                    "linear" if n.attrs.get("placement").is_some_and(|p| p == "tp_columns") => "ascend_token_v1/linear/local_columns_bf16_partial_v1".into(),
                    "linear" if n.attrs.contains_key("paired_weight") => "ascend_token_v1/linear/paired_replicated_bf16_v1".into(),
                    "expert_linear" | "expert_swiglu_down" => format!("ascend_token_v1/{}/count_gmm_bf16_scales_v1", n.op),
                    "expert_reduce" => "ascend_token_v1/expert_reduce/fused_finalize_inverse_fp32_v1".into(),
                    "mhc_pre" | "mhc_merge" => format!("ascend_token_v1/{}/inference_outputs_v1", n.op),
                    "rope" => "ascend_token_v1/rope/fused_fp32_v1".into(),
                    "mla_query" | "mla_value" => format!("ascend_token_v1/{}/batched_heads_bf16_v1", n.op),
                    _ => format!("ascend_token_v1/{}", n.op),
                },
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
