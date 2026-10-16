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
    source.extend_from_slice(include_bytes!("token_physical.rs"));
    let revision = digest(&source);
    let mut values = vec![];
    let mut operations = vec![];
    let mut collective_sequence = 0;
    for (i, n) in graph.nodes.iter().enumerate() {
        let (dtype, shape, bytes, layout, location) = match types[i] {
            Type::Tensor(w) => (
                Some("f32".into()),
                vec![w as u64],
                Some(w as u64 * 4),
                "host_contiguous_payload",
                Location::Host,
            ),
            Type::Experts(_) => (
                None,
                vec![],
                None,
                "opaque_owner_local_expert_rows",
                Location::Opaque,
            ),
            Type::HyperConnection => (
                None,
                vec![],
                None,
                "opaque_state_and_host_values",
                Location::Opaque,
            ),
            Type::Routes => (
                None,
                vec![],
                None,
                "opaque_routing_ids_and_scores",
                Location::Opaque,
            ),
            Type::Cache => (
                None,
                vec![],
                None,
                "opaque_device_cache_reference",
                Location::Opaque,
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
        let host = matches!(
            n.op.as_str(),
            "repeat" | "slice" | "concat" | "rope" | "add" | "mhc_input" | "expert_reduce"
        );
        let mut boundaries = vec![];
        let mut boundary = |kind, detail: &str| {
            boundaries.push(Boundary {
                kind,
                detail: detail.into(),
            })
        };
        if n.op == "embedding" {
            boundary(
                BoundaryKind::DeviceToHost,
                "Read the selected weight row; unpack BF16 into a host f32 vector",
            );
        } else if !host {
            boundary(
                BoundaryKind::HostToDevice,
                "Host-staged input/control uploads occur inside this backend call; multiplicity depends on implementation/guards",
            );
            if n.op != "cache_append" {
                boundary(
                    BoundaryKind::InternalLaunches,
                    "Opaque native launches: exact descriptors, internal temporaries and workspace are not yet exported",
                );
                boundary(
                    BoundaryKind::Synchronize,
                    "Existing backend synchronizes before consuming device results; no overlap is claimed",
                );
                boundary(
                    BoundaryKind::DeviceToHost,
                    "Device results return through host values; cache hits and optional branches may bypass work",
                );
            }
        }
        boundary(
            BoundaryKind::DynamicAllocation,
            if host {
                "Host value/container allocations remain owned by the current executor"
            } else {
                "Existing host/device allocation or lazy state preparation remains internal; no arena placement is claimed"
            },
        );
        if matches!(
            n.op.as_str(),
            "mome"
                | "mhc_pre"
                | "mhc_merge"
                | "mhc_post"
                | "cache_append"
                | "latent_attention"
                | "sink_normalize"
                | "router"
        ) {
            boundary(
                BoundaryKind::PersistentState,
                "Runtime-owned prepared/state resources; sizes and reset/capture contracts require explicit lowering",
            );
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
        operations.push(Operation{name:n.name.clone(),logical_nodes:vec![i],inputs:n.inputs.clone(),collective,candidates:vec![Candidate{id:format!("ascend.token.host_staged.{}.v1",n.op),contract_revision:revision.clone(),numerical_contract:format!("ascend_token_v1/{}",n.op),location:if host{Location::Host}else{Location::Opaque},eligible:true,rejection:None,workspace_bound_bytes:if host||n.op=="embedding"{Some(0)}else{None},capture_qualified:false,boundaries,unresolved:if host{vec!["Native binary fingerprint is not bound by this inspection artifact".into()]}else{vec!["Internal launch/buffer DAG is opaque; complete device memory bound unavailable".into(),"Native binary, vendor runtime and custom kernel identities require deployment binding".into()]},cost:Cost::default()}]});
    }
    serial_plan(
        graph.name.clone(),
        "ascend_token_v1".into(),
        logical_digest,
        devices,
        operations,
        values,
        vec![graph.output],
    )
}
