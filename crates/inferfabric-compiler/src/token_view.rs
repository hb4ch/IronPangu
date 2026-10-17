//! Offline hierarchical inspector for the validated token execution IR.
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_model::{Result, invalid};

pub fn render(graph: &Graph) -> Result<String> {
    let physical = crate::token_physical::lower(graph, &[0, 1, 2, 3, 4, 5, 6, 7])?;
    render_with_physical(graph, &physical)
}

pub fn render_with_physical(
    graph: &Graph,
    physical: &inferfabric_plan::physical::Plan,
) -> Result<String> {
    let program = crate::token_program::lower(graph, None)?;
    render_with_program(graph, physical, &program)
}
pub fn render_with_program(
    graph: &Graph,
    physical: &inferfabric_plan::physical::Plan,
    program: &crate::token_program::Program,
) -> Result<String> {
    program.validate()?;
    if program.instructions.iter().any(|i| i.logical_node >= graph.nodes.len()) {
        return Err(invalid("physical program source mapping"));
    }
    inferfabric_plan::physical::verify(physical)?;
    let graph_digest = crate::token_physical::digest(
        &serde_json::to_vec(graph).map_err(|e| invalid(e.to_string()))?,
    );
    if physical.logical_digest != graph_digest {
        return Err(invalid(
            "physical plan does not belong to this logical graph",
        ));
    }
    for schedule in &physical.schedules {
        if schedule.steps.len() != graph.nodes.len()
            || schedule
                .steps
                .iter()
                .zip(&graph.nodes)
                .enumerate()
                .any(|(i, (s, n))| {
                    s.operation.logical_nodes != vec![i]
                        || s.operation.inputs != n.inputs
                        || s.operation.name != n.name
                })
            || schedule.outputs != vec![graph.output]
        {
            return Err(invalid("physical/logical mapping mismatch"));
        }
    }
    let types: Vec<_> = graph
        .validate()?
        .into_iter()
        .map(|t| match t {
            Type::Tensor(w) => serde_json::json!({"kind":"tensor", "width":w}),
            Type::Experts(w) => serde_json::json!({"kind":"experts", "width":w}),
            Type::HyperConnection => serde_json::json!({"kind":"hyperconnection"}),
            Type::Routes => serde_json::json!({"kind":"routes"}),
            Type::Cache => serde_json::json!({"kind":"cache"}),
            Type::Quantized(w) => serde_json::json!({"kind":"quantized", "width":w,"dtype":"int8","scale_dtype":"f32"}),
        })
        .collect();
    // Binding roles belong to the validated operator schema, not to the browser.
    // Keep all attributes in the graph as well; this metadata only identifies
    // which attributes represent checkpoint bindings for the detail panel.
    let node_metadata: Vec<_> = graph
        .nodes
        .iter()
        .map(|node| {
            let bindings: Vec<_> = node
                .attrs
                .iter()
                .filter(|(key, _)| {
                    key.as_str() == "weight"
                        || (matches!(node.op.as_str(), "linear" | "expert_linear") && key.as_str() == "paired_weight")
                        || (node.op == "sink_normalize" && key.as_str() == "norm")
                })
                .map(|(key, value)| serde_json::json!({"attribute":key,"value":value}))
                .collect();
            serde_json::json!({"bindings":bindings})
        })
        .collect();
    // JSON lives in a script data block. Escape HTML delimiters even for imported
    // graphs whose names/source frames did not originate in the DSL lexer.
    let data = serde_json::to_string(
        &serde_json::json!({"graph":graph,"types":types,"node_metadata":node_metadata,"physical":physical,
            "program":program, "recipes":crate::token_program::recipes(program, graph),
            "serving_envelope":crate::token_serving::program_envelope(graph, physical.ranks.len(),program)?,
            "parallel": {
                "tp":physical.ranks.len(),
                "ep":if graph.nodes.iter().any(|n| n.op == "expert_linear") { physical.ranks.len() } else { 1 },
                "expert_count":graph.nodes.iter().find(|n| n.op == "router").and_then(|n| n.attrs.get("experts")).and_then(|s|s.parse::<usize>().ok()),
                "method":"Replicated token/routing → owner-local experts → explicit AllReduce of partial outputs",
                "unsupported":["All-to-all", "AllGather/ReduceScatter (AGRS)"],
                "transport":"HCCL transport is runtime-configured, not chosen by this planner"
            },
            "device_buffer_contract":inferfabric_plan::device_buffer::support()}),
    )
    .map_err(|e| invalid(e.to_string()))?
    .replace('&', "\\u0026")
    .replace('<', "\\u003c")
    .replace('>', "\\u003e");
    Ok(include_str!("token_view.html").replace("__PLAN_DATA__", &data))
}
