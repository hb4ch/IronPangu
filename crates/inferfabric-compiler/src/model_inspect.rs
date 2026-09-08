//! Offline inspection of the checkpoint-bound model DAG.
use crate::{
    bound::{self, Op, Plan},
    checkpoint::PREFIX,
};
use inferfabric_model::{Result, invalid};
use serde::Serialize;

#[derive(Serialize)]
pub struct Group {
    pub name: String,
    pub kind: String,
    pub nodes: Vec<usize>,
}
#[derive(Serialize)]
pub struct Inspection<'a> {
    pub stage: &'static str,
    pub plan: &'a Plan,
    pub groups: Vec<Group>,
}
/// Group by explicit canonical layer-entry norms, never by tensor-name heuristics.
pub fn inspect(plan: &Plan) -> Result<Inspection<'_>> {
    bound::validate_graph(plan)?;
    let canonical = bound::compile(&plan.spec, plan.checkpoint.clone())?;
    if canonical.nodes != plan.nodes
        || canonical.states != plan.states
        || canonical.inputs != plan.inputs
        || canonical.output != plan.output
        || canonical.text_weight_bytes != plan.text_weight_bytes
        || canonical.recurrent_and_conv_bytes_per_request
            != plan.recurrent_and_conv_bytes_per_request
        || canonical.kv_bytes_per_page != plan.kv_bytes_per_page
    {
        return Err(invalid(
            "model inspection requires the canonical checkpoint-bound Qwen graph",
        ));
    }
    let mut groups = vec![Group {
        name: "Embedding".into(),
        kind: "Input".into(),
        nodes: vec![],
    }];
    for (id, node) in plan.nodes.iter().enumerate() {
        if let Op::ZeroCenteredRms { weight, .. } = &node.operation {
            if weight == &format!("{PREFIX}norm.weight") {
                groups.push(Group {
                    name: "Output head".into(),
                    kind: "Output".into(),
                    nodes: vec![],
                });
            } else if let Some(layer) = weight
                .strip_prefix(&format!("{PREFIX}layers."))
                .and_then(|s| s.strip_suffix(".input_layernorm.weight"))
            {
                let layer: usize = layer.parse().map_err(|_| invalid("invalid layer index"))?;
                groups.push(Group {
                    name: format!("Layer {layer}"),
                    kind: format!("{:?}", plan.spec.model.layers[layer]),
                    nodes: vec![],
                });
            }
        }
        groups.last_mut().unwrap().nodes.push(id);
    }
    Ok(Inspection {
        stage: "checkpoint-bound typed mathematical DAG",
        plan,
        groups,
    })
}
pub fn html(plan: &Plan) -> Result<String> {
    let data = serde_json::to_string(&inspect(plan)?)
        .map_err(|e| invalid(e.to_string()))?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    Ok(include_str!("model_inspect.html").replace("__MODEL_DATA__", &data))
}
