//! Shared model authoring boundary. Compiled code and DSL elaborate to this
//! representation; neither path bypasses semantic or backend validation.
pub mod execution;
use inferfabric_model::{OpenPanguParallel, Result, Spec, invalid};
use inferfabric_plan::ir::LogicalGraph;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelDefinition {
    Qwen {
        checkpoint: bool,
        spec: Spec,
    },
    OpenPangu {
        parallel: OpenPanguParallel,
    },
    Graph {
        graph: LogicalGraph,
    },
    Execution {
        parallel: OpenPanguParallel,
        graph: execution::Graph,
    },
}
impl ModelDefinition {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Qwen { spec, .. } => spec.validate(),
            Self::OpenPangu { parallel } => parallel.validate(),
            Self::Execution { parallel, graph } => {
                parallel.validate()?;
                graph.validate()?;
                if parallel.tp != 8
                    || parallel.max_model_len != graph.max_context
                    || parallel.max_num_seqs != 1
                    || parallel.max_num_batched_tokens != 1
                {
                    return Err(invalid("ascend_token_v1 currently requires TP=EP=8"));
                }
                Ok(())
            }
            Self::Graph { graph } => inferfabric_plan::compile(graph.clone()).map(|_| ()),
        }
    }
}
/// Implement this in a compiled Rust model package, using ordinary loops,
/// functions and type checking. The result is validated just like DSL output.
pub trait ModelAuthor {
    fn define(&self) -> Result<ModelDefinition>;
}
pub fn elaborate(author: &impl ModelAuthor) -> Result<ModelDefinition> {
    let definition = author.define()?;
    definition.validate()?;
    Ok(definition)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    format: String,
    version: u32,
    definition: ModelDefinition,
}
pub fn encode(definition: &ModelDefinition) -> Result<Vec<u8>> {
    definition.validate()?;
    serde_json::to_vec_pretty(&Artifact {
        format: "inferfabric.model".into(),
        version: 1,
        definition: definition.clone(),
    })
    .map_err(|e| invalid(e.to_string()))
}
pub fn decode(bytes: &[u8]) -> Result<ModelDefinition> {
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(invalid("model artifact exceeds 16 MiB"));
    }
    let artifact: Artifact = serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
    if artifact.format != "inferfabric.model" || artifact.version != 1 {
        return Err(invalid("unsupported compiled model artifact"));
    }
    artifact.definition.validate()?;
    Ok(artifact.definition)
}
