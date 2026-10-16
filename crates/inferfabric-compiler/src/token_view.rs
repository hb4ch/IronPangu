//! Offline hierarchical inspector for the validated token execution IR.
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_model::{Result, invalid};

pub fn render(graph: &Graph) -> Result<String> {
    let types: Vec<_> = graph
        .validate()?
        .into_iter()
        .map(|t| match t {
            Type::Tensor(w) => serde_json::json!({"kind":"tensor", "width":w}),
            Type::Experts(w) => serde_json::json!({"kind":"experts", "width":w}),
            Type::HyperConnection => serde_json::json!({"kind":"hyperconnection"}),
            Type::Routes => serde_json::json!({"kind":"routes"}),
            Type::Cache => serde_json::json!({"kind":"cache"}),
        })
        .collect();
    // JSON lives in a script data block. Escape HTML delimiters even for imported
    // graphs whose names/source frames did not originate in the DSL lexer.
    let data = serde_json::to_string(&serde_json::json!({"graph":graph,"types":types}))
        .map_err(|e| invalid(e.to_string()))?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e");
    Ok(include_str!("token_view.html").replace("__PLAN_DATA__", &data))
}
