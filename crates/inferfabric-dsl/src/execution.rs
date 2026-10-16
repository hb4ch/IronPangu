use super::{
    Fields,
    syntax::{Document, Expr, Member},
};
use inferfabric_authoring::{
    ModelDefinition,
    execution::{Graph, Node},
};
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use std::collections::BTreeMap;
pub fn elaborate(ast: &Document) -> Result<ModelDefinition> {
    let (expanded, origins) = crate::expand::expand(ast)?;
    let ast = &expanded;
    let mut p = Fields::new(&ast.plan)?;
    p.text("target")?;
    let parallel = OpenPanguParallel {
        tp: p.number("tp")?,
        ep: p.number("ep")?,
        devices: p
            .numbers("devices")?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| invalid("device overflow")))
            .collect::<Result<_>>()?,
        max_num_seqs: p.number("max_num_seqs")?,
        max_num_batched_tokens: p.number("max_num_batched_tokens")?,
        max_model_len: p.number("max_model_len")?,
    };
    p.finish()?;
    let mut graph = Graph {
        name: ast.name.clone(),
        max_context: 0,
        eos: usize::MAX,
        nodes: vec![],
        output: usize::MAX,
    };
    let mut names = BTreeMap::new();
    let mut fields = BTreeMap::new();
    for m in &ast.model {
        match &m.member {
            Member::Field(k, v) => {
                if fields.insert(k.clone(), v.clone()).is_some() {
                    return Err(invalid("duplicate graph field"));
                }
            }
            Member::Let(name, Expr::Call(op, args)) => {
                let mut inputs = vec![];
                let mut attrs = BTreeMap::new();
                for (key, v) in args {
                    if let Some(key) = key {
                        let value = match v {
                            Expr::Name(v) | Expr::String(v) | Expr::Number(v) => v.clone(),
                            _ => return Err(invalid("scalar attribute required")),
                        };
                        if attrs.insert(key.clone(), value).is_some() {
                            return Err(invalid("duplicate operator attribute"));
                        }
                    } else {
                        let Expr::Name(v) = v else {
                            return Err(invalid("operator input must name a prior value"));
                        };
                        inputs.push(
                            *names
                                .get(v)
                                .ok_or_else(|| invalid(format!("unknown input {v}")))?,
                        );
                    }
                }
                if names.insert(name.clone(), graph.nodes.len()).is_some() {
                    return Err(invalid("duplicate value name"));
                }
                graph.nodes.push(Node {
                    source: origins.get(name).cloned(),
                    name: name.clone(),
                    op: op.clone(),
                    inputs,
                    attrs,
                });
            }
            Member::Yield(name) => {
                if graph.output != usize::MAX {
                    return Err(invalid("multiple outputs"));
                }
                graph.output = *names.get(name).ok_or_else(|| invalid("unknown output"))?;
            }
            _ => {
                return Err(invalid(format!(
                    "byte {}: unsupported token graph declaration",
                    m.start
                )));
            }
        }
    }
    let mut f = Fields(fields);
    graph.max_context = f.number("context")?;
    graph.eos = f.number("eos")?;
    f.finish()?;
    if parallel.max_model_len != graph.max_context
        || parallel.max_num_seqs != 1
        || parallel.max_num_batched_tokens != 1
    {
        return Err(invalid(
            "ascend_token_v1 is single-sequence/token; plan budgets must match graph",
        ));
    }
    let result = ModelDefinition::Execution { parallel, graph };
    result.validate()?;
    Ok(result)
}
