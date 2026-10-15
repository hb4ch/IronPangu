//! Versioned authoring frontend. Parsing creates an AST, then elaboration
//! produces the same model definition as compiled-language authors.
pub mod openpangu;
pub mod syntax;
use authoring::ModelDefinition;
pub use inferfabric_authoring as authoring;
use inferfabric_model::{Layer, Mesh, Model, OpenPanguParallel, Result, Spec, invalid};
use inferfabric_plan::ir::{Constant, LogicalGraph, LogicalNode, Op, Tensor};
use std::collections::BTreeMap;
use syntax::{Expr, Member, SpannedMember};
struct Fields(BTreeMap<String, Expr>);
impl Fields {
    fn new(members: &[SpannedMember]) -> Result<Self> {
        let mut fields = BTreeMap::new();
        for m in members {
            let Member::Field(k, v) = &m.member else {
                return Err(invalid(format!(
                    "byte {}: graph declaration not allowed in contract adapter",
                    m.start
                )));
            };
            if fields.insert(k.clone(), v.clone()).is_some() {
                return Err(invalid(format!("byte {}: duplicate field {k}", m.start)));
            }
        }
        Ok(Self(fields))
    }
    fn take(&mut self, k: &str) -> Result<Expr> {
        self.0
            .remove(k)
            .ok_or_else(|| invalid(format!("missing field {k}")))
    }
    fn text(&mut self, k: &str) -> Result<String> {
        match self.take(k)? {
            Expr::Name(s) | Expr::String(s) => Ok(s),
            _ => Err(invalid(format!("{k} requires name/string"))),
        }
    }
    fn number(&mut self, k: &str) -> Result<usize> {
        match self.take(k)? {
            Expr::Number(s) => s
                .parse()
                .map_err(|_| invalid(format!("invalid integer {k}"))),
            _ => Err(invalid(format!("{k} requires integer"))),
        }
    }
    fn numbers(&mut self, k: &str) -> Result<Vec<usize>> {
        match self.take(k)? {
            Expr::List(values) => values
                .into_iter()
                .map(|v| match v {
                    Expr::Number(s) => s.parse().map_err(|_| invalid("invalid integer list")),
                    _ => Err(invalid("expected integer list")),
                })
                .collect(),
            _ => Err(invalid("expected list")),
        }
    }
    fn mesh(&mut self, k: &str) -> Result<Mesh> {
        let Expr::Call(name, args) = self.take(k)? else {
            return Err(invalid("expected mesh(tp=..., cp=..., sp=...)"));
        };
        if name != "mesh" {
            return Err(invalid("expected mesh constructor"));
        }
        let mut fields = BTreeMap::new();
        for (key, value) in args {
            let key = key.ok_or_else(|| invalid("mesh requires named arguments"))?;
            if fields.insert(key, value).is_some() {
                return Err(invalid("duplicate mesh argument"));
            }
        }
        let mut f = Self(fields);
        let tp = f.number("tp")?;
        let cp = f.number("cp")?;
        let sp = match f.text("sp")?.as_str() {
            "true" => true,
            "false" => false,
            _ => return Err(invalid("invalid SP boolean")),
        };
        f.finish()?;
        Ok(Mesh { tp, cp, sp })
    }
    fn finish(self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid(format!("unsupported fields: {:?}", self.0.keys())))
        }
    }
}
pub fn elaborate(ast: &syntax::Document) -> Result<ModelDefinition> {
    let contract = ast.model.iter().find_map(|m| match &m.member {
        Member::Field(k, Expr::Name(v)) if k == "contract" => Some(v.as_str()),
        _ => None,
    });
    let result = if let Some(contract) = contract {
        let mut m = Fields::new(&ast.model)?;
        let mut p = Fields::new(&ast.plan)?;
        m.text("contract")?;
        match contract {
            "openpangu_v2_tp_ep_v1" => {
                m.finish()?;
                let parallel = OpenPanguParallel {
                    tp: p.number("tp")?,
                    ep: p.number("ep")?,
                    devices: p
                        .numbers("devices")?
                        .into_iter()
                        .map(|n| u32::try_from(n).map_err(|_| invalid("device ID overflow")))
                        .collect::<Result<_>>()?,
                    max_num_seqs: p.number("max_num_seqs")?,
                    max_num_batched_tokens: p.number("max_num_batched_tokens")?,
                    max_model_len: p.number("max_model_len")?,
                };
                p.finish()?;
                ModelDefinition::OpenPangu { parallel }
            }
            "qwen_structure_v1" | "qwen35_text_v1" => {
                let model = Model {
                    name: m.text("label")?,
                    hidden: m.number("hidden")?,
                    intermediate: m.number("intermediate")?,
                    vocab: m.number("vocab")?,
                    query_heads: m.number("query_heads")?,
                    kv_heads: m.number("kv_heads")?,
                    head_dim: m.number("head_dim")?,
                    delta_heads: m.number("delta_heads")?,
                    delta_dim: m.number("delta_dim")?,
                    conv_width: m.number("conv_width")?,
                    layers: match m.take("layers")? {
                        Expr::List(values) => values
                            .into_iter()
                            .map(|v| match v {
                                Expr::Name(s) if s == "full" => Ok(Layer::Full),
                                Expr::Name(s) if s == "delta" => Ok(Layer::Delta),
                                _ => Err(invalid("unsupported layer contract")),
                            })
                            .collect::<Result<_>>()?,
                        _ => return Err(invalid("layers requires a list")),
                    },
                };
                let spec = Spec {
                    model,
                    prefill: p.mesh("prefill")?,
                    decode: p.mesh("decode")?,
                    page_tokens: p.number("page_tokens")?,
                    token_budget: p.number("token_budget")?,
                    batch_buckets: p.numbers("batch_buckets")?,
                    context_buckets: p.numbers("context_buckets")?,
                };
                m.finish()?;
                p.finish()?;
                ModelDefinition::Qwen {
                    checkpoint: contract == "qwen35_text_v1",
                    spec,
                }
            }
            _ => return Err(invalid("unsupported model contract")),
        }
    } else {
        let mut p = Fields::new(&ast.plan)?;
        if p.text("target")? != "cpu_f32_v1" {
            return Err(invalid("graph target not implemented"));
        }
        p.finish()?;
        let mut graph = LogicalGraph {
            version: 1,
            name: ast.name.clone(),
            inputs: vec![],
            constants: vec![],
            states: vec![],
            nodes: vec![],
            outputs: vec![],
            updates: BTreeMap::new(),
        };
        for member in &ast.model {
            match &member.member {
                Member::Input(name, shape) => graph.inputs.push(Tensor {
                    name: name.clone(),
                    shape: shape.clone(),
                }),
                Member::Constant(name, shape, data) => graph.constants.push(Constant {
                    name: name.clone(),
                    shape: shape.clone(),
                    data: data.clone(),
                }),
                Member::State(name, shape, data) => graph.states.push(Constant {
                    name: name.clone(),
                    shape: shape.clone(),
                    data: data.clone(),
                }),
                Member::Let(name, Expr::Call(op, args)) => {
                    let op = match op.as_str() {
                        "add" => Op::Add,
                        "mul" => Op::Mul,
                        "relu" => Op::Relu,
                        "matmul" => Op::Matmul,
                        _ => {
                            return Err(invalid(format!(
                                "byte {}: unsupported graph operation {op}",
                                member.start
                            )));
                        }
                    };
                    let inputs = args
                        .iter()
                        .map(|(key, value)| match (key, value) {
                            (None, Expr::Name(n)) => Ok(n.clone()),
                            _ => Err(invalid("graph operation requires positional value names")),
                        })
                        .collect::<Result<_>>()?;
                    graph.nodes.push(LogicalNode {
                        name: name.clone(),
                        op,
                        inputs,
                    });
                }
                Member::Yield(name) => graph.outputs.push(name.clone()),
                Member::Update(name, value) => {
                    if graph.updates.insert(name.clone(), value.clone()).is_some() {
                        return Err(invalid("duplicate state update"));
                    }
                }
                _ => {
                    return Err(invalid(format!(
                        "byte {}: unsupported graph declaration",
                        member.start
                    )));
                }
            }
        }
        ModelDefinition::Graph { graph }
    };
    result.validate()?;
    Ok(result)
}
/// Accept a v1 DSL document or a versioned artifact emitted by compiled code.
/// Never execute a supplied program, import, or host callback while parsing.
pub fn definition(source: &str) -> Result<ModelDefinition> {
    if source.trim_start().starts_with('{') {
        authoring::decode(source.as_bytes())
    } else {
        elaborate(&syntax::parse(source)?)
    }
}
pub fn parse(source: &str) -> Result<Spec> {
    match definition(source)? {
        ModelDefinition::Qwen { spec, .. } => Ok(spec),
        _ => Err(invalid("expected Qwen model definition")),
    }
}
pub fn parse_checkpoint(source: &str) -> Result<Spec> {
    match definition(source)? {
        ModelDefinition::Qwen {
            checkpoint: true,
            spec,
        } => Ok(spec),
        _ => Err(invalid("requires qwen35_text_v1 checkpoint contract")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    const EXAMPLE: &str = include_str!("../../../examples/qwen35-2b.inferfabric");
    #[test]
    fn versioned_contracts_and_compiled_artifacts() {
        assert_eq!(parse(EXAMPLE).unwrap().model.layers.len(), 24);
        assert!(parse_checkpoint(EXAMPLE).is_err());
        assert!(parse_checkpoint(&EXAMPLE.replace("qwen_structure_v1", "qwen35_text_v1")).is_ok());
        let d = definition(EXAMPLE).unwrap();
        let bytes = authoring::encode(&d).unwrap();
        assert_eq!(definition(std::str::from_utf8(&bytes).unwrap()).unwrap(), d);
    }
    #[test]
    fn reject_legacy_and_invalid_sources() {
        for source in [
            "model = Qwen\nhidden = 2048".to_string(),
            EXAMPLE.replace("inferfabric 1;", "inferfabric 2;"),
            EXAMPLE.replace("hidden = 2048;", "hidden = 0;"),
            EXAMPLE.replace("hidden = 2048;", "hidden = 2048; hidden = 2048;"),
            EXAMPLE.replace("hidden = 2048;", "unknown = 2048;"),
            EXAMPLE.replace("cp = 1", "cp = 2"),
            format!("{EXAMPLE} junk"),
        ] {
            assert!(definition(&source).is_err(), "{source}");
        }
    }
}
