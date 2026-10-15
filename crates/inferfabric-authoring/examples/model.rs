//! Compiled model authoring: no DSL parser dependency.
use inferfabric_authoring::{ModelAuthor, ModelDefinition, elaborate, encode};
use inferfabric_model::{OpenPanguParallel, Result, invalid};
use inferfabric_plan::ir::{Constant, LogicalGraph, LogicalNode, Op, Tensor};
struct Author {
    openpangu: bool,
}
impl ModelAuthor for Author {
    fn define(&self) -> Result<ModelDefinition> {
        if self.openpangu {
            return Ok(ModelDefinition::OpenPangu {
                parallel: OpenPanguParallel {
                    tp: 8,
                    ep: 8,
                    devices: (0..8).collect(),
                    max_num_seqs: 4,
                    max_num_batched_tokens: 16,
                    max_model_len: 2048,
                },
            });
        }
        Ok(ModelDefinition::Graph {
            graph: LogicalGraph {
                version: 1,
                name: "Residual".into(),
                inputs: vec![Tensor {
                    name: "x".into(),
                    shape: vec![2, 2],
                }],
                constants: vec![Constant {
                    name: "weight".into(),
                    shape: vec![2, 2],
                    data: vec![1., 0., 0., 1.],
                }],
                states: vec![],
                nodes: vec![
                    LogicalNode {
                        name: "projected".into(),
                        op: Op::Matmul,
                        inputs: vec!["x".into(), "weight".into()],
                    },
                    LogicalNode {
                        name: "summed".into(),
                        op: Op::Add,
                        inputs: vec!["projected".into(), "x".into()],
                    },
                    LogicalNode {
                        name: "output".into(),
                        op: Op::Relu,
                        inputs: vec!["summed".into()],
                    },
                ],
                outputs: vec!["output".into()],
                updates: Default::default(),
            },
        })
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || !["residual", "openpangu92"].contains(&args[0].as_str()) {
        return Err(invalid(
            "usage: model residual|openpangu92 OUTPUT.ifmodel.json",
        ));
    }
    let definition = elaborate(&Author {
        openpangu: args[0] == "openpangu92",
    })?;
    std::fs::write(&args[1], encode(&definition)?)?;
    Ok(())
}
