//! Versioned interchange schemas. Unknown fields are rejected at every level.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Add,
    Mul,
    Relu,
    Matmul,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Constant {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LogicalNode {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LogicalGraph {
    pub version: u32,
    pub name: String,
    pub inputs: Vec<Tensor>,
    pub constants: Vec<Constant>,
    pub states: Vec<Constant>,
    pub nodes: Vec<LogicalNode>,
    pub outputs: Vec<String>,
    /// Simultaneous invocation-end assignment: state name -> value name.
    pub updates: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Storage {
    Input,
    Constant,
    State,
    Arena,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Value {
    pub name: String,
    pub shape: Vec<usize>,
    pub storage: Storage,
    pub initial: Option<Vec<f32>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TypedNode {
    pub name: String,
    pub op: Op,
    pub inputs: Vec<usize>,
    pub output: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TypedGraph {
    pub version: u32,
    pub name: String,
    pub values: Vec<Value>,
    pub nodes: Vec<TypedNode>,
    pub outputs: Vec<usize>,
    pub updates: BTreeMap<usize, usize>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kernel {
    AddF32V1,
    MulF32V1,
    ReluF32V1,
    MatmulF32V1,
    MatmulBlockedF32V1,
}
impl Kernel {
    pub fn op(self) -> Op {
        match self {
            Self::AddF32V1 => Op::Add,
            Self::MulF32V1 => Op::Mul,
            Self::ReluF32V1 => Op::Relu,
            Self::MatmulF32V1 | Self::MatmulBlockedF32V1 => Op::Matmul,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub value: Value,
    /// Arena offset, in bytes. Owned input/constants/state never alias the arena.
    pub offset: Option<usize>,
    /// Inclusive producing/last-consuming schedule positions; N means final commit/output.
    pub lifetime: Option<[usize; 2]>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub name: String,
    pub kernel: Kernel,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub rank: usize,
    pub stream: usize,
    pub dependencies: Vec<usize>,
    pub reuse_dependencies: Vec<usize>,
    pub selection_reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicalPlan {
    pub version: u32,
    pub name: String,
    pub target: String,
    pub policy: String,
    pub alignment: usize,
    pub arena_bytes: usize,
    pub buffers: Vec<Buffer>,
    pub steps: Vec<Step>,
    pub outputs: Vec<usize>,
    pub updates: BTreeMap<usize, usize>,
    pub removed_nodes: Vec<String>,
    pub folded_nodes: Vec<String>,
}
