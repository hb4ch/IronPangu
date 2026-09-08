//! Serializable logical rank programs. Operations are contracts, not implemented math.
use inferfabric_model::{Role, Spec};
use serde::{Deserialize, Serialize};

pub const FORMAT_VERSION: u32 = 1;
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendKind {
    Mock,
    Ascend,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Target {
    pub backend: BackendKind,
    pub soc: String,
    pub toolchain: String,
    pub kernel_revision: String,
    pub flags: Vec<String>,
}
impl Target {
    pub fn mock() -> Self {
        Self {
            backend: BackendKind::Mock,
            soc: "cpu-mock".into(),
            toolchain: "none".into(),
            kernel_revision: "mock-v1".into(),
            flags: vec![],
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank {
    pub role: Role,
    pub tp: usize,
    pub cp: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StateKind {
    Kv,
    Recurrent,
    Conv,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StateLayout {
    pub layer: usize,
    pub kind: StateKind,
    /// Per-page KV, or per-request recurrent/conv shape. CP partitions KV pages.
    pub shape: Vec<usize>,
    pub dtype: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CommOp {
    AllReduce,
    AllGather,
    ReduceScatter,
    Send { peer: usize },
    Receive { peer: usize },
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Instruction {
    Kernel {
        layer: Option<usize>,
        name: String,
    },
    Communication {
        group: String,
        sequence: usize,
        op: CommOp,
        tensor: String,
    },
    Fence,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RankPlan {
    pub rank: Rank,
    pub states: Vec<StateLayout>,
    pub instructions: Vec<Instruction>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Artifact {
    pub version: u32,
    pub key: String,
    pub target: Target,
    pub spec: Spec,
    pub ranks: Vec<RankPlan>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bucket {
    pub batch: usize,
    pub context: usize,
}
