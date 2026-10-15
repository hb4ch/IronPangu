//! OpenPangu single-node correctness topology. TP and EP share the same ranks.
use crate::{Result, invalid};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenPanguParallel {
    pub tp: usize,
    pub ep: usize,
    pub devices: Vec<u32>,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
}
impl OpenPanguParallel {
    pub fn validate(&self) -> Result<()> {
        if self.tp != self.ep || ![1, 2, 4, 8].contains(&self.tp) || self.devices.len() != self.tp {
            return Err(invalid(
                "OpenPangu requires TP=EP in {1,2,4,8} on the same device group",
            ));
        }
        let mut ids = self.devices.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.tp || ids.iter().any(|&d| d > 1023) {
            return Err(invalid("invalid or duplicate device IDs"));
        }
        if self.max_num_seqs == 0
            || self.max_num_seqs > 1024
            || self.max_num_batched_tokens == 0
            || self.max_num_batched_tokens > 1048576
            || self.max_model_len == 0
            || self.max_model_len > 1048576
        {
            return Err(invalid("invalid OpenPangu serving capacity"));
        }
        Ok(())
    }
}
