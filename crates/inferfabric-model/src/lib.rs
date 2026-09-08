//! Hardware-independent model metadata and common errors. No tensor loading yet.
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Capacity(String),
    NotImplemented(&'static str),
    Backend(String),
    Stale,
    Io(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Layer {
    Full,
    Delta,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Prefill,
    Decode,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Mesh {
    pub tp: usize,
    pub cp: usize,
    pub sp: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Model {
    pub name: String,
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub delta_heads: usize,
    pub delta_dim: usize,
    pub conv_width: usize,
    pub layers: Vec<Layer>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Spec {
    pub model: Model,
    pub prefill: Mesh,
    pub decode: Mesh,
    pub page_tokens: usize,
    pub token_budget: usize,
    pub batch_buckets: Vec<usize>,
    pub context_buckets: Vec<usize>,
}
impl Spec {
    pub fn validate(&self) -> Result<()> {
        let m = &self.model;
        if m.name.is_empty()
            || m.layers.is_empty()
            || [
                m.hidden,
                m.intermediate,
                m.vocab,
                m.query_heads,
                m.kv_heads,
                m.head_dim,
                m.delta_heads,
                m.delta_dim,
                m.conv_width,
                self.page_tokens,
                self.token_budget,
            ]
            .contains(&0)
        {
            return Err(invalid("names, layers and dimensions must be nonzero"));
        }
        if [
            m.hidden,
            m.intermediate,
            m.vocab,
            m.query_heads,
            m.kv_heads,
            m.head_dim,
            m.delta_heads,
            m.delta_dim,
            m.conv_width,
            self.page_tokens,
            self.token_budget,
        ]
        .iter()
        .any(|&n| n > 16_777_216)
            || self
                .batch_buckets
                .iter()
                .chain(&self.context_buckets)
                .any(|&n| n > 16_777_216)
        {
            return Err(invalid("dimension exceeds skeleton metadata limit"));
        }
        if !m.query_heads.is_multiple_of(m.kv_heads) {
            return Err(invalid("Q heads must divide into KV groups"));
        }
        for mesh in [&self.prefill, &self.decode] {
            if ![1, 2].contains(&mesh.tp) || ![1, 2].contains(&mesh.cp) {
                return Err(invalid("only TP/CP 1 or 2 supported"));
            }
            if [
                m.hidden,
                m.intermediate,
                m.query_heads,
                m.kv_heads,
                m.delta_heads,
            ]
            .iter()
            .any(|n| !n.is_multiple_of(mesh.tp))
            {
                return Err(invalid("non-divisible TP shard"));
            }
            if mesh.sp && mesh.tp == 1 {
                return Err(invalid("SP requires TP=2"));
            }
        }
        if self.decode.cp != 1 || self.decode.sp || self.prefill.tp != self.decode.tp {
            return Err(invalid(
                "decode CP/SP and differing PD TP widths are unsupported",
            ));
        }
        for buckets in [&self.batch_buckets, &self.context_buckets] {
            if buckets.is_empty() || buckets[0] == 0 || buckets.windows(2).any(|w| w[0] >= w[1]) {
                return Err(invalid("buckets must be positive, ascending and unique"));
            }
        }
        // Bound metadata expansion; hardware capacity is checked separately by the backend.
        if m.layers.len() > 4096 || self.batch_buckets.len() > 64 || self.context_buckets.len() > 64
        {
            return Err(invalid("metadata exceeds skeleton limits"));
        }
        Ok(())
    }
    pub fn max_context(&self) -> usize {
        self.context_buckets.last().copied().unwrap_or(0)
    }
}
