//! Versioned, fail-closed serving contracts shared by compiler and scheduler.
//! This metadata never grants device readiness or numerical equivalence by itself.
use inferfabric_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub program_sha256: String,
    pub checkpoint_metadata_sha256: String,
    pub backend_revision: String,
    pub devices: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Decode,
    Prefill,
    Extend,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub id: String,
    pub phase: Phase,
    pub max_sequences: u32,
    pub max_tokens: u32,
    pub max_query_tokens: u32,
    pub max_context: u32,
    /// Equal names denote the same arithmetic contract, not merely a storage dtype.
    pub numerical_class: String,
    pub graph_capable: bool,
    pub packed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Storage {
    Pages {
        page_tokens: u32,
        row_bytes: u64,
        encoding: String,
    },
    Slot {
        bytes: u64,
        dtype: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub id: String,
    pub logical_node: usize,
    pub ranks: Vec<u32>,
    pub storage: Storage,
    /// Physical per-request retention, rounded to pages. Executable context is
    /// declared separately by variants; full retention still bounds requests.
    pub capacity_tokens: u32,
    /// Explicit addressing policy. An attention window alone never enables recycling.
    #[serde(default, skip_serializing_if = "Retention::is_full")]
    pub retention: Retention,
    /// None means retain the whole causal prefix, not an unknown window.
    pub window_tokens: Option<u32>,
    pub checkpoint_supported: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Retention {
    #[default]
    Full,
    SlidingWindow { max_query_tokens: u32 },
}
impl Retention {
    pub fn is_full(&self) -> bool { matches!(self, Self::Full) }
}

/// Maximum number of pages touched by a window plus all later writes in a chunk.
pub fn sliding_capacity(context:u32, window:u32, query:u32, page:u32)->Result<u32> {
    if context==0 || window==0 || query==0 || page==0 || query>context {
        return Err(invalid("invalid sliding retention dimensions"));
    }
    let retained=u64::from(window)+u64::from(query)-1;
    let pages=((retained+u64::from(page)-2)/u64::from(page)+1)
        .min(u64::from(context).div_ceil(u64::from(page)));
    u32::try_from(pages*u64::from(page)).map_err(|_|invalid("sliding retention overflow"))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// Storage capacity, not a claim of packed multi-sequence kernel support.
    #[serde(default = "single_resident")]
    pub resident_sequences: u32,
    /// Shared page ownership; recurrent slots remain indexed by resident ID.
    #[serde(default)]
    pub shared_pages: bool,
    pub identity: Identity,
    pub vocab_size: u32,
    pub eos: u32,
    pub variants: Vec<Variant>,
    pub states: Vec<State>,
    /// Compilation is not hardware qualification; missing evidence stays absent.
    pub qualification: Option<String>,
    pub limitations: Vec<String>,
}

fn single_resident() -> u32 { 1 }

fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
impl Manifest {
    pub fn validate(&self) -> Result<()> {
        if self.version != VERSION
            || self.resident_sequences == 0 || self.resident_sequences > 64
            || !hash(&self.identity.program_sha256)
            || !hash(&self.identity.checkpoint_metadata_sha256)
            || !hash(&self.identity.backend_revision)
            || self.identity.devices.is_empty()
            || self.identity.devices.iter().collect::<BTreeSet<_>>().len()
                != self.identity.devices.len()
            || self.vocab_size == 0
            || self.eos >= self.vocab_size
            || self.variants.is_empty()
            || self.qualification.as_ref().is_some_and(|q| !hash(q))
        {
            return Err(invalid("invalid serving identity/capabilities"));
        }
        let mut ids = BTreeSet::new();
        for v in &self.variants {
            if v.id.is_empty()
                || !ids.insert(&v.id)
                || v.numerical_class.is_empty()
                || v.max_sequences == 0
                || v.max_tokens == 0
                || v.max_query_tokens == 0
                || v.max_context == 0
                || v.max_query_tokens > v.max_context
                || v.max_query_tokens > v.max_tokens
                || (v.phase == Phase::Decode && v.max_query_tokens != 1)
                || (!v.packed && v.max_query_tokens != 1)
            {
                return Err(invalid("invalid serving variant"));
            }
        }
        let mut states = BTreeSet::new();
        let context=self.variants.iter().map(|v|v.max_context).max().unwrap();
        let query=self.variants.iter().map(|v|v.max_query_tokens).max().unwrap();
        for s in &self.states {
            if s.id.is_empty()
                || !states.insert(&s.id)
                || s.capacity_tokens == 0
                || s.ranks.is_empty()
                || s.ranks.iter().collect::<BTreeSet<_>>().len() != s.ranks.len()
                || s.ranks
                    .iter()
                    .any(|&r| r as usize >= self.identity.devices.len())
                || s.window_tokens == Some(0)
            {
                return Err(invalid("invalid serving state ownership"));
            }
            match &s.storage {
                Storage::Pages {
                    page_tokens,
                    row_bytes,
                    encoding,
                } => {
                    if *page_tokens == 0
                        || *row_bytes == 0
                        || encoding.is_empty()
                        || !s.capacity_tokens.is_multiple_of(*page_tokens)
                        || u64::from(s.capacity_tokens)
                            .checked_mul(*row_bytes)
                            .is_none()
                    {
                        return Err(invalid("invalid serving page layout"));
                    }
                }
                Storage::Slot { bytes, dtype } if *bytes == 0 || dtype.is_empty() => {
                    return Err(invalid("invalid serving state slot"));
                }
                _ => {}
            }
            if let Retention::SlidingWindow { max_query_tokens }=s.retention {
                let Storage::Pages {page_tokens,..}=s.storage else {
                    return Err(invalid("sliding retention requires paged state"));
                };
                let window=s.window_tokens.ok_or_else(||invalid("sliding retention requires a window"))?;
                if max_query_tokens<query ||
                    s.capacity_tokens!=sliding_capacity(context,window,max_query_tokens,page_tokens)? {
                    return Err(invalid("invalid sliding retention capacity/variant/checkpoint contract"));
                }
            }
        }
        Ok(())
    }

    /// Device handles and lease generations are checked by the owning runtime,
    /// after this structural check and before publishing the rank batch.
    pub fn validate_batch(&self, batch: &Batch) -> Result<()> {
        self.validate()?;
        let v = self
            .variants
            .iter()
            .find(|v| v.id == batch.variant)
            .ok_or_else(|| invalid("unknown serving variant"))?;
        if batch.engine_epoch == 0
            || batch.sequence == 0
            || batch.requests.is_empty()
            || batch.requests.len() > v.max_sequences as usize
            || batch.tokens.len() > v.max_tokens as usize
            || batch.tokens.iter().any(|&t| t >= self.vocab_size)
        {
            return Err(invalid("serving batch exceeds envelope"));
        }
        let mut ids = BTreeSet::new();
        let mut offset = 0u32;
        for r in &batch.requests {
            if r.generation == 0
                || !ids.insert(r.id)
                || r.token_offset != offset
                || r.query_tokens == 0
                || r.query_tokens > v.max_query_tokens
                || r.prior_tokens
                    .checked_add(r.query_tokens)
                    .is_none_or(|n| n > v.max_context)
                || (v.phase == Phase::Prefill && r.prior_tokens != 0)
                || (v.phase == Phase::Decode && r.query_tokens != 1)
                || r.sample_row.is_some_and(|row| row != r.query_tokens - 1)
            {
                return Err(invalid("invalid request span in serving batch"));
            }
            offset = offset
                .checked_add(r.query_tokens)
                .ok_or_else(|| invalid("batch token overflow"))?;
        }
        if offset as usize != batch.tokens.len() {
            return Err(invalid("batch token span mismatch"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RequestSpan {
    pub id: u64,
    pub generation: u64,
    pub token_offset: u32,
    pub query_tokens: u32,
    pub prior_tokens: u32,
    /// Relative final row to sample; None means a non-final prompt chunk.
    pub sample_row: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub engine_epoch: u64,
    pub sequence: u64,
    pub variant: String,
    pub requests: Vec<RequestSpan>,
    pub tokens: Vec<u32>,
}

/// A reusable checkpoint is bound to arithmetic as well as tokens and layout.
/// Cross-class equivalence is deliberately not inferred in version 1.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrefixIdentity {
    pub engine: Identity,
    pub numerical_class: String,
    pub tokenizer_template_sha256: String,
    pub tenant_namespace: String,
    pub adapter_sha256: Option<String>,
}
impl PrefixIdentity {
    pub fn compatible(&self, other: &Self) -> bool {
        !self.numerical_class.is_empty()
            && !self.tenant_namespace.is_empty()
            && hash(&self.tokenizer_template_sha256)
            && hash(&self.engine.program_sha256)
            && hash(&self.engine.backend_revision)
            && hash(&self.engine.checkpoint_metadata_sha256)
            && !self.engine.devices.is_empty()
            && self.adapter_sha256.as_ref().is_none_or(|h| hash(h))
            && self == other
    }
}
