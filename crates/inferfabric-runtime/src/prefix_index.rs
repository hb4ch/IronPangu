//! Bounded compressed token radix index. This owns host metadata only.
//! A match is not a device-readiness certificate: the cache owner must pin a
//! complete sealed state bundle before restore and retire it after removal.
use inferfabric_ir::{arithmetic_history::ArithmeticHistory, serving::PrefixIdentity};
use inferfabric_model::{Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_INDEX: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointId {
    index: u64,
    generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Match {
    pub matched_tokens: u32,
    pub checkpoint: Option<(CheckpointId, u32)>,
}
#[derive(Default)]
struct Node {
    edge: Vec<u32>,
    checkpoints: BTreeSet<CheckpointId>,
    children: BTreeMap<u32, Node>,
}
impl Node {
    fn insert(&mut self, tokens: &[u32], id: CheckpointId) {
        if tokens.is_empty() {
            self.checkpoints.insert(id);
            return;
        }
        let Some(child) = self.children.get_mut(&tokens[0]) else {
            self.children.insert(
                tokens[0],
                Node {
                    edge: tokens.to_vec(),
                    checkpoints: BTreeSet::from([id]),
                    children: BTreeMap::new(),
                },
            );
            return;
        };
        let common = child
            .edge
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.edge.len() {
            let mut suffix = std::mem::take(child);
            child.edge = suffix.edge[..common].to_vec();
            suffix.edge.drain(..common);
            child.children.insert(suffix.edge[0], suffix);
            // Splitting a token edge creates metadata, never a checkpoint.
        }
        child.insert(&tokens[common..], id);
    }
    fn remove(&mut self, id: CheckpointId) {
        self.checkpoints.remove(&id);
        self.children.retain(|_, child| {
            child.remove(id);
            if child.checkpoints.is_empty() && child.children.len() == 1 {
                let (_, suffix) = child.children.pop_first().unwrap();
                child.edge.extend(suffix.edge);
                child.checkpoints = suffix.checkpoints;
                child.children = suffix.children;
            }
            !child.checkpoints.is_empty() || !child.children.is_empty()
        });
    }
}
struct Entry {
    boundary: u32,
    history: ArithmeticHistory,
}
pub struct PrefixIndex {
    identity: PrefixIdentity,
    index: u64,
    generation: u64,
    alignment: u32,
    max_entries: usize,
    max_token_units: u64,
    token_units: u64,
    root: Node,
    entries: BTreeMap<CheckpointId, Entry>,
}
impl PrefixIndex {
    pub fn new(
        identity: PrefixIdentity,
        alignment: u32,
        max_entries: usize,
        max_token_units: u64,
    ) -> Result<Self> {
        if !identity.compatible(&identity)
            || alignment == 0
            || max_entries == 0
            || max_token_units == 0
        {
            return Err(invalid("invalid prefix index identity/capacity"));
        }
        let index = NEXT_INDEX
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("prefix index identities exhausted"))?;
        Ok(Self {
            identity,
            index,
            generation: 0,
            alignment,
            max_entries,
            max_token_units,
            token_units: 0,
            root: Node::default(),
            entries: BTreeMap::new(),
        })
    }
    /// Caller publishes only after all state components/ranks complete and seal.
    /// Capacity rejection is atomic; eviction and device retirement are explicit.
    pub fn insert(
        &mut self,
        identity: &PrefixIdentity,
        tokens: &[u32],
        history: ArithmeticHistory,
    ) -> Result<CheckpointId> {
        let boundary =
            u32::try_from(tokens.len()).map_err(|_| invalid("prefix length overflow"))?;
        let units = self
            .token_units
            .checked_add(u64::from(boundary))
            .ok_or_else(|| invalid("prefix metadata overflow"))?;
        if !self.identity.compatible(identity)
            || boundary == 0
            || boundary % self.alignment != 0
            || history.tokens() != boundary
            || self.entries.len() >= self.max_entries
            || units > self.max_token_units
        {
            return Err(invalid(
                "prefix identity, history, alignment or metadata capacity",
            ));
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("checkpoint generations exhausted"))?;
        let id = CheckpointId {
            index: self.index,
            generation,
        };
        self.root.insert(tokens, id);
        self.entries.insert(id, Entry { boundary, history });
        self.generation = generation;
        self.token_units = units;
        Ok(id)
    }
    /// max_restore is normally prompt_len-1 when logits must be recomputed.
    /// Earlier complete checkpoints remain eligible when an edge match is longer.
    pub fn lookup(
        &self,
        identity: &PrefixIdentity,
        tokens: &[u32],
        history: &ArithmeticHistory,
        max_restore: u32,
    ) -> Match {
        let mut result = Match {
            matched_tokens: 0,
            checkpoint: None,
        };
        if !self.identity.compatible(identity) {
            return result;
        }
        let mut node = &self.root;
        let mut offset = 0;
        while let Some(token) = tokens.get(offset) {
            let Some(child) = node.children.get(token) else {
                break;
            };
            let common = child
                .edge
                .iter()
                .zip(&tokens[offset..])
                .take_while(|(a, b)| a == b)
                .count();
            offset += common;
            result.matched_tokens = offset as u32;
            if common != child.edge.len() {
                break;
            }
            for id in &child.checkpoints {
                let entry = &self.entries[id];
                if entry.boundary <= max_restore
                    && entry.history.same_prefix(history, entry.boundary)
                {
                    result.checkpoint = Some((*id, entry.boundary));
                }
            }
            node = child;
        }
        result
    }
    /// Remove metadata only after the caller has ruled out active restore pins.
    /// Stale/foreign IDs cannot remove a newer checkpoint or another index's entry.
    pub fn remove(&mut self, id: CheckpointId) -> bool {
        let Some(entry) = self.entries.remove(&id) else {
            return false;
        };
        self.root.remove(id);
        self.token_units -= u64::from(entry.boundary);
        true
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Conservative metadata bound: sum of published prefix lengths, before
    /// radix sharing. Tree edge tokens never exceed this count.
    pub fn token_units(&self) -> u64 {
        self.token_units
    }
}
