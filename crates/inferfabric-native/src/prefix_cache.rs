//! Complete device snapshots indexed by exact token/arithmetic identity.
//! All methods run on the engine's single owning worker. Restore calls complete
//! synchronously, so LRU retirement cannot race an in-progress restore here.
use crate::program::{Engine, Snapshot};
use inferfabric_ir::{arithmetic_history::ArithmeticHistory, serving::PrefixIdentity};
use inferfabric_model::{Result, invalid};
use inferfabric_runtime::prefix_index::{CheckpointId, PrefixIndex};
use std::collections::BTreeMap;

struct Entry {
    snapshot: Snapshot,
    slot: u32,
    touched: u64,
    namespace: String,
}
pub struct PrefixCache {
    identity: PrefixIdentity,
    indexes: BTreeMap<String, PrefixIndex>,
    alignment: u32,
    max_context: u32,
    entries: BTreeMap<CheckpointId, Entry>,
    active_slots: u32,
    snapshot_slots: u32,
    clock: u64,
}
impl PrefixCache {
    /// Snapshot residents are additional compiled storage. They are never
    /// deducted silently from an explicitly requested active-request capacity.
    pub fn new(
        engine: &Engine,
        identity: PrefixIdentity,
        active_slots: u32,
        snapshot_slots: u32,
        max_context: u32,
    ) -> Result<Self> {
        if identity.engine != *engine.identity()
            || active_slots == 0
            || snapshot_slots == 0
            || max_context == 0
            || active_slots
                .checked_add(snapshot_slots)
                .is_none_or(|n| n > engine.resident_capacity())
        {
            return Err(invalid(
                "prefix cache identity or compiled resident capacity",
            ));
        }
        PrefixIndex::new(
            identity.clone(),
            engine.checkpoint_alignment()?,
            snapshot_slots as usize,
            u64::from(snapshot_slots) * u64::from(max_context),
        )?;
        Ok(Self {
            identity,
            indexes: BTreeMap::new(),
            alignment: engine.checkpoint_alignment()?,
            max_context,
            entries: BTreeMap::new(),
            active_slots,
            snapshot_slots,
            clock: 0,
        })
    }
    fn tick(&mut self) -> Result<u64> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| invalid("prefix LRU clock exhausted"))?;
        Ok(self.clock)
    }
    fn compatible(&self, engine: &Engine, identity: &PrefixIdentity, slot: u32) -> Result<()> {
        let mut expected = self.identity.clone();
        expected.tenant_namespace = identity.tenant_namespace.clone();
        if slot >= self.active_slots
            || identity.engine != *engine.identity()
            || identity.tenant_namespace.len() > 256
            || !expected.compatible(identity)
        {
            return Err(invalid("prefix request namespace or active resident"));
        }
        Ok(())
    }
    fn evict(&mut self, engine: &mut Engine, id: CheckpointId) -> Result<u32> {
        let entry = self
            .entries
            .get(&id)
            .ok_or_else(|| invalid("prefix entry missing"))?;
        engine.retire_snapshot(entry.snapshot)?;
        let slot = entry.slot;
        let namespace = entry.namespace.clone();
        self.entries.remove(&id);
        let index = self
            .indexes
            .get_mut(&namespace)
            .ok_or_else(|| invalid("prefix namespace missing"))?;
        if !index.remove(id) {
            return Err(invalid("prefix index ownership mismatch"));
        }
        if index.is_empty() {
            self.indexes.remove(&namespace);
        }
        Ok(slot)
    }
    /// Publish a completed prefix at the source's current position. Prefix text
    /// never appears in logs; only the radix index owns its token metadata.
    pub fn publish(
        &mut self,
        engine: &mut Engine,
        identity: &PrefixIdentity,
        source: u32,
        tokens: &[u32],
    ) -> Result<bool> {
        self.compatible(engine, identity, source)?;
        let history = engine.arithmetic_history(source)?.clone();
        let boundary = history.tokens();
        if boundary == 0
            || tokens.len() != boundary as usize
            || boundary % engine.checkpoint_alignment()? != 0
            || boundary > self.max_context
        {
            return Err(invalid(
                "prefix publication requires exact completed aligned history",
            ));
        }
        let touched = self.tick()?;
        if let Some((id, length)) = self
            .indexes
            .get(&identity.tenant_namespace)
            .and_then(|index| {
                index
                    .lookup(identity, tokens, &history, boundary)
                    .checkpoint
            })
            && length == boundary
        {
            self.entries
                .get_mut(&id)
                .ok_or_else(|| invalid("prefix owner missing"))?
                .touched = touched;
            return Ok(false);
        }
        let free = (self.active_slots..self.active_slots + self.snapshot_slots)
            .find(|slot| self.entries.values().all(|entry| entry.slot != *slot));
        let slot = if let Some(slot) = free {
            slot
        } else {
            let id = *self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .ok_or_else(|| invalid("prefix LRU empty"))?
                .0;
            self.evict(engine, id)?
        };
        engine.prepare_resident(slot, boundary)?;
        engine.copy_resident(source, slot, boundary)?;
        let snapshot = engine.seal_resident(slot)?;
        if !self.indexes.contains_key(&identity.tenant_namespace) {
            self.indexes.insert(
                identity.tenant_namespace.clone(),
                PrefixIndex::new(
                    identity.clone(),
                    self.alignment,
                    self.snapshot_slots as usize,
                    u64::from(self.snapshot_slots) * u64::from(self.max_context),
                )?,
            );
        }
        let id = match self
            .indexes
            .get_mut(&identity.tenant_namespace)
            .unwrap()
            .insert(identity, tokens, history)
        {
            Ok(id) => id,
            Err(error) => {
                engine.retire_snapshot(snapshot)?;
                return Err(error);
            }
        };
        self.entries.insert(
            id,
            Entry {
                snapshot,
                slot,
                touched,
                namespace: identity.tenant_namespace.clone(),
            },
        );
        // Publication stages destination controls. Continue the source request
        // with its own tables and absolute position, preserving graph addresses.
        engine.select_resident(source)?;
        Ok(true)
    }
    /// Destination must already have a private full-request reservation.
    /// Caller retains independent sampling state and complete prompt token usage.
    pub fn restore(
        &mut self,
        engine: &mut Engine,
        identity: &PrefixIdentity,
        destination: u32,
        tokens: &[u32],
        history: &ArithmeticHistory,
    ) -> Result<u32> {
        self.compatible(engine, identity, destination)?;
        let maximum = u32::try_from(tokens.len().saturating_sub(1))
            .map_err(|_| invalid("prefix request length overflow"))?;
        let Some((id, boundary)) = self
            .indexes
            .get(&identity.tenant_namespace)
            .and_then(|index| index.lookup(identity, tokens, history, maximum).checkpoint)
        else {
            return Ok(0);
        };
        let touched = self.tick()?;
        let entry = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| invalid("prefix owner missing"))?;
        engine.restore_snapshot(entry.snapshot, destination)?;
        entry.touched = touched;
        Ok(boundary)
    }
    pub fn clear(&mut self, engine: &mut Engine) -> Result<()> {
        let ids = self.entries.keys().copied().collect::<Vec<_>>();
        for id in ids {
            self.evict(engine, id)?;
        }
        Ok(())
    }
}
