//! Host ownership metadata for preallocated device state. No tensor bytes live here.
//! Reservations span every component/rank atomically. Retired resources remain
//! occupied until explicit rank completions and outstanding borrowers release them.
use inferfabric_ir::serving::{Manifest, Storage, Retention};
use inferfabric_model::{Error, Result, invalid};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_POOL: AtomicU64 = AtomicU64::new(1);
type Key = (String, u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Lease {
    pool: u64,
    generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Borrow {
    pool: u64,
    id: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticket {
    pool: u64,
    id: u64,
}

/// Physical IDs index stable, preallocated storage; they are not device pointers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub component: String,
    pub rank: u32,
    pub units: Vec<u32>,
    pub bytes_per_unit: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageAddress {
    pub physical_page: u32,
    pub row_in_page: u32,
    /// Relative to this component/rank's preallocated storage base.
    pub byte_offset: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub component: String,
    pub rank: u32,
    pub reserved_bytes: u64,
    pub occupied_bytes: u64,
    pub retiring_bytes: u64,
    pub borrowed_bytes: u64,
}
struct Pool {
    key: Key,
    unit_bytes: u64,
    capacity: u32,
    page_tokens: Option<u32>,
    max_tokens: u32,
    retention: Retention,
    free: BTreeSet<u32>,
}
struct Allocation {
    request: u64,
    token_limit: u32,
    bindings: Vec<Binding>,
    retired: bool,
    borrowers: BTreeSet<u64>,
    pending: BTreeSet<u64>,
}
struct Flight {
    remaining_ranks: BTreeSet<u32>,
    leases: Vec<Lease>,
}

pub struct StatePool {
    id: u64,
    epoch: u64,
    max_context: u32,
    next: u64,
    pools: Vec<Pool>,
    allocations: BTreeMap<Lease, Allocation>,
    borrowers: BTreeMap<u64, Lease>,
    flights: BTreeMap<u64, Flight>,
    poisoned: bool,
}
impl StatePool {
    /// Capacities are units (pages or slots), explicitly budgeted per component/rank.
    /// Recycling is enabled only by an explicit validated retention contract.
    pub fn new(manifest: &Manifest, epoch: u64, capacities: BTreeMap<Key, u32>) -> Result<Self> {
        manifest.validate()?;
        if epoch == 0 || manifest.states.is_empty() {
            return Err(invalid("empty state pool/epoch"));
        }
        let mut pools = Vec::new();
        for state in &manifest.states {
            let (page_tokens, unit_bytes) = match state.storage {
                Storage::Pages {
                    page_tokens,
                    row_bytes,
                    ..
                } => (
                    Some(page_tokens),
                    u64::from(page_tokens)
                        .checked_mul(row_bytes)
                        .ok_or_else(|| invalid("page bytes overflow"))?,
                ),
                Storage::Slot { bytes, .. } => (None, bytes),
            };
            for &rank in &state.ranks {
                let key = (state.id.clone(), rank);
                let capacity = *capacities
                    .get(&key)
                    .ok_or_else(|| invalid("missing rank pool budget"))?;
                if capacity == 0 || unit_bytes.checked_mul(u64::from(capacity)).is_none() {
                    return Err(invalid("invalid rank pool budget"));
                }
                pools.push(Pool {
                    key,
                    unit_bytes,
                    capacity,
                    page_tokens,
                    max_tokens: state.capacity_tokens,
                    retention: state.retention,
                    free: (0..capacity).collect(),
                });
            }
        }
        if capacities.len() != pools.len() {
            return Err(invalid("unknown rank pool budget"));
        }
        let id = NEXT_POOL
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| invalid("pool identity exhausted"))?;
        Ok(Self {
            id,
            epoch,
            // Page rounding and spare pool capacity do not expand the executable
            // envelope. Per-invocation legality is still checked by validate_batch.
            max_context: manifest
                .variants
                .iter()
                .map(|v| v.max_context)
                .max()
                .unwrap(),
            next: 0,
            pools,
            allocations: BTreeMap::new(),
            borrowers: BTreeMap::new(),
            flights: BTreeMap::new(),
            poisoned: false,
        })
    }
    fn live(&self, epoch: u64) -> Result<()> {
        if self.poisoned {
            return Err(invalid("state pool poisoned by rank failure"));
        }
        if epoch != self.epoch {
            return Err(Error::Stale);
        }
        Ok(())
    }
    fn serial(&mut self) -> Result<u64> {
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| invalid("state generation exhausted"))?;
        Ok(self.next)
    }
    pub fn reserve(&mut self, epoch: u64, request: u64, tokens: u32) -> Result<Lease> {
        self.reserve_inner(epoch, request, tokens, None)
    }
    /// Some compiled backends index recurrent state by resident ID while cache
    /// pages remain shared. Pin every slot component atomically to that ID.
    pub fn reserve_at_slot(
        &mut self,
        epoch: u64,
        request: u64,
        tokens: u32,
        slot: u32,
    ) -> Result<Lease> {
        self.reserve_inner(epoch, request, tokens, Some(slot))
    }
    fn reserve_inner(
        &mut self,
        epoch: u64,
        request: u64,
        tokens: u32,
        slot: Option<u32>,
    ) -> Result<Lease> {
        self.live(epoch)?;
        if tokens == 0 || self.allocations.values().any(|a| a.request == request) {
            return Err(invalid("empty or duplicate state reservation"));
        }
        if tokens > self.max_context {
            return Err(invalid("state reservation exceeds executable context"));
        }
        let mut bindings = Vec::new();
        // Prepare every reservation before mutating any free list.
        for p in &self.pools {
            if p.retention.is_full() && tokens > p.max_tokens {
                return Err(invalid("state reservation exceeds compiled capacity"));
            }
            let retained=if p.retention.is_full(){tokens}else{tokens.min(p.max_tokens)};
            let count = p.page_tokens.map_or(1, |n| retained.div_ceil(n));
            if count as usize > p.free.len() {
                return Err(Error::Capacity(format!(
                    "state pool {} rank {}",
                    p.key.0, p.key.1
                )));
            }
            let units = if p.page_tokens.is_none()
                && let Some(slot) = slot
            {
                if !p.free.contains(&slot) {
                    return Err(Error::Capacity(format!(
                        "resident slot {slot} unavailable for {}",
                        p.key.0
                    )));
                }
                vec![slot]
            } else {
                p.free.iter().take(count as usize).copied().collect()
            };
            bindings.push(Binding {
                component: p.key.0.clone(),
                rank: p.key.1,
                units,
                bytes_per_unit: p.unit_bytes,
            });
        }
        let lease = Lease {
            pool: self.id,
            generation: self.serial()?,
        };
        for (p, b) in self.pools.iter_mut().zip(&bindings) {
            for unit in &b.units {
                p.free.remove(unit);
            }
        }
        self.allocations.insert(
            lease,
            Allocation {
                request,
                token_limit: tokens,
                bindings,
                retired: false,
                borrowers: BTreeSet::new(),
                pending: BTreeSet::new(),
            },
        );
        Ok(lease)
    }
    pub fn bindings(&self, epoch: u64, lease: Lease) -> Result<&[Binding]> {
        self.live(epoch)?;
        let a = self.allocations.get(&lease).ok_or(Error::Stale)?;
        if a.retired {
            return Err(Error::Stale);
        }
        Ok(&a.bindings)
    }
    /// Translate an absolute token position for a planned cache append. Page IDs
    /// need not be contiguous or equal across ranks. No tensor data is accessed.
    pub fn page_address(
        &self,
        epoch: u64,
        lease: Lease,
        component: &str,
        rank: u32,
        position: u32,
    ) -> Result<PageAddress> {
        let bindings = self.bindings(epoch, lease)?;
        if position >= self.allocations[&lease].token_limit {
            return Err(invalid("append exceeds request reservation"));
        }
        let index = self
            .pools
            .iter()
            .position(|p| p.key.0 == component && p.key.1 == rank)
            .ok_or_else(|| invalid("unknown state component/rank"))?;
        let pool = &self.pools[index];
        let page_tokens = pool
            .page_tokens
            .ok_or_else(|| invalid("slot is not paged state"))?;
        let logical_page=position/page_tokens;
        let page=if pool.retention.is_full(){logical_page}else{logical_page%(pool.max_tokens/page_tokens)};
        let physical_page = *bindings[index].units.get(page as usize)
            .ok_or_else(||invalid("append exceeds reserved pages"))?;
        let row_in_page = position % page_tokens;
        Ok(PageAddress {
            physical_page,
            row_in_page,
            byte_offset: u64::from(physical_page) * pool.unit_bytes
                + u64::from(row_in_page) * (pool.unit_bytes / u64::from(page_tokens)),
        })
    }
    /// Pin before publishing a reusable checkpoint or starting a restore.
    /// This is ownership only; it does not certify immutable/complete state.
    pub fn borrow(&mut self, epoch: u64, lease: Lease) -> Result<Borrow> {
        self.bindings(epoch, lease)?;
        let id = self.serial()?;
        self.allocations
            .get_mut(&lease)
            .unwrap()
            .borrowers
            .insert(id);
        self.borrowers.insert(id, lease);
        Ok(Borrow { pool: self.id, id })
    }
    pub fn borrowed_bindings(&self, epoch: u64, borrowed: Borrow) -> Result<&[Binding]> {
        self.live(epoch)?;
        if borrowed.pool != self.id {
            return Err(Error::Stale);
        }
        let lease = self.borrowers.get(&borrowed.id).ok_or(Error::Stale)?;
        Ok(&self.allocations[lease].bindings)
    }
    pub fn release(&mut self, epoch: u64, borrowed: Borrow) -> Result<()> {
        self.borrowed_bindings(epoch, borrowed)?;
        let lease = self.borrowers.remove(&borrowed.id).unwrap();
        self.allocations
            .get_mut(&lease)
            .unwrap()
            .borrowers
            .remove(&borrowed.id);
        self.reclaim();
        Ok(())
    }
    /// Call before any rank enqueues work. Borrowed reads must participate too.
    /// A failed/partially submitted group must poison the pool, never fake completion.
    pub fn submit(&mut self, epoch: u64, leases: &[Lease], reads: &[Borrow]) -> Result<Ticket> {
        self.live(epoch)?;
        let mut all = BTreeSet::new();
        for &lease in leases {
            self.bindings(epoch, lease)?;
            all.insert(lease);
        }
        for &read in reads {
            self.borrowed_bindings(epoch, read)?;
            all.insert(self.borrowers[&read.id]);
        }
        if all.is_empty() {
            return Err(invalid("empty state submission"));
        }
        let ranks = all
            .iter()
            .flat_map(|l| self.allocations[l].bindings.iter().map(|b| b.rank))
            .collect();
        let ticket = Ticket {
            pool: self.id,
            id: self.serial()?,
        };
        for lease in &all {
            self.allocations
                .get_mut(lease)
                .unwrap()
                .pending
                .insert(ticket.id);
        }
        self.flights.insert(
            ticket.id,
            Flight {
                remaining_ranks: ranks,
                leases: all.into_iter().collect(),
            },
        );
        Ok(ticket)
    }
    /// The native adapter supplies this only after the corresponding stream event.
    /// Out-of-order tickets are safe; duplicate or wrong-rank completions are errors.
    pub fn complete(&mut self, epoch: u64, ticket: Ticket, rank: u32) -> Result<()> {
        self.live(epoch)?;
        if ticket.pool != self.id {
            return Err(Error::Stale);
        }
        let flight = self.flights.get_mut(&ticket.id).ok_or(Error::Stale)?;
        if !flight.remaining_ranks.remove(&rank) {
            return Err(Error::Stale);
        }
        if flight.remaining_ranks.is_empty() {
            let flight = self.flights.remove(&ticket.id).unwrap();
            for lease in flight.leases {
                self.allocations
                    .get_mut(&lease)
                    .unwrap()
                    .pending
                    .remove(&ticket.id);
            }
            self.reclaim();
        }
        Ok(())
    }
    pub fn retire(&mut self, epoch: u64, lease: Lease) -> Result<()> {
        self.bindings(epoch, lease)?;
        self.allocations.get_mut(&lease).unwrap().retired = true;
        self.reclaim();
        Ok(())
    }
    fn reclaim(&mut self) {
        let ready: Vec<_> = self
            .allocations
            .iter()
            .filter(|(_, a)| a.retired && a.borrowers.is_empty() && a.pending.is_empty())
            .map(|(&l, _)| l)
            .collect();
        for lease in ready {
            let a = self.allocations.remove(&lease).unwrap();
            for (p, b) in self.pools.iter_mut().zip(a.bindings) {
                p.free.extend(b.units);
            }
        }
    }
    pub fn poison(&mut self) {
        self.poisoned = true;
    }
    /// Borrowed/retiring counters overlap occupied bytes; never sum those columns.
    pub fn usage(&self) -> Vec<Usage> {
        self.pools
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let bytes = |filter: fn(&Allocation) -> bool| {
                    self.allocations
                        .values()
                        .filter(|a| filter(a))
                        .map(|a| a.bindings[i].units.len() as u64 * p.unit_bytes)
                        .sum()
                };
                Usage {
                    component: p.key.0.clone(),
                    rank: p.key.1,
                    reserved_bytes: u64::from(p.capacity) * p.unit_bytes,
                    occupied_bytes: u64::from(p.capacity - p.free.len() as u32) * p.unit_bytes,
                    retiring_bytes: bytes(|a| a.retired),
                    borrowed_bytes: bytes(|a| !a.borrowers.is_empty()),
                }
            })
            .collect()
    }
}
