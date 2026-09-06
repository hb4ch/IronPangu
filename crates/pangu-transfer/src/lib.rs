//! PD ownership protocol and logical resource allocation; in-memory transport only.
use pangu_model::{Error, Result, Spec, invalid};
use pangu_runtime::{DeviceRegion, HybridState};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub request: u64,
    pub epoch: u64,
    pub slot: usize,
    pub generation: u64,
    pub pages: Vec<usize>,
}
impl Lease {
    pub fn binding(&self) -> pangu_runtime::StateBinding {
        pangu_runtime::StateBinding {
            slot: self.slot,
            generation: self.generation,
            pages: self.pages.clone(),
        }
    }
}
/// Pages and slots cannot be reused until an explicit completion fence.
pub struct ResourcePool {
    page_tokens: usize,
    free_pages: BTreeSet<usize>,
    free_slots: BTreeSet<usize>,
    active: BTreeMap<u64, Lease>,
    retired: Vec<Lease>,
    generation: u64,
}
impl ResourcePool {
    pub fn new(pages: usize, slots: usize, page_tokens: usize) -> Result<Self> {
        if page_tokens == 0 || slots == 0 || pages == 0 {
            return Err(invalid("empty resource pool"));
        }
        Ok(Self {
            page_tokens,
            free_pages: (0..pages).collect(),
            free_slots: (0..slots).collect(),
            active: BTreeMap::new(),
            retired: vec![],
            generation: 0,
        })
    }
    pub fn reserve(&mut self, request: u64, epoch: u64, tokens: usize) -> Result<Lease> {
        if self.active.contains_key(&request) || self.retired.iter().any(|l| l.request == request) {
            return Err(invalid("request already owns resources"));
        }
        let count = tokens.div_ceil(self.page_tokens);
        if count > self.free_pages.len() || self.free_slots.is_empty() {
            return Err(Error::Capacity("pages/slots".into()));
        }
        let pages = self
            .free_pages
            .iter()
            .take(count)
            .copied()
            .collect::<Vec<_>>();
        for p in &pages {
            self.free_pages.remove(p);
        }
        let slot = self
            .free_slots
            .pop_first()
            .ok_or_else(|| invalid("no slot"))?;
        self.generation += 1;
        let lease = Lease {
            request,
            epoch,
            slot,
            generation: self.generation,
            pages,
        };
        self.active.insert(request, lease.clone());
        Ok(lease)
    }
    pub fn check(&self, lease: &Lease) -> Result<()> {
        match self.active.get(&lease.request) {
            Some(current)
                if current.epoch == lease.epoch
                    && current.generation == lease.generation
                    && current.slot == lease.slot =>
            {
                Ok(())
            }
            _ => Err(Error::Stale),
        }
    }
    pub fn ensure(&mut self, lease: &Lease, tokens: usize) -> Result<Lease> {
        self.check(lease)?;
        let current = self.active.get_mut(&lease.request).ok_or(Error::Stale)?;
        let additional = tokens
            .div_ceil(self.page_tokens)
            .saturating_sub(current.pages.len());
        if additional > self.free_pages.len() {
            return Err(Error::Capacity("KV growth".into()));
        }
        for _ in 0..additional {
            current.pages.push(
                self.free_pages
                    .pop_first()
                    .ok_or_else(|| invalid("no page"))?,
            );
        }
        Ok(current.clone())
    }
    pub fn retire(&mut self, lease: &Lease) -> Result<()> {
        self.check(lease)?;
        self.retired
            .push(self.active.remove(&lease.request).ok_or(Error::Stale)?);
        Ok(())
    }
    /// Caller guarantees every kernel/transfer referencing retired leases has completed.
    pub fn completion_fence(&mut self) {
        for lease in self.retired.drain(..) {
            self.free_pages.extend(lease.pages);
            self.free_slots.insert(lease.slot);
        }
    }
    pub fn allocations(&self) -> usize {
        self.active.len() + self.retired.len()
    }
    pub fn retired(&self) -> usize {
        self.retired.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub lease: Lease,
    pub model_key: String,
    pub state: HybridState,
    pub next_token: u32,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Reserved,
    InFlight,
    Complete,
    Committed,
    Acknowledged,
}
struct Entry {
    lease: Lease,
    status: Status,
    manifest: Option<Manifest>,
}
pub struct InMemoryTransport {
    pub pool: ResourcePool,
    entries: BTreeMap<u64, Entry>,
    spec: Spec,
    key: String,
}
impl InMemoryTransport {
    pub fn new(spec: Spec, key: String, pages: usize, slots: usize) -> Result<Self> {
        let pool = ResourcePool::new(pages, slots, spec.page_tokens)?;
        Ok(Self {
            pool,
            entries: BTreeMap::new(),
            spec,
            key,
        })
    }
    pub fn reserve(&mut self, id: u64, epoch: u64, tokens: usize) -> Result<Lease> {
        let lease = self.pool.reserve(id, epoch, tokens)?;
        self.entries.insert(
            id,
            Entry {
                lease: lease.clone(),
                status: Status::Reserved,
                manifest: None,
            },
        );
        Ok(lease)
    }
    fn entry(&mut self, lease: &Lease) -> Result<&mut Entry> {
        self.pool.check(lease)?;
        self.entries
            .get_mut(&lease.request)
            .filter(|e| e.lease.epoch == lease.epoch && e.lease.generation == lease.generation)
            .ok_or(Error::Stale)
    }
    pub fn transfer(&mut self, manifest: Manifest) -> Result<()> {
        if manifest.model_key != self.key {
            return Err(invalid("PD artifact mismatch"));
        }
        manifest.state.validate(&self.spec)?;
        if manifest.next_token as usize >= self.spec.model.vocab {
            return Err(invalid("PD token"));
        }
        let capacity = manifest
            .lease
            .pages
            .len()
            .checked_mul(self.spec.page_tokens)
            .ok_or_else(|| invalid("page overflow"))?;
        if manifest.state.consumed > capacity {
            return Err(invalid("PD destination too small"));
        }
        let entry = self.entry(&manifest.lease)?;
        if entry.status != Status::Reserved {
            return if entry.manifest.as_ref() == Some(&manifest) {
                Ok(())
            } else {
                Err(invalid("conflicting/late transfer"))
            };
        }
        if manifest.lease != entry.lease {
            return Err(invalid("lease mapping mismatch"));
        }
        entry.manifest = Some(manifest);
        entry.status = Status::InFlight;
        Ok(())
    }
    pub fn complete(&mut self, lease: &Lease) -> Result<()> {
        let e = self.entry(lease)?;
        match e.status {
            Status::InFlight => e.status = Status::Complete,
            Status::Complete | Status::Committed | Status::Acknowledged => {}
            _ => return Err(invalid("transfer not started")),
        };
        Ok(())
    }
    pub fn commit(&mut self, lease: &Lease) -> Result<Manifest> {
        let e = self.entry(lease)?;
        if !matches!(
            e.status,
            Status::Complete | Status::Committed | Status::Acknowledged
        ) {
            return Err(invalid("transfer not ready for commit"));
        }
        if e.status != Status::Acknowledged {
            e.status = Status::Committed;
        }
        e.manifest.clone().ok_or_else(|| invalid("missing payload"))
    }
    pub fn acknowledge(&mut self, lease: &Lease) -> Result<()> {
        let e = self.entry(lease)?;
        if !matches!(e.status, Status::Committed | Status::Acknowledged) {
            return Err(invalid("not committed"));
        }
        e.status = Status::Acknowledged;
        Ok(())
    }
    pub fn finish(&mut self, lease: &Lease) -> Result<()> {
        self.pool.check(lease)?;
        self.entries.remove(&lease.request);
        self.pool.retire(lease)
    }
    pub fn abort(&mut self, lease: &Lease) -> Result<()> {
        self.finish(lease)
    }
    pub fn completion_fence(&mut self) {
        self.pool.completion_fence();
    }
    pub fn pending(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Registration(pub u64);
#[derive(Debug, Clone, Copy)]
pub struct TransferHandle(pub u64);
/// NPU integration contract; completion must guarantee destination device visibility.
pub trait RegisteredTransport {
    fn register(&mut self, _region: DeviceRegion) -> Result<Registration> {
        Err(Error::NotImplemented("HIXL register"))
    }
    fn transfer(
        &mut self,
        _source: Registration,
        _destination: Registration,
        _bytes: usize,
    ) -> Result<TransferHandle> {
        Err(Error::NotImplemented("HIXL transfer"))
    }
    fn wait_visible(&mut self, _transfer: TransferHandle) -> Result<()> {
        Err(Error::NotImplemented("HIXL completion/device visibility"))
    }
    fn deregister(&mut self, _registration: Registration) -> Result<()> {
        Err(Error::NotImplemented("HIXL deregister"))
    }
}
pub struct AscendTransport;
impl RegisteredTransport for AscendTransport {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pages_epochs_and_fences() {
        let mut p = ResourcePool::new(4, 3, 2).unwrap();
        let a = p.reserve(1, 1, 2).unwrap();
        let b = p.reserve(2, 1, 2).unwrap();
        let a = p.ensure(&a, 4).unwrap();
        assert_eq!(a.pages, vec![0, 2]);
        assert!(p.reserve(3, 1, 4).is_err());
        p.retire(&b).unwrap();
        assert_eq!(p.retired(), 1);
        assert!(p.reserve(2, 2, 2).is_err());
        p.completion_fence();
        let new = p.reserve(2, 2, 2).unwrap();
        assert_eq!(new.slot, b.slot);
        assert_eq!(p.check(&b), Err(Error::Stale));
        p.retire(&a).unwrap();
        p.retire(&new).unwrap();
        p.completion_fence();
        assert_eq!(p.allocations(), 0);
    }
}
