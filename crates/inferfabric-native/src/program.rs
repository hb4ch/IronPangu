//! Persistent executable-program ABI. The handle stays on its owning worker.
use inferfabric_ir::serving::Storage;
use inferfabric_ir::serving::{Manifest, Phase};
use inferfabric_ir::{arithmetic_history::ArithmeticHistory, serving::Variant};
use inferfabric_model::{Result, invalid};
use inferfabric_runtime::state_pool::{Lease, StatePool, Ticket};
use libloading::Library;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};
type Open = unsafe extern "C" fn(*const c_char, *const c_char, u32, *mut *mut c_void) -> i32;
type Close = unsafe extern "C" fn(*mut c_void) -> i32;
type Token = unsafe extern "C" fn(*mut c_void, u32, u32, u32, *mut u32) -> i32;
type Prefill = unsafe extern "C" fn(*mut c_void, *const u32, u32, u32, u32, *mut u32) -> i32;
type Error = unsafe extern "C" fn() -> *const c_char;
type BindPages = unsafe extern "C" fn(*mut c_void, u32, *const c_char, *const i32, u32) -> i32;
type Resident = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type Capacity = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type BindState = unsafe extern "C" fn(*mut c_void, u32, u32, *const i32, u32) -> i32;
struct SharedState {
    pool: StatePool,
    leases: Vec<Option<Lease>>,
    cache_nodes: BTreeMap<String, u32>,
    ranks: u32,
    next_request: u64,
    cache_page_limits: BTreeMap<String, u32>,
    admissions: u64,
    cross_partition_bindings: u64,
    peak_occupied_bytes: u64,
    token_limits: Vec<u32>,
}
impl SharedState {
    fn new(m: &Manifest) -> Result<Self> {
        let mut capacities = BTreeMap::new();
        let mut cache_nodes = BTreeMap::new();
        let mut cache_page_limits = BTreeMap::new();
        for state in &m.states {
            let units = match state.storage {
                Storage::Pages { page_tokens, .. } => {
                    cache_nodes.insert(
                        state.id.clone(),
                        u32::try_from(state.logical_node)
                            .map_err(|_| invalid("state node overflow"))?,
                    );
                    cache_page_limits.insert(state.id.clone(), state.capacity_tokens / page_tokens);
                    state.capacity_tokens / page_tokens
                }
                Storage::Slot { .. } => 1,
            };
            for &rank in &state.ranks {
                capacities.insert(
                    (state.id.clone(), rank),
                    units
                        .checked_mul(m.resident_sequences)
                        .ok_or_else(|| invalid("pool capacity overflow"))?,
                );
            }
        }
        Ok(Self {
            pool: StatePool::new(m, 1, capacities)?,
            leases: vec![None; m.resident_sequences as usize],
            cache_nodes,
            ranks: m.identity.devices.len() as u32,
            next_request: 0,
            cache_page_limits,
            admissions: 0,
            cross_partition_bindings: 0,
            peak_occupied_bytes: 0,
            token_limits: vec![0; m.resident_sequences as usize],
        })
    }
}
pub fn manifest(bundle: &Path) -> Result<Manifest> {
    let m: Manifest = serde_json::from_slice(&std::fs::read(bundle.join("serving.json"))?)
        .map_err(|e| invalid(e.to_string()))?;
    m.validate()?;
    let hash = Sha256::digest(std::fs::read(bundle.join("execution.bin"))?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if m.identity.program_sha256 != hash {
        return Err(invalid("serving manifest/program digest mismatch"));
    }
    prefill_capacity(&m)?;
    Ok(m)
}
/// Only declared exact native shapes are accepted, not arbitrary batches.
pub fn prefill_capacity(m: &Manifest) -> Result<u32> {
    m.validate()?;
    for s in &m.states {
        if let inferfabric_ir::serving::Retention::SlidingWindow{max_query_tokens}=s.retention {
            if !matches!(&s.storage,Storage::Pages{page_tokens:16,encoding,..} if encoding=="bf16_latent512_rotary64") ||
                max_query_tokens!=m.variants.iter().map(|v|v.max_query_tokens).max().unwrap() {
                return Err(invalid("unsupported native recycled state addressing"));
            }
        }
    }
    if m.variants.is_empty()
        || m.variants.len() > 5
        || m.variants[0].id != "serial_token_v1"
        || m.variants[0].phase != Phase::Decode
        || m.variants[0].max_sequences != 1
        || m.variants[0].max_tokens != 1
        || m.variants[0].max_query_tokens != 1
        || m.variants[0].packed
        || !m.variants[0].graph_capable
        || m.variants[0].numerical_class != "ascend_serial_token_v1"
    {
        return Err(invalid("unsupported program engine serving envelope"));
    }
    let mut maximum=1;
    for v in m.variants.iter().skip(1) {
        let count = v.max_tokens;
        if !matches!(count, 2 | 4 | 8 | 16) || count<=maximum
            || v.id != format!("packed_prefill_{count}_v1")
            || v.phase != Phase::Prefill
            || v.max_sequences != 1
            || v.max_query_tokens != count
            || v.max_context != m.variants[0].max_context
            || !v.graph_capable
            || !v.packed
            || v.numerical_class != format!("packed_prefill_{count}_v1")
        {
            return Err(invalid("unsupported packed prefill contract"));
        }
        maximum=count;
    }
    Ok(maximum)
}
pub struct Engine {
    _library: Library,
    ptr: *mut c_void,
    close: Close,
    reset_fn: Close,
    token_fn: Token,
    prefill_fn: Option<Prefill>,
    prefill_tokens: u32,
    prefill_variants: Vec<u32>,
    variants: Vec<Variant>,
    histories: Vec<ArithmeticHistory>,
    history_audit_sequence: u64,
    error: Error,
    bind_pages_fn: Option<BindPages>,
    select_resident_fn: Option<Resident>,
    reset_resident_fn: Option<Resident>,
    bind_state_fn: Option<BindState>,
    shared: Option<SharedState>,
    selected: usize,
    _thread: PhantomData<Rc<()>>,
}
impl Engine {
    pub fn open(bundle: &Path, checkpoint: &Path, library: &Path) -> Result<Self> {
        let contract = manifest(bundle)?;
        let inventory = inferfabric_compiler::openpangu::inspect_inventory(checkpoint, true)?;
        if inventory.metadata_sha256 != contract.identity.checkpoint_metadata_sha256 {
            return Err(invalid("serving checkpoint metadata mismatch"));
        }
        let path = |p: &Path| {
            CString::new(p.to_string_lossy().as_bytes()).map_err(|_| invalid("path contains NUL"))
        };
        let bundle = path(bundle)?;
        let checkpoint = path(checkpoint)?;
        // Library is explicitly selected by the operator and retained beyond its handle.
        unsafe {
            let lib = Library::new(library.canonicalize()?).map_err(|e| invalid(e.to_string()))?;
            macro_rules! symbol {
                ($n:literal, $t:ty) => {
                    *lib.get::<$t>(concat!($n, "\0").as_bytes())
                        .map_err(|e| invalid(e.to_string()))?
                };
            }
            let version = symbol!(
                "inferfabric_program_abi_version",
                unsafe extern "C" fn() -> u32
            );
            if version() != 1 {
                return Err(invalid("program ABI mismatch"));
            }
            let open = symbol!("inferfabric_program_open", Open);
            let info = symbol!(
                "inferfabric_program_info",
                unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32, *mut u32) -> i32
            );
            let mut result = Self {
                close: symbol!("inferfabric_program_close", Close),
                reset_fn: symbol!("inferfabric_program_reset", Close),
                token_fn: symbol!("inferfabric_program_token", Token),
                prefill_fn: lib
                    .get::<Prefill>(b"inferfabric_program_prefill\0")
                    .ok()
                    .map(|f| *f),
                prefill_tokens: prefill_capacity(&contract)?,
                prefill_variants: contract.variants.iter().skip(1).map(|v|v.max_tokens).collect(),
                variants: contract.variants.clone(),
                histories: vec![ArithmeticHistory::default();contract.resident_sequences as usize],
                history_audit_sequence: 0,
                bind_state_fn: lib
                    .get::<BindState>(b"inferfabric_program_bind_state\0")
                    .ok()
                    .map(|f| *f),
                shared: if contract.shared_pages {
                    Some(SharedState::new(&contract)?)
                } else {
                    None
                },
                selected: 0,
                error: symbol!("inferfabric_program_error", Error),
                bind_pages_fn: lib
                    .get::<BindPages>(b"inferfabric_program_bind_pages\0")
                    .ok()
                    .map(|f| *f),
                select_resident_fn: lib
                    .get::<Resident>(b"inferfabric_program_select_resident\0")
                    .ok()
                    .map(|f| *f),
                reset_resident_fn: lib
                    .get::<Resident>(b"inferfabric_program_reset_resident\0")
                    .ok()
                    .map(|f| *f),
                ptr: std::ptr::null_mut(),
                _thread: PhantomData,
                _library: lib,
            };
            let status = open(bundle.as_ptr(), checkpoint.as_ptr(), 0, &mut result.ptr);
            result.check(status)?;
            if result.ptr.is_null() {
                return Err(invalid("native returned null engine"));
            }
            let (mut context, mut vocab, mut eos) = (0, 0, 0);
            result.check(info(result.ptr, &mut context, &mut vocab, &mut eos))?;
            let mut residents = 1;
            if let Ok(capacity) = result
                ._library
                .get::<Capacity>(b"inferfabric_program_resident_capacity\0")
            {
                result.check(capacity(result.ptr, &mut residents))?;
            }
            if residents != contract.resident_sequences
                || (residents > 1
                    && (result.select_resident_fn.is_none() || result.reset_resident_fn.is_none()))
            {
                return Err(invalid(
                    "native resident capacity differs from serving manifest",
                ));
            }
            let mut chunk = 1;
            if let Ok(capacity) = result
                ._library
                .get::<Capacity>(b"inferfabric_program_prefill_capacity\0")
            {
                result.check(capacity(result.ptr, &mut chunk))?;
            }
            if chunk != result.prefill_tokens || (chunk > 1 && result.prefill_fn.is_none()) {
                return Err(invalid(
                    "native prefill capacity differs from serving manifest",
                ));
            }
            let mut mask=if chunk>1 {1u32<<chunk} else {0};
            if let Ok(variants)=result._library.get::<Capacity>(b"inferfabric_program_prefill_variants\0") {
                result.check(variants(result.ptr,&mut mask))?;
            }
            let expected=result.prefill_variants.iter().fold(0,|bits,n|bits|(1u32<<n));
            if mask!=expected {return Err(invalid("native exact prefill variants differ from serving manifest"));}
            let mut shared = 0;
            if let Ok(policy) = result
                ._library
                .get::<Capacity>(b"inferfabric_program_shared_pages\0")
            {
                result.check(policy(result.ptr, &mut shared))?;
            }
            if shared != u32::from(contract.shared_pages)
                || (shared != 0 && (result.bind_state_fn.is_none() || result.select_resident_fn.is_none() || result.reset_resident_fn.is_none()))
            {
                return Err(invalid("native page policy differs from serving manifest"));
            }
            if (context, vocab, eos)
                != (
                    contract.variants[0].max_context,
                    contract.vocab_size,
                    contract.eos,
                )
            {
                return Err(invalid(
                    "native program capabilities differ from serving manifest",
                ));
            }
            Ok(result)
        }
    }
    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        let message = unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy();
        Err(invalid(format!("program engine: {message}")))
    }
    pub fn reset(&mut self) -> Result<()> {
        if let Some(shared) = &self.shared {
            let count = shared.leases.len();
            for slot in 0..count {
                self.reset_resident(slot as u32)?;
            }
            return self.select_resident(0);
        }
        self.check(unsafe { (self.reset_fn)(self.ptr) })?;
        for slot in 0..self.histories.len() {self.audit_history(slot)?;}
        self.histories.fill(ArithmeticHistory::default());
        self.selected=0;
        Ok(())
    }
    fn audit_history(&mut self, slot:usize) -> Result<()> {
        let Some(directory)=std::env::var_os("INFERFABRIC_ARITHMETIC_AUDIT_DIR") else {return Ok(());};
        let history=self.histories.get(slot).ok_or_else(||invalid("resident history bounds"))?;
        if history.tokens()==0 {return Ok(());}
        let sequence=self.history_audit_sequence.checked_add(1).ok_or_else(||invalid("arithmetic audit sequence overflow"))?;
        let directory=std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        let value=serde_json::json!({"format":"inferfabric.completed-arithmetic.v1","sequence":sequence,"resident":slot,"history":history});
        // Exclusive files prevent accidental replacement of evidence from another run.
        use std::io::Write;
        let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(directory.join(format!("request-{sequence}.json")))?;
        file.write_all(&serde_json::to_vec_pretty(&value).map_err(|e|invalid(e.to_string()))?)?;
        self.history_audit_sequence=sequence;
        Ok(())
    }
    pub fn select_resident(&mut self, slot: u32) -> Result<()> {
        if slot == 0 && self.select_resident_fn.is_none() {
            return Ok(());
        }
        let select = self
            .select_resident_fn
            .ok_or_else(|| invalid("backend does not support resident selection"))?;
        self.check(unsafe { select(self.ptr, slot) })?;
        self.selected = slot as usize;
        Ok(())
    }
    pub fn reset_resident(&mut self, slot: u32) -> Result<()> {
        if slot == 0 && self.reset_resident_fn.is_none() {
            return self.reset();
        }
        let reset = self
            .reset_resident_fn
            .ok_or_else(|| invalid("backend does not support resident reset"))?;
        if let Err(e) = self.check(unsafe { reset(self.ptr, slot) }) {
            if let Some(shared) = &mut self.shared {
                shared.pool.poison();
            }
            return Err(e);
        }
        self.audit_history(slot as usize)?;
        *self.histories.get_mut(slot as usize).ok_or_else(||invalid("resident history bounds"))?=ArithmeticHistory::default();
        if let Some(shared) = &mut self.shared
            && let Some(lease) = shared
                .leases
                .get_mut(slot as usize)
                .ok_or_else(|| invalid("resident slot bounds"))?
                .take()
        {
            shared.pool.retire(1, lease)?;
        }
        Ok(())
    }
    pub fn prepare_resident(&mut self, slot: u32, tokens: u32) -> Result<()> {
        self.reset_resident(slot)?;
        self.select_resident(slot)?;
        let Some(shared) = &mut self.shared else {
            return Ok(());
        };
        shared.next_request = shared
            .next_request
            .checked_add(1)
            .ok_or_else(|| invalid("request generation exhausted"))?;
        let lease = shared
            .pool
            .reserve_at_slot(1, shared.next_request, tokens, slot)?;
        shared.leases[slot as usize] = Some(lease);
        shared.token_limits[slot as usize] = tokens;
        let bindings = shared
            .pool
            .bindings(1, lease)?
            .iter()
            .filter_map(|b| {
                shared
                    .cache_nodes
                    .get(&b.component)
                    .map(|&n| (n, b.clone()))
            })
            .collect::<Vec<_>>();
        shared.cross_partition_bindings += bindings
            .iter()
            .filter(|(_, b)| {
                b.units
                    .iter()
                    .any(|&p| p / shared.cache_page_limits[&b.component] != slot)
            })
            .count() as u64;
        shared.admissions += 1;
        shared.peak_occupied_bytes = shared
            .peak_occupied_bytes
            .max(shared.pool.usage().iter().map(|u| u.occupied_bytes).sum());
        let bind = self
            .bind_state_fn
            .ok_or_else(|| invalid("missing shared page binding ABI"))?;
        for (node, b) in bindings {
            let pages = b
                .units
                .iter()
                .map(|&p| i32::try_from(p).map_err(|_| invalid("page ID overflow")))
                .collect::<Result<Vec<_>>>()?;
            if let Err(e) = self
                .check(unsafe { bind(self.ptr, b.rank, node, pages.as_ptr(), pages.len() as u32) })
            {
                self.shared.as_mut().unwrap().pool.poison();
                return Err(e);
            }
        }
        Ok(())
    }
    pub fn release_resident(&mut self, slot: u32) -> Result<()> {
        if self.shared.is_some() {
            self.reset_resident(slot)
        } else {
            Ok(())
        }
    }
    fn begin_invocation(&mut self, position: u32, count: u32) -> Result<Option<Ticket>> {
        if let Some(shared) = &mut self.shared {
            let lease = shared.leases[self.selected]
                .ok_or_else(|| invalid("resident has no state reservation"))?;
            if position
                .checked_add(count)
                .is_none_or(|end| end > shared.token_limits[self.selected])
            {
                return Err(invalid("invocation exceeds request state reservation"));
            }
            Ok(Some(shared.pool.submit(1, &[lease], &[])?))
        } else {
            Ok(None)
        }
    }
    fn finish_invocation(&mut self, ticket: Option<Ticket>, status: i32) -> Result<()> {
        let result = self.check(status);
        if let Some(shared) = &mut self.shared {
            if result.is_err() {
                shared.pool.poison();
            } else if let Some(ticket) = ticket {
                // The synchronous native ABI returns only after every rank's
                // owning stream completed. This is not a global device fence.
                for rank in 0..shared.ranks {
                    shared.pool.complete(1, ticket, rank)?;
                }
            }
        }
        result
    }
    /// Update device page metadata after reset; no cache content or graph address moves.
    pub fn bind_pages(&mut self, rank: u32, component: &str, pages: &[u32]) -> Result<()> {
        let bind = self
            .bind_pages_fn
            .ok_or_else(|| invalid("backend does not support page bindings"))?;
        let component = CString::new(component).map_err(|_| invalid("component contains NUL"))?;
        let pages = pages
            .iter()
            .map(|&p| i32::try_from(p).map_err(|_| invalid("page ID overflow")))
            .collect::<Result<Vec<_>>>()?;
        let count =
            u32::try_from(pages.len()).map_err(|_| invalid("page table length overflow"))?;
        self.check(unsafe { bind(self.ptr, rank, component.as_ptr(), pages.as_ptr(), count) })
    }
    pub fn token(&mut self, token: u32, position: u32, sample: bool) -> Result<Option<u32>> {
        self.histories[self.selected].check_append(&self.variants[0],position,1)?;
        let mut output = 0;
        let ticket = self.begin_invocation(position, 1)?;
        let status =
            unsafe { (self.token_fn)(self.ptr, token, position, sample.into(), &mut output) };
        self.finish_invocation(ticket, status)?;
        self.histories[self.selected].append_completed(&self.variants[0],position,1)?;
        Ok(sample.then_some(output))
    }
    pub fn arithmetic_history(&self, slot:u32) -> Result<&ArithmeticHistory> {
        self.histories.get(slot as usize).ok_or_else(||invalid("resident history bounds"))
    }
    pub fn prefill_capacity(&self) -> u32 {
        self.prefill_tokens
    }
    pub fn prefill_chunk(&self, limit: u32) -> u32 {
        self.prefill_variants.iter().rev().copied().find(|&n|n<=limit).unwrap_or(1)
    }
    pub fn prefill(&mut self, tokens: &[u32], position: u32, sample: bool) -> Result<Option<u32>> {
        let count=u32::try_from(tokens.len()).map_err(|_|invalid("prefill count overflow"))?;
        if !self.prefill_variants.contains(&count) {
            return Err(invalid("prefill must match the compiled chunk width"));
        }
        let variant=self.variants.iter().position(|v|v.max_tokens==count).ok_or_else(||invalid("missing arithmetic variant"))?;
        self.histories[self.selected].check_append(&self.variants[variant],position,count)?;
        let prefill = self
            .prefill_fn
            .ok_or_else(|| invalid("backend has no packed prefill entry point"))?;
        let mut output = 0;
        let ticket = self.begin_invocation(position, count)?;
        let status = unsafe {
            prefill(
                self.ptr,
                tokens.as_ptr(),
                count,
                position,
                sample.into(),
                &mut output,
            )
        };
        self.finish_invocation(ticket, status)?;
        self.histories[self.selected].append_completed(&self.variants[variant],position,count)?;
        Ok(sample.then_some(output))
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        if let Some(shared) = &self.shared {
            let occupied: u64 = shared.pool.usage().iter().map(|u| u.occupied_bytes).sum();
            eprintln!(
                "shared state pool admissions={} cross_partition_bindings={} peak_occupied_bytes={} remaining_occupied_bytes={}",
                shared.admissions,
                shared.cross_partition_bindings,
                shared.peak_occupied_bytes,
                occupied
            );
        }
        if !self.ptr.is_null() {
            unsafe {
                (self.close)(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_declared_native_chunk_envelopes_are_accepted() {
        let mut m: Manifest = serde_json::from_str(include_str!(
            "../../../docs/validation/2026-10-17-persistent-serving/serving.json"
        ))
        .unwrap();
        assert_eq!(prefill_capacity(&m).unwrap(), 1);
        let mut v = m.variants[0].clone();
        v.id = "packed_prefill_4_v1".into();
        v.numerical_class = v.id.clone();
        v.phase = Phase::Prefill;
        v.max_tokens = 4;
        v.max_query_tokens = 4;
        v.packed = true;
        m.variants.push(v);
        assert_eq!(prefill_capacity(&m).unwrap(), 4);
        let valid = m.clone();
        for count in [8,16] {
            let mut v=m.variants[1].clone();
            v.id=format!("packed_prefill_{count}_v1");v.numerical_class=v.id.clone();
            v.max_tokens=count;v.max_query_tokens=count;m.variants.push(v);
        }
        assert_eq!(prefill_capacity(&m).unwrap(),16);
        m.variants.swap(1,2);
        assert!(prefill_capacity(&m).is_err());
        m=valid.clone();
        let state=m.states.iter_mut().find(|s|matches!(s.storage,Storage::Pages{..})).unwrap();
        state.retention=inferfabric_ir::serving::Retention::SlidingWindow{max_query_tokens:4};
        state.window_tokens=Some(512);state.checkpoint_supported=false;
        state.storage=Storage::Pages{page_tokens:16,row_bytes:1152,encoding:"bf16_latent512_rotary64".into()};
        let Storage::Pages{page_tokens,..}=state.storage else {unreachable!()};
        state.capacity_tokens=inferfabric_ir::serving::sliding_capacity(512,512,4,page_tokens).unwrap();
        m.validate().unwrap();
        assert_eq!(prefill_capacity(&m).unwrap(),4);
        if let Storage::Pages{encoding,..}=&mut m.states.iter_mut().find(|s|!s.retention.is_full()).unwrap().storage {*encoding="unknown".into();}
        assert!(prefill_capacity(&m).unwrap_err().to_string().contains("recycled state addressing"));
        m=valid.clone();
        m.variants[1].max_sequences = 2;
        assert!(prefill_capacity(&m).is_err());
        m = valid.clone();
        m.variants[1].max_query_tokens = 3;
        assert!(prefill_capacity(&m).is_err());
        m = valid.clone();
        m.variants[1].max_context -= 1;
        assert!(prefill_capacity(&m).is_err());
        m = valid.clone();
        m.variants[1].numerical_class = "unqualified_arithmetic".into();
        assert!(prefill_capacity(&m).is_err());
        m = valid.clone();
        m.variants.push(valid.variants[1].clone());
        assert!(prefill_capacity(&m).is_err());
    }
}
