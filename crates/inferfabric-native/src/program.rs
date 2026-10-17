//! Persistent executable-program ABI. The handle stays on its owning worker.
use inferfabric_ir::serving::Storage;
use inferfabric_ir::serving::{Manifest, Phase};
use inferfabric_ir::{arithmetic_history::ArithmeticHistory, serving::Variant};
use inferfabric_model::{Result, invalid};
use inferfabric_runtime::state_pool::{Lease, SealedLease, StatePool, Ticket};
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
type Batch = unsafe extern "C" fn(*mut c_void,*const NativeSequence,u32,*const u32,u32,*mut u32,u32,*mut u32)->i32;
#[repr(C)]
#[derive(Clone, Copy)]
struct NativeSequence { resident:u32, position:u32, token_count:u32, sample:u32 }
#[derive(Clone, Debug)]
pub struct BatchInput {
    pub resident:u32,
    pub position:u32,
    pub tokens:Vec<u32>,
    pub sample:bool,
}
fn pack_batch(inputs:&[BatchInput],residents:usize,budget:u32)->Result<(Vec<NativeSequence>,Vec<u32>,String)> {
    if inputs.is_empty() || inputs.len()>residents {return Err(invalid("batch resident count"));}
    let mut seen=std::collections::BTreeSet::new();
    let mut descriptors=Vec::new();let mut tokens=Vec::new();let mut hash=Sha256::new();
    for input in inputs {
        let count=u32::try_from(input.tokens.len()).map_err(|_|invalid("batch token count overflow"))?;
        if input.resident as usize>=residents || !seen.insert(input.resident) || count==0 ||
            input.position.checked_add(count).is_none() || tokens.len()+input.tokens.len()>budget as usize {
            return Err(invalid("invalid batch member or token budget"));
        }
        descriptors.push(NativeSequence{resident:input.resident,position:input.position,token_count:count,sample:input.sample.into()});
        // Until batching-independent arithmetic is qualified, peer content and
        // sequence geometry are part of the exact prefix compatibility domain.
        for value in [input.position,count] {hash.update(value.to_le_bytes());}
        for token in &input.tokens {hash.update(token.to_le_bytes());}
        tokens.extend_from_slice(&input.tokens);
    }
    let digest=hash.finalize().iter().map(|b|format!("{b:02x}")).collect();
    Ok((descriptors,tokens,digest))
}

fn batch_base_variant(variants:&[Variant],tokens:u32)->Result<&Variant> {
    variants.iter().find(|v|v.phase==Phase::Prefill && !v.graph_capable && v.accepts_query_tokens(tokens))
        .ok_or_else(||invalid("batch requires uncaptured packed variant"))
}
fn batch_variant(base:&Variant,digest:&str,member:usize)->Variant {
    let mut variant=base.clone();
    variant.numerical_class=format!("{}:sequence-batch-v1:{digest}:member={member}",base.numerical_class);
    variant
}
/// Prospective identity for an isolated chunk. Physical slot assignment does
/// not affect arithmetic; peer content and batch geometry still do. This must
/// never grant compatibility to a snapshot produced in a different batch.
pub fn prospective_single_batch_variant(variants:&[Variant],tokens:&[u32],position:u32,budget:u32)->Result<Variant> {
    let inputs=[BatchInput{resident:0,position,tokens:tokens.to_vec(),sample:false}];
    let (_,packed,digest)=pack_batch(&inputs,1,budget)?;
    let base=batch_base_variant(variants,packed.len() as u32)?;
    if position+packed.len() as u32>base.max_context {return Err(invalid("prospective batch exceeds context"));}
    Ok(batch_variant(base,&digest,0))
}
type Error = unsafe extern "C" fn() -> *const c_char;
type BindPages = unsafe extern "C" fn(*mut c_void, u32, *const c_char, *const i32, u32) -> i32;
type Resident = unsafe extern "C" fn(*mut c_void, u32) -> i32;
type CopyResident = unsafe extern "C" fn(*mut c_void, u32, u32, u32) -> i32;
type Capacity = unsafe extern "C" fn(*mut c_void, *mut u32) -> i32;
type PrefillSizes = unsafe extern "C" fn(*mut c_void, *mut u32, u32, *mut u32) -> i32;
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
        || m.variants.len() > 6
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
        let uncaptured = !v.graph_capable;
        let id = if uncaptured {format!("uncaptured_prefill_{count}_v1")} else {format!("packed_prefill_{count}_v1")};
        if (if uncaptured { count!=4096 || m.variants.len()!=2 } else { !matches!(count, 2 | 4 | 8 | 16 | 32) }) || count<=maximum
            || v.id != id
            || v.phase != Phase::Prefill
            || v.max_sequences != 1
            || v.max_query_tokens != count
            || v.max_context != m.variants[0].max_context
            || !v.packed
            || v.numerical_class != id
        {
            return Err(invalid("unsupported packed prefill contract"));
        }
        maximum=count;
    }
    Ok(maximum)
}
fn copy_history(histories:&[ArithmeticHistory],source:u32,destination:u32,boundary:u32,pages:&[u32])->Result<ArithmeticHistory> {
    let from=histories.get(source as usize).ok_or_else(||invalid("state copy source bounds"))?;
    let to=histories.get(destination as usize).ok_or_else(||invalid("state copy destination bounds"))?;
    if source==destination || boundary==0 || from.tokens()!=boundary || to.tokens()!=0 ||
        pages.iter().any(|&page|page==0 || boundary%page!=0) {
        return Err(invalid("state copy requires complete aligned source and empty destination history"));
    }
    from.prefix(boundary)
}
/// Opaque immutable resident generation. It does not carry host tensor contents.
#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Snapshot {slot:u32, lease:SealedLease}
pub struct Engine {
    identity: inferfabric_ir::serving::Identity,
    _library: Library,
    ptr: *mut c_void,
    close: Close,
    reset_fn: Close,
    token_fn: Token,
    prefill_fn: Option<Prefill>,
    batch_fn: Option<Batch>,
    prefill_tokens: u32,
    uncaptured_prefill: bool,
    prefill_variants: Vec<u32>,
    variants: Vec<Variant>,
    histories: Vec<ArithmeticHistory>,
    imported_tokens: Vec<u32>,
    sealed: Vec<Option<SealedLease>>,
    copy_page_tokens: Vec<u32>,
    history_audit_sequence: u64,
    error: Error,
    bind_pages_fn: Option<BindPages>,
    select_resident_fn: Option<Resident>,
    reset_resident_fn: Option<Resident>,
    copy_resident_fn: Option<CopyResident>,
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
                batch_fn:lib.get::<Batch>(b"inferfabric_program_batch\0").ok().map(|f|*f),
                uncaptured_prefill: contract.variants.iter().any(|v|v.phase==Phase::Prefill && !v.graph_capable),
                prefill_variants: contract.variants.iter().skip(1).filter(|v|v.graph_capable).map(|v|v.max_tokens).collect(),
                variants: contract.variants.clone(),
                identity: contract.identity.clone(),
                histories: vec![ArithmeticHistory::default();contract.resident_sequences as usize],
                imported_tokens: vec![0;contract.resident_sequences as usize],
                sealed: vec![None;contract.resident_sequences as usize],
                copy_page_tokens: contract.states.iter().filter_map(|s|match s.storage {Storage::Pages{page_tokens,..}=>Some(page_tokens),_=>None}).collect(),
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
                copy_resident_fn: lib.get::<CopyResident>(b"inferfabric_program_copy_resident\0").ok().map(|f|*f),
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
            let mut mode=0;
            if let Ok(query)=result._library.get::<Capacity>(b"inferfabric_program_prefill_mode\0") {
                result.check(query(result.ptr,&mut mode))?;
            }
            if mode!=u32::from(result.uncaptured_prefill) {
                return Err(invalid("native prefill execution mode differs from serving manifest"));
            }
            if let Ok(sizes)=result._library.get::<PrefillSizes>(b"inferfabric_program_prefill_sizes\0") {
                let mut observed=vec![0;result.prefill_variants.len()];
                let mut count=0;
                result.check(sizes(result.ptr,observed.as_mut_ptr(),observed.len() as u32,&mut count))?;
                if count as usize!=observed.len() || observed!=result.prefill_variants {
                    return Err(invalid("native exact prefill sizes differ from serving manifest"));
                }
            } else {
                if chunk>=32 || result.prefill_variants.iter().any(|&n|n>=32) {
                    return Err(invalid("large prefill requires native exact-size ABI"));
                }
                let mut mask=if chunk>1 {1u32<<chunk} else {0};
                if let Ok(variants)=result._library.get::<Capacity>(b"inferfabric_program_prefill_variants\0") {
                    result.check(variants(result.ptr,&mut mask))?;
                }
                let expected=result.prefill_variants.iter().fold(0,|bits,n|bits|(1u32<<n));
                if mask!=expected {return Err(invalid("native exact prefill variants differ from serving manifest"));}
            }
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
        if self.sealed.iter().any(Option::is_some) {return Err(invalid("explicitly retire snapshots before engine reset"));}
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
        self.imported_tokens.fill(0);
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
        let value=serde_json::json!({"format":"inferfabric.completed-arithmetic.v1","sequence":sequence,"resident":slot,"history":history,"imported_tokens":self.imported_tokens[slot]});
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
        if self.sealed.get(slot as usize).ok_or_else(||invalid("resident slot bounds"))?.is_some() {
            return Err(invalid("sealed snapshot cannot be reset as a request"));
        }
        if let Some(shared)=&self.shared && let Some(lease)=shared.leases[slot as usize] {
            shared.pool.check_idle(1,lease)?;
        }
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
        self.imported_tokens[slot as usize]=0;
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
    /// Explicit complete-state copy into an already reserved, reset resident.
    /// This is not cache publication: the caller must separately seal ownership
    /// and establish namespace/token identity before offering a prefix hit.
    pub fn copy_resident(&mut self, source:u32, destination:u32, boundary:u32) -> Result<()> {
        let copy=self.copy_resident_fn.ok_or_else(||invalid("backend has no explicit state-copy ABI"))?;
        let history=copy_history(&self.histories,source,destination,boundary,&self.copy_page_tokens)?;
        let shared=self.shared.as_mut().ok_or_else(||invalid("state copy requires shared ownership reservations"))?;
        let source_lease=shared.leases[source as usize].ok_or_else(||invalid("copy source has no reservation"))?;
        let destination_lease=shared.leases[destination as usize].ok_or_else(||invalid("copy destination has no reservation"))?;
        if boundary>shared.token_limits[destination as usize] {
            return Err(invalid("state copy exceeds destination reservation"));
        }
        let read=shared.pool.borrow(1,source_lease)?;
        let ticket=match shared.pool.submit(1,&[destination_lease],&[read]) {
            Ok(ticket)=>ticket,
            Err(error)=>{shared.pool.release(1,read)?;return Err(error);}
        };
        let status=unsafe{copy(self.ptr,source,destination,boundary)};
        self.finish_invocation(Some(ticket),status)?;
        self.shared.as_mut().unwrap().pool.release(1,read)?;
        self.histories[destination as usize]=history;
        self.imported_tokens[destination as usize]=boundary;
        Ok(())
    }
    /// Seal only complete, page-aligned state after its synchronous execution or
    /// copy. Publication in a token index remains the cache owner's next step.
    pub fn seal_resident(&mut self,slot:u32)->Result<Snapshot> {
        let history=self.arithmetic_history(slot)?;
        if history.tokens()==0 || self.copy_page_tokens.iter().any(|p|history.tokens()%p!=0) {
            return Err(invalid("snapshot requires a complete aligned history"));
        }
        if self.copy_resident_fn.is_none() {return Err(invalid("backend cannot restore device snapshots"));}
        let shared=self.shared.as_mut().ok_or_else(||invalid("snapshot requires shared reservations"))?;
        let lease=shared.leases[slot as usize].ok_or_else(||invalid("snapshot resident has no reservation"))?;
        let sealed=shared.pool.seal(1,lease)?;
        self.sealed[slot as usize]=Some(sealed);
        Ok(Snapshot{slot,lease:sealed})
    }
    pub fn identity(&self)->&inferfabric_ir::serving::Identity {&self.identity}
    pub fn resident_capacity(&self)->u32 {self.histories.len() as u32}
    pub fn checkpoint_alignment(&self)->Result<u32> {
        let mut result=1u32;
        for &page in &self.copy_page_tokens {
            let (mut a,mut b)=(result,page);
            while b!=0 {(a,b)=(b,a%b);}
            result=(result/a).checked_mul(page).ok_or_else(||invalid("checkpoint alignment overflow"))?;
        }
        Ok(result)
    }
    fn check_snapshot(&self,snapshot:Snapshot)->Result<()> {
        if self.sealed.get(snapshot.slot as usize)!=Some(&Some(snapshot.lease)) {
            return Err(invalid("stale or foreign device snapshot"));
        }
        Ok(())
    }
    pub fn restore_snapshot(&mut self,snapshot:Snapshot,destination:u32)->Result<()> {
        self.check_snapshot(snapshot)?;
        self.copy_resident(snapshot.slot,destination,self.histories[snapshot.slot as usize].tokens())
    }
    /// The owning worker serializes restore and retirement. Check readers/ranks
    /// before allowing the ordinary reset path to clear and retire storage.
    pub fn retire_snapshot(&mut self,snapshot:Snapshot)->Result<()> {
        self.check_snapshot(snapshot)?;
        let shared=self.shared.as_ref().ok_or_else(||invalid("snapshot pool missing"))?;
        let lease=shared.leases[snapshot.slot as usize].ok_or_else(||invalid("snapshot lease missing"))?;
        shared.pool.check_idle(1,lease)?;
        self.sealed[snapshot.slot as usize]=None;
        if let Err(error)=self.reset_resident(snapshot.slot) {
            // A failure may occur after device zeroing (for example audit I/O).
            // Never expose the old checkpoint as usable after partial teardown.
            if let Some(shared)=&mut self.shared {shared.pool.poison();}
            self.sealed[snapshot.slot as usize]=Some(snapshot.lease);
            return Err(error);
        }
        Ok(())
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
    /// One native forward; all leases remain in flight until all rank streams
    /// complete. The current native implementation rejects unqualified arenas.
    pub fn batch(&mut self,inputs:&[BatchInput])->Result<Vec<Option<u32>>> {
        let call=self.batch_fn.ok_or_else(||invalid("native batch ABI unavailable"))?;
        let (descriptors,tokens,digest)=pack_batch(inputs,self.histories.len(),self.prefill_tokens)?;
        let base=batch_base_variant(&self.variants,tokens.len() as u32)?;
        let mut variants=Vec::new();
        for (index,input) in inputs.iter().enumerate() {
            let variant=batch_variant(base,&digest,index);
            self.histories[input.resident as usize].check_append(&variant,input.position,input.tokens.len() as u32)?;
            if self.sealed[input.resident as usize].is_some(){return Err(invalid("batch cannot mutate sealed resident"));}
            variants.push(variant);
        }
        let ticket=if let Some(shared)=&mut self.shared {
            let leases=inputs.iter().map(|input| {
                let slot=input.resident as usize;
                if input.position+input.tokens.len() as u32>shared.token_limits[slot] {
                    return Err(invalid("batch exceeds request reservation"));
                }
                shared.leases[slot].ok_or_else(||invalid("batch resident has no state reservation"))
            }).collect::<Result<Vec<_>>>()?;
            Some(shared.pool.submit(1,&leases,&[])?)
        } else {None};
        let expected=inputs.iter().filter(|x|x.sample).count();let mut output=vec![0u32;expected];let mut count=0;
        let status=unsafe{call(self.ptr,descriptors.as_ptr(),descriptors.len() as u32,tokens.as_ptr(),tokens.len() as u32,
            output.as_mut_ptr(),output.len() as u32,&mut count)};
        self.finish_invocation(ticket,status)?;
        if count as usize!=expected {
            if let Some(shared)=&mut self.shared{shared.pool.poison();}
            return Err(invalid("native batch output count mismatch"));
        }
        let mut sampled=output.into_iter();let mut result=Vec::new();
        for (input,variant) in inputs.iter().zip(&variants) {
            self.histories[input.resident as usize].append_completed(variant,input.position,input.tokens.len() as u32)?;
            result.push(if input.sample{sampled.next()}else{None});
        }
        Ok(result)
    }
    pub fn batch_capacity(&self)->Result<u32> {
        let Some(call)=(unsafe{self._library.get::<Capacity>(b"inferfabric_program_batch_capacity\0").ok()}) else{return Ok(0)};
        let mut capacity=0;self.check(unsafe{call(self.ptr,&mut capacity)})?;
        if capacity>self.prefill_tokens || (capacity>0 && self.batch_fn.is_none()) {
            return Err(invalid("native batch capacity disagrees with executable"));
        }
        Ok(capacity)
    }
    pub fn arithmetic_history(&self, slot:u32) -> Result<&ArithmeticHistory> {
        self.histories.get(slot as usize).ok_or_else(||invalid("resident history bounds"))
    }
    pub fn prefill_capacity(&self) -> u32 {
        self.prefill_tokens
    }
    pub fn uncaptured_prefill(&self) -> bool { self.uncaptured_prefill }
    pub fn prefill_chunk(&self, limit: u32) -> u32 {
        if self.uncaptured_prefill {return limit.min(self.prefill_tokens);}
        self.prefill_variants.iter().rev().copied().find(|&n|n<=limit).unwrap_or(1)
    }
    pub fn prefill(&mut self, tokens: &[u32], position: u32, sample: bool) -> Result<Option<u32>> {
        let count=u32::try_from(tokens.len()).map_err(|_|invalid("prefill count overflow"))?;
        if count<=1 || (!self.uncaptured_prefill && !self.prefill_variants.contains(&count)) || count>self.prefill_tokens {
            return Err(invalid("prefill must match the compiled chunk width"));
        }
        let variant=self.variants.iter().position(|v|v.phase==Phase::Prefill && v.accepts_query_tokens(count)).ok_or_else(||invalid("missing arithmetic variant"))?;
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
    fn prospective_single_chunks_match_completed_batches_without_relaxing_peer_identity() {
        let m:Manifest=serde_json::from_str(include_str!("../../../docs/validation/2026-10-17-persistent-serving/serving.json")).unwrap();
        let mut base=m.variants[0].clone();
        base.phase=Phase::Prefill;base.packed=true;base.graph_capable=false;
        base.max_tokens=4;base.max_query_tokens=4;base.max_context=32;
        let variants=vec![base.clone()];let prompt=[1,2,3,4,5,6,7,8];
        let mut prospective=ArithmeticHistory::default();let mut completed=ArithmeticHistory::default();
        for position in [0u32,4] {
            let tokens=&prompt[position as usize..position as usize+4];
            let predicted=prospective_single_batch_variant(&variants,tokens,position,4).unwrap();
            let actual=[BatchInput{resident:2,position,tokens:tokens.to_vec(),sample:true}];
            let (_,_,digest)=pack_batch(&actual,3,4).unwrap();
            prospective.append_completed(&predicted,position,4).unwrap();
            completed.append_completed(&batch_variant(&base,&digest,0),position,4).unwrap();
        }
        assert_eq!(prospective,completed);
        let isolated=prospective_single_batch_variant(&variants,&prompt[..2],0,4).unwrap();
        let peers=[BatchInput{resident:0,position:0,tokens:prompt[..2].to_vec(),sample:false},
            BatchInput{resident:1,position:0,tokens:vec![9,10],sample:false}];
        let (_,_,digest)=pack_batch(&peers,2,4).unwrap();
        assert_ne!(isolated.numerical_class,batch_variant(&base,&digest,0).numerical_class);
        assert_ne!(isolated.numerical_class,prospective_single_batch_variant(&variants,&[1,3],0,4).unwrap().numerical_class);
        assert_ne!(isolated.numerical_class,prospective_single_batch_variant(&variants,&prompt[..2],1,4).unwrap().numerical_class);
        for (tokens,position,budget) in [(&prompt[..0],0,4),(&prompt[..4],0,3),(&prompt[..4],31,4),(&prompt[..4],u32::MAX,4)] {
            assert!(prospective_single_batch_variant(&variants,tokens,position,budget).is_err());
        }
        base.graph_capable=true;
        assert!(prospective_single_batch_variant(&[base],&prompt[..2],0,4).is_err());
    }
    #[test]
    fn packed_batch_preserves_members_and_rejects_aliasing() {
        let inputs=vec![BatchInput{resident:2,position:63,tokens:vec![9],sample:true},
            BatchInput{resident:0,position:0,tokens:vec![7,8,9],sample:false}];
        let (descriptors,tokens,digest)=pack_batch(&inputs,3,4).unwrap();
        assert_eq!(std::mem::size_of::<NativeSequence>(),16);
        assert_eq!(tokens,vec![9,7,8,9]);
        assert_eq!((descriptors[0].resident,descriptors[0].position,descriptors[0].token_count,descriptors[0].sample),(2,63,1,1));
        assert_eq!((descriptors[1].resident,descriptors[1].token_count,descriptors[1].sample),(0,3,0));
        assert!(pack_batch(&inputs,3,3).is_err());
        assert!(pack_batch(&inputs,2,4).is_err());
        assert!(pack_batch(&[],3,4).is_err());
        let mut bad=inputs.clone();bad[1].resident=2;assert!(pack_batch(&bad,3,4).is_err());
        bad=inputs.clone();bad[1].tokens.clear();assert!(pack_batch(&bad,3,4).is_err());
        bad=inputs.clone();bad[0].position=u32::MAX;assert!(pack_batch(&bad,3,4).is_err());
        let mut changed=inputs.clone();changed[1].tokens[0]=6;
        assert_ne!(digest,pack_batch(&changed,3,4).unwrap().2);
        changed=inputs.clone();changed[1].position=10;
        assert_ne!(digest,pack_batch(&changed,3,4).unwrap().2);
        changed=inputs.clone();changed.swap(0,1);
        assert_ne!(digest,pack_batch(&changed,3,4).unwrap().2);
        // Physical slot reassignment alone does not change arithmetic.
        changed=inputs;changed[0].resident=1;
        assert_eq!(digest,pack_batch(&changed,3,4).unwrap().2);
    }
    #[test]
    fn uncaptured_envelope_cannot_masquerade_as_captured_variants() {
        let mut m:Manifest=serde_json::from_str(include_str!("../../../docs/validation/2026-10-17-persistent-serving/serving.json")).unwrap();
        m.states.clear();
        m.variants[0].max_context=65536;
        let mut v=m.variants[0].clone();
        v.id="uncaptured_prefill_4096_v1".into();v.numerical_class=v.id.clone();
        v.phase=Phase::Prefill;v.max_tokens=4096;v.max_query_tokens=4096;
        v.packed=true;v.graph_capable=false;m.variants.push(v);
        assert_eq!(prefill_capacity(&m).unwrap(),4096);
        let valid=m.clone();
        m.variants[1].graph_capable=true;
        assert!(prefill_capacity(&m).is_err());
        m=valid.clone();m.variants[1].max_tokens=8192;m.variants[1].max_query_tokens=8192;
        assert!(prefill_capacity(&m).is_err());
        m=valid.clone();m.variants[1].numerical_class="packed_prefill_4096_v1".into();
        assert!(prefill_capacity(&m).is_err());
        m=valid.clone();m.variants.push(valid.variants[1].clone());
        assert!(prefill_capacity(&m).is_err());
    }
    #[test]
    fn copy_preflight_preserves_completed_history_and_rejects_invalid_destinations() {
        let m:Manifest=serde_json::from_str(include_str!("../../../docs/validation/2026-10-17-persistent-serving/serving.json")).unwrap();
        let mut history=ArithmeticHistory::default();
        for position in 0..64 {history.append_completed(&m.variants[0],position,1).unwrap();}
        let histories=vec![history.clone(),ArithmeticHistory::default(),history.clone()];
        assert_eq!(copy_history(&histories,0,1,64,&[16,32,64]).unwrap(),history);
        for (source,destination,boundary) in [(0,0,64),(0,2,64),(0,1,0),(0,1,32),(0,1,128),(3,1,64),(0,3,64)] {
            assert!(copy_history(&histories,source,destination,boundary,&[16,32,64]).is_err());
        }
        assert!(copy_history(&histories,0,1,64,&[0]).is_err());
        assert!(copy_history(&histories,0,1,64,&[128]).is_err());
        assert_eq!(histories[0],history);
        assert_eq!(histories[1].tokens(),0);
    }
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
        for count in [8,16,32] {
            let mut v=m.variants[1].clone();
            v.id=format!("packed_prefill_{count}_v1");v.numerical_class=v.id.clone();
            v.max_tokens=count;v.max_query_tokens=count;m.variants.push(v);
        }
        assert_eq!(prefill_capacity(&m).unwrap(),32);
        let mut full_family=m.clone();
        let mut two=full_family.variants[1].clone();
        two.id="packed_prefill_2_v1".into();two.numerical_class=two.id.clone();
        two.max_tokens=2;two.max_query_tokens=2;full_family.variants.insert(1,two);
        assert_eq!(prefill_capacity(&full_family).unwrap(),32);
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
