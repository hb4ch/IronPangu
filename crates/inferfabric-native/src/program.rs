//! Persistent executable-program ABI. The handle stays on its owning worker.
use inferfabric_ir::serving::{Manifest, Phase};
use inferfabric_model::{Result, invalid};
use libloading::Library;
use sha2::{Digest, Sha256};
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
/// Only the two concrete native invocation contracts are accepted, not arbitrary batches.
pub fn prefill_capacity(m: &Manifest) -> Result<u32> {
    m.validate()?;
    if m.variants.is_empty()
        || m.variants.len() > 2
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
    if let Some(v) = m.variants.get(1) {
        let count = v.max_tokens;
        if !matches!(count, 2 | 4 | 8 | 16)
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
        Ok(count)
    } else {
        Ok(1)
    }
}
pub struct Engine {
    _library: Library,
    ptr: *mut c_void,
    close: Close,
    reset_fn: Close,
    token_fn: Token,
    prefill_fn: Option<Prefill>,
    prefill_tokens: u32,
    error: Error,
    bind_pages_fn: Option<BindPages>,
    select_resident_fn: Option<Resident>,
    reset_resident_fn: Option<Resident>,
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
        self.check(unsafe { (self.reset_fn)(self.ptr) })
    }
    pub fn select_resident(&mut self, slot: u32) -> Result<()> {
        if slot == 0 && self.select_resident_fn.is_none() {
            return Ok(());
        }
        let select = self
            .select_resident_fn
            .ok_or_else(|| invalid("backend does not support resident selection"))?;
        self.check(unsafe { select(self.ptr, slot) })
    }
    pub fn reset_resident(&mut self, slot: u32) -> Result<()> {
        if slot == 0 && self.reset_resident_fn.is_none() {
            return self.reset();
        }
        let reset = self
            .reset_resident_fn
            .ok_or_else(|| invalid("backend does not support resident reset"))?;
        self.check(unsafe { reset(self.ptr, slot) })
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
        let mut output = 0;
        self.check(unsafe {
            (self.token_fn)(self.ptr, token, position, sample.into(), &mut output)
        })?;
        Ok(sample.then_some(output))
    }
    pub fn prefill_capacity(&self) -> u32 {
        self.prefill_tokens
    }
    pub fn prefill(&mut self, tokens: &[u32], position: u32, sample: bool) -> Result<Option<u32>> {
        if tokens.len() != self.prefill_tokens as usize || tokens.len() <= 1 {
            return Err(invalid("prefill must match the compiled chunk width"));
        }
        let prefill = self
            .prefill_fn
            .ok_or_else(|| invalid("backend has no packed prefill entry point"))?;
        let mut output = 0;
        self.check(unsafe {
            prefill(
                self.ptr,
                tokens.as_ptr(),
                self.prefill_tokens,
                position,
                sample.into(),
                &mut output,
            )
        })?;
        Ok(sample.then_some(output))
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
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
