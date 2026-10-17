//! Persistent executable-program ABI. The handle stays on its owning worker.
use inferfabric_ir::serving::Manifest;
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
type Error = unsafe extern "C" fn() -> *const c_char;
type BindPages = unsafe extern "C" fn(*mut c_void, u32, *const c_char, *const i32, u32) -> i32;
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
    // ABI v1 is the qualified serial token implementation, not a general batch ABI.
    if m.variants.len() != 1
        || m.variants[0].id != "serial_token_v1"
        || m.variants[0].max_sequences != 1
        || m.variants[0].max_tokens != 1
    {
        return Err(invalid("unsupported program engine serving envelope"));
    }
    Ok(m)
}
pub struct Engine {
    _library: Library,
    ptr: *mut c_void,
    close: Close,
    reset_fn: Close,
    token_fn: Token,
    error: Error,
    bind_pages_fn: Option<BindPages>,
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
                error: symbol!("inferfabric_program_error", Error),
                bind_pages_fn: lib.get::<BindPages>(b"inferfabric_program_bind_pages\0").ok().map(|f| *f),
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
    /// Update device page metadata after reset; no cache content or graph address moves.
    pub fn bind_pages(&mut self, rank: u32, component: &str, pages: &[u32]) -> Result<()> {
        let bind = self.bind_pages_fn.ok_or_else(|| invalid("backend does not support page bindings"))?;
        let component = CString::new(component).map_err(|_| invalid("component contains NUL"))?;
        let pages = pages.iter().map(|&p| i32::try_from(p).map_err(|_| invalid("page ID overflow"))).collect::<Result<Vec<_>>>()?;
        let count = u32::try_from(pages.len()).map_err(|_| invalid("page table length overflow"))?;
        self.check(unsafe { bind(self.ptr, rank, component.as_ptr(), pages.as_ptr(), count) })
    }
    pub fn token(&mut self, token: u32, position: u32, sample: bool) -> Result<Option<u32>> {
        let mut output = 0;
        self.check(unsafe {
            (self.token_fn)(self.ptr, token, position, sample.into(), &mut output)
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
