use libloading::Library;

use pangu_model::{Result, invalid};
use std::{
    ffi::{CStr, c_char, c_void},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

pub(crate) type SessionPtr = *mut c_void;
type Open = unsafe extern "C" fn(i32, *mut SessionPtr) -> i32;
type Close = unsafe extern "C" fn(SessionPtr) -> i32;
type Allocate = unsafe extern "C" fn(SessionPtr, u64, *mut u64) -> i32;
type Write = unsafe extern "C" fn(SessionPtr, u64, u64, *const c_void, u64) -> i32;
type Readback = unsafe extern "C" fn(SessionPtr, u64, u64, *mut c_void, u64) -> i32;
type Prepare = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, i64, *mut u64) -> i32;
type NormPrepare = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Execute = unsafe extern "C" fn(SessionPtr, u64) -> i32;
type Capture = unsafe extern "C" fn(SessionPtr, u64, *mut u64) -> i32;
pub(crate) struct Api {
    pub(crate) _library: Library,
    pub(crate) open: Open,
    pub(crate) close: Close,
    pub(crate) allocate: Allocate,
    pub(crate) write: Write,
    pub(crate) read: Readback,
    pub(crate) prepare: Prepare,
    pub(crate) norm_prepare: NormPrepare,
    pub(crate) execute: Execute,
    pub(crate) capture: Capture,
    pub(crate) replay: Execute,
    pub(crate) error: unsafe extern "C" fn() -> *const c_char,
}
impl Api {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        // The explicit command loads the user-selected native ABI, never an implicit search path.
        unsafe {
            let lib = Library::new(path.canonicalize()?).map_err(|e| invalid(e.to_string()))?;
            let version = lib
                .get::<unsafe extern "C" fn() -> u32>(b"pangu_acl_abi_version\0")
                .map_err(|e| invalid(e.to_string()))?;
            if version() != 2 {
                return Err(invalid("native ABI version mismatch"));
            }
            macro_rules! get {
                ($name:literal,$t:ty) => {
                    *lib.get::<$t>(concat!($name, "\0").as_bytes())
                        .map_err(|e| invalid(e.to_string()))?
                };
            }
            Ok(Self {
                open: get!("pangu_acl_open", Open),
                close: get!("pangu_acl_close", Close),
                allocate: get!("pangu_acl_allocate", Allocate),
                write: get!("pangu_acl_write", Write),
                read: get!("pangu_acl_read", Readback),
                prepare: get!("pangu_acl_linear_prepare", Prepare),
                norm_prepare: get!("pangu_acl_rms_prepare", NormPrepare),
                execute: get!("pangu_acl_operation_execute", Execute),
                capture: get!("pangu_acl_operation_capture", Capture),
                replay: get!("pangu_acl_replay", Execute),
                error: get!(
                    "pangu_acl_last_error",
                    unsafe extern "C" fn() -> *const c_char
                ),
                _library: lib,
            })
        }
    }
    pub(crate) fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // Native error is a thread-local, NUL-terminated string valid until the next ABI call.
        let message = unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy();
        Err(invalid(format!("native: {message}")))
    }
}
pub(crate) struct Session<'a> {
    pub(crate) rank: u32,
    pub(crate) world: u32,
    pub(crate) api: &'a Api,
    pub(crate) ptr: SessionPtr,
    pub(crate) _thread: PhantomData<Rc<()>>,
}
impl<'a> Session<'a> {
    pub(crate) fn open(api: &'a Api, device: i32) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        api.check(unsafe { (api.open)(device, &mut ptr) })?;
        Ok(Self {
            rank: 0,
            world: 1,
            api,
            ptr,
            _thread: PhantomData,
        })
    }
    pub(crate) fn init_parallel(&mut self, config: &crate::parallel::ParallelConfig) -> Result<()> {
        type Init = unsafe extern "C" fn(SessionPtr, u32, u32, *const c_void, u64) -> i32;
        let init: Init = unsafe {
            *self
                .api
                ._library
                .get(b"pangu_acl_tp_init\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        self.api.check(unsafe {
            init(
                self.ptr,
                config.world,
                config.rank,
                config.root.as_ptr().cast(),
                config.root.len() as u64,
            )
        })?;
        self.rank = config.rank;
        self.world = config.world;
        Ok(())
    }
    pub(crate) fn allocate(&self, bytes: usize) -> Result<u64> {
        let mut id = 0;
        self.api
            .check(unsafe { (self.api.allocate)(self.ptr, bytes as u64, &mut id) })?;
        Ok(id)
    }
    #[allow(dead_code)]
    pub(crate) fn write(&self, id: u64, data: &[u16]) -> Result<()> {
        self.api.check(unsafe {
            (self.api.write)(
                self.ptr,
                id,
                0,
                data.as_ptr().cast(),
                std::mem::size_of_val(data) as u64,
            )
        })
    }
    pub(crate) fn read(&self, id: u64, count: usize) -> Result<Vec<u16>> {
        let mut out = vec![0; count];
        self.api.check(unsafe {
            (self.api.read)(self.ptr, id, 0, out.as_mut_ptr().cast(), (count * 2) as u64)
        })?;
        Ok(out)
    }
    #[allow(dead_code)]
    pub(crate) fn close(mut self) -> Result<()> {
        let status = unsafe { (self.api.close)(self.ptr) };
        if status == 0 {
            self.ptr = std::ptr::null_mut();
        }
        self.api.check(status)
    }
}
impl Drop for Session<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let status = unsafe { (self.api.close)(self.ptr) };
            if let Err(e) = self.api.check(status) {
                eprintln!("native cleanup failed: {e}");
            }
        }
    }
}
