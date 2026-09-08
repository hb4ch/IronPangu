mod cpu_model;
mod model;
pub use model::run_model;
mod attention;
pub use attention::run_attention;
mod layer;
pub use layer::run_layer;
mod convolution;
pub use convolution::run_conv;
mod recurrent;
pub use recurrent::run_delta;
// Explicit hardware qualification command, separate from mock runtime/backend selection.
use inferfabric_compiler::checkpoint::{Checkpoint, Dtype, PREFIX, sha};
use inferfabric_model::{Result, invalid};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    fs::File,
    io::{Read, Seek, SeekFrom},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

type SessionPtr = *mut c_void;
type Open = unsafe extern "C" fn(i32, *mut SessionPtr) -> i32;
type Close = unsafe extern "C" fn(SessionPtr) -> i32;
type Allocate = unsafe extern "C" fn(SessionPtr, u64, *mut u64) -> i32;
type Write = unsafe extern "C" fn(SessionPtr, u64, u64, *const c_void, u64) -> i32;
type Readback = unsafe extern "C" fn(SessionPtr, u64, u64, *mut c_void, u64) -> i32;
type Prepare = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, i64, *mut u64) -> i32;
type NormPrepare = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Execute = unsafe extern "C" fn(SessionPtr, u64) -> i32;
type Capture = unsafe extern "C" fn(SessionPtr, u64, *mut u64) -> i32;
struct Api {
    _library: Library,
    open: Open,
    close: Close,
    allocate: Allocate,
    write: Write,
    read: Readback,
    prepare: Prepare,
    norm_prepare: NormPrepare,
    execute: Execute,
    capture: Capture,
    replay: Execute,
    error: unsafe extern "C" fn() -> *const c_char,
}
impl Api {
    fn load(path: &Path) -> Result<Self> {
        // The explicit command loads the user-selected native ABI, never an implicit search path.
        unsafe {
            let lib = Library::new(path.canonicalize()?).map_err(|e| invalid(e.to_string()))?;
            let version = lib
                .get::<unsafe extern "C" fn() -> u32>(b"inferfabric_acl_abi_version\0")
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
                open: get!("inferfabric_acl_open", Open),
                close: get!("inferfabric_acl_close", Close),
                allocate: get!("inferfabric_acl_allocate", Allocate),
                write: get!("inferfabric_acl_write", Write),
                read: get!("inferfabric_acl_read", Readback),
                prepare: get!("inferfabric_acl_linear_prepare", Prepare),
                norm_prepare: get!("inferfabric_acl_rms_prepare", NormPrepare),
                execute: get!("inferfabric_acl_operation_execute", Execute),
                capture: get!("inferfabric_acl_operation_capture", Capture),
                replay: get!("inferfabric_acl_replay", Execute),
                error: get!(
                    "inferfabric_acl_last_error",
                    unsafe extern "C" fn() -> *const c_char
                ),
                _library: lib,
            })
        }
    }
    fn check(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // Native error is a thread-local, NUL-terminated string valid until the next ABI call.
        let message = unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy();
        Err(invalid(format!("native: {message}")))
    }
}
struct Session<'a> {
    api: &'a Api,
    ptr: SessionPtr,
    _thread: PhantomData<Rc<()>>,
}
impl<'a> Session<'a> {
    fn open(api: &'a Api, device: i32) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        api.check(unsafe { (api.open)(device, &mut ptr) })?;
        Ok(Self {
            api,
            ptr,
            _thread: PhantomData,
        })
    }
    fn allocate(&self, bytes: usize) -> Result<u64> {
        let mut id = 0;
        self.api
            .check(unsafe { (self.api.allocate)(self.ptr, bytes as u64, &mut id) })?;
        Ok(id)
    }
    fn write(&self, id: u64, data: &[u16]) -> Result<()> {
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
    fn read(&self, id: u64, count: usize) -> Result<Vec<u16>> {
        let mut out = vec![0; count];
        self.api.check(unsafe {
            (self.api.read)(self.ptr, id, 0, out.as_mut_ptr().cast(), (count * 2) as u64)
        })?;
        Ok(out)
    }
    fn close(mut self) -> Result<()> {
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
fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn float(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
fn load_weight(root: &Path, c: &Checkpoint, key: &str) -> Result<(Vec<u16>, String, usize, usize)> {
    let w = &c.weights[key];
    if w.dtype != Dtype::BF16 || !(1..=3).contains(&w.shape.len()) {
        return Err(invalid("probe requires BF16 weight with rank 1 to 3"));
    }
    let root = root.canonicalize()?;
    let path = root.join(&w.shard).canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        return Err(invalid("weight shard escaped checkpoint"));
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(w.offset))?;
    let mut bytes = vec![0; usize::try_from(w.bytes).map_err(|_| invalid("weight too large"))?];
    file.read_exact(&mut bytes)?;
    let digest = sha(&bytes);
    let values = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok((values, digest, w.shape[0], *w.shape.last().unwrap()))
}
fn reference(x: &[u16], w: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
    let x: Vec<_> = x.iter().copied().map(float).collect();
    let w: Vec<_> = w.iter().copied().map(float).collect();
    let mut out = vec![0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0f64;
            // FP64 reference reduces accumulation-order noise; compare after BF16 output rounding.
            for j in 0..k {
                sum += (x[row * k + j] as f64) * (w[col * k + j] as f64);
            }
            out[row * n + col] = bf16(sum as f32);
        }
    }
    out
}
fn compare(actual: &[u16], expected: &[u16]) -> Result<(f32, usize)> {
    let mut max_error = 0f32;
    let mut different = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        let a = float(a);
        let e = float(e);
        let err = (a - e).abs();
        if !a.is_finite() || !e.is_finite() || err > 0.005 + 0.01 * e.abs() {
            return Err(invalid(format!(
                "linear numerical mismatch: actual={a} reference={e} error={err}"
            )));
        }
        max_error = max_error.max(err);
        different += usize::from(a != e);
    }
    Ok((max_error, different))
}
pub fn run(
    dsl: &Path,
    checkpoint_dir: &Path,
    library: &Path,
    device: i32,
) -> Result<serde_json::Value> {
    if device < 0 {
        return Err(invalid("device ID must be nonnegative"));
    }
    let spec = inferfabric_dsl::parse_checkpoint(&std::fs::read_to_string(dsl)?)?;
    let checkpoint = inferfabric_compiler::checkpoint::inspect(checkpoint_dir)?;
    let plan = inferfabric_compiler::bound::compile(&spec, checkpoint)?;
    let api = Api::load(library)?;
    let mut reports = vec![];
    for suffix in [
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.3.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ] {
        let key = format!("{PREFIX}{suffix}");
        let (weights, digest, n, k) = load_weight(checkpoint_dir, &plan.checkpoint, &key)?;
        for m in [1usize, 4] {
            let session = Session::open(&api, device)?;
            let x = session.allocate(m * k * 2)?;
            let w = session.allocate(weights.len() * 2)?;
            let y = session.allocate(m * n * 2)?;
            session.write(w, &weights)?;
            let mut operation = 0;
            api.check(unsafe {
                (api.prepare)(
                    session.ptr,
                    x,
                    w,
                    y,
                    m as i64,
                    n as i64,
                    k as i64,
                    &mut operation,
                )
            })?;
            let mut graph = 0;
            let mut worst = 0f32;
            let mut differences = 0;
            for iteration in 0..3 {
                let input: Vec<_> = (0..m * k)
                    .map(|i| bf16((((i * 17 + iteration * 11) % 97) as f32 - 48.) / 128.))
                    .collect();
                session.write(x, &input)?;
                api.check(unsafe { (api.execute)(session.ptr, operation) })?;
                let eager = session.read(y, m * n)?;
                let expected = reference(&input, &weights, m, n, k);
                let (error, diff) = compare(&eager, &expected)?;
                worst = worst.max(error);
                differences += diff;
                if iteration == 0 {
                    api.check(unsafe { (api.capture)(session.ptr, operation, &mut graph) })?;
                }
                // Poison output; replay must overwrite it using the current input values.
                session.write(y, &vec![bf16(-123.); m * n])?;
                api.check(unsafe { (api.replay)(session.ptr, graph) })?;
                let replay = session.read(y, m * n)?;
                if replay != eager {
                    return Err(invalid("graph replay differs from eager linear"));
                }
            }
            session.close()?;
            reports.push(serde_json::json!({"operation":"linear","weight":key,"weight_sha256":digest,"m":m,"n":n,"k":k,"iterations":3,"max_absolute_error":worst,"bf16_values_differing_from_reference":differences,"eager_reference":"passed","graph_eager":"bitwise_equal","input_mutation":"passed"}));
            eprintln!("device={device} linear={suffix} m={m} passed");
        }
    }

    for suffix in [
        "layers.0.input_layernorm.weight",
        "layers.3.self_attn.q_norm.weight",
        "norm.weight",
    ] {
        let key = format!("{PREFIX}{suffix}");
        let (weights, digest, _, k) = load_weight(checkpoint_dir, &plan.checkpoint, &key)?;
        let gamma: Vec<f32> = weights.iter().map(|&w| 1.0 + float(w)).collect();
        for m in [1usize, 4] {
            let session = Session::open(&api, device)?;
            let x = session.allocate(m * k * 2)?;
            let g = session.allocate(k * 4)?;
            let y = session.allocate(m * k * 2)?;
            api.check(unsafe {
                (api.write)(session.ptr, g, 0, gamma.as_ptr().cast(), (k * 4) as u64)
            })?;
            let mut operation = 0;
            api.check(unsafe {
                (api.norm_prepare)(session.ptr, x, g, y, m as i64, k as i64, &mut operation)
            })?;
            let mut graph = 0;
            let mut worst = 0f32;
            let mut differences = 0;
            for iteration in 0..3 {
                let input: Vec<_> = (0..m * k)
                    .map(|i| {
                        bf16(
                            (((i * 17 + iteration * 11) % 97) as f32 - 48.)
                                * [0.0, 0.00001, 0.0078125][iteration],
                        )
                    })
                    .collect();
                session.write(x, &input)?;
                api.check(unsafe { (api.execute)(session.ptr, operation) })?;
                let eager = session.read(y, m * k)?;
                let mut expected = vec![0u16; m * k];
                for row in 0..m {
                    let mean = input[row * k..(row + 1) * k]
                        .iter()
                        .map(|&v| (float(v) as f64).powi(2))
                        .sum::<f64>()
                        / k as f64;
                    let inv = (mean + 1e-6).sqrt().recip() as f32;
                    for j in 0..k {
                        expected[row * k + j] = bf16((float(input[row * k + j]) * inv) * gamma[j]);
                    }
                }
                let (error, diff) = compare(&eager, &expected)?;
                worst = worst.max(error);
                differences += diff;
                if iteration == 0 {
                    api.check(unsafe { (api.capture)(session.ptr, operation, &mut graph) })?;
                }
                session.write(y, &vec![bf16(-123.); m * k])?;
                api.check(unsafe { (api.replay)(session.ptr, graph) })?;
                if session.read(y, m * k)? != eager {
                    return Err(invalid("RMS graph replay differs from eager"));
                }
            }
            session.close()?;
            reports.push(serde_json::json!({"operation":"zero_centered_rms","weight":key,"weight_sha256":digest,"m":m,"k":k,"iterations":3,"max_absolute_error":worst,"bf16_values_differing_from_reference":differences,"eager_reference":"passed","graph_eager":"bitwise_equal","input_mutation":"passed"}));
            eprintln!("device={device} rms={suffix} m={m} passed");
        }
    }
    Ok(
        serde_json::json!({"device":device,"abi":2,"unix_time":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| invalid(e.to_string()))?.as_secs(),"native_library_sha256":sha(&std::fs::read(library)?),"rust_probe_sha256":sha(&std::fs::read(std::env::current_exe()?)?),"plan_key":plan.key,"scope":"checkpoint linear and zero-centered RMS qualification, not model inference","cases":reports}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bfloat_rounding_and_transposed_weight_reference() {
        assert_eq!(float(bf16(1.0)), 1.0);
        assert_eq!(bf16(f32::from_bits(0x3f808000)), 0x3f80);
        assert_eq!(bf16(f32::from_bits(0x3f818000)), 0x3f82);
        let x = [1., 2., 3., 4.].map(bf16);
        let w = [1., 0., 0., 1., 2., 3.].map(bf16);
        assert_eq!(
            reference(&x, &w, 2, 3, 2)
                .iter()
                .copied()
                .map(float)
                .collect::<Vec<_>>(),
            [1., 2., 8., 3., 4., 18.]
        );
    }
}
