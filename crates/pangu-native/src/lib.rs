//! Experimental one-request native engine. Devices and all resources stay on one thread.
mod ffi;
use ffi::{Api, Session, SessionPtr};
use pangu_compiler::{
    bound::Plan,
    checkpoint::{Checkpoint, Dtype, sha},
    lower::{Binding, Transform},
};
use pangu_model::{Result, invalid};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type Embedding = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Block = unsafe extern "C" fn(
    SessionPtr,
    i32,
    *const u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i64,
    i64,
    *mut u64,
    *mut u64,
    *mut u64,
) -> i32;
type Sequence = unsafe extern "C" fn(SessionPtr, *const u64, u64, *mut u64) -> i32;
struct Region {
    handle: u64,
    bytes: usize,
}
fn write(s: &Session<'_>, id: u64, data: &[u8]) -> Result<()> {
    s.api
        .check(unsafe { (s.api.write)(s.ptr, id, 0, data.as_ptr().cast(), data.len() as u64) })
}
fn read(s: &Session<'_>, r: &Region) -> Result<Vec<u8>> {
    let mut data = vec![0; r.bytes];
    s.api.check(unsafe {
        (s.api.read)(s.ptr, r.handle, 0, data.as_mut_ptr().cast(), r.bytes as u64)
    })?;
    Ok(data)
}
fn float(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}
fn upload(
    s: &Session<'_>,
    dir: &Path,
    c: &Checkpoint,
    b: &Binding,
    cache: &mut BTreeMap<(String, bool), u64>,
    hashes: &mut BTreeMap<String, String>,
) -> Result<u64> {
    let transform = b.transform == Transform::OnePlusFp32;
    let key = (b.name.clone(), transform);
    if let Some(&id) = cache.get(&key) {
        return Ok(id);
    }
    let w = c
        .weights
        .get(&b.name)
        .ok_or_else(|| invalid("missing native weight"))?;
    let root = dir.canonicalize()?;
    let path = root.join(&w.shard).canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        return Err(invalid("native shard escapes checkpoint"));
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(w.offset))?;
    let mut bytes = vec![0; w.bytes as usize];
    file.read_exact(&mut bytes)?;
    hashes.insert(b.name.clone(), sha(&bytes));
    if transform {
        if w.dtype != Dtype::BF16 || w.shape.len() != 1 {
            return Err(invalid("invalid norm weight transformation"));
        }
        bytes = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|x| (1. + float(u16::from_le_bytes(*x))).to_le_bytes())
            .collect();
    }
    let id = s.allocate(bytes.len())?;
    write(s, id, &bytes)?;
    cache.insert(key, id);
    Ok(id)
}
/// The closure runs on the caller's device thread and cannot move the engine elsewhere.
pub fn with_engine<R>(
    plan: &Plan,
    dir: &Path,
    library: &Path,
    device: i32,
    context: usize,
    run: impl FnOnce(&mut Engine<'_>) -> Result<R>,
) -> Result<R> {
    let api = Api::load(library)?;
    let mut engine = Engine::prepare(&api, plan, dir, device, context)?;
    run(&mut engine)
}
pub struct Engine<'a> {
    session: Session<'a>,
    states: Vec<Region>,
    ids: u64,
    positions: u64,
    append: u64,
    mask: u64,
    logits: u64,
    graph: u64,
    context: usize,
    frontier: usize,
    mask_values: Vec<f32>,
    pub program_key: String,
    pub weight_digest: String,
}
impl<'a> Engine<'a> {
    fn prepare(api: &'a Api, plan: &Plan, dir: &Path, device: i32, context: usize) -> Result<Self> {
        let program = pangu_compiler::lower::lower(plan, context)?;
        let s = Session::open(api, device)?;
        unsafe {
            let embed: Embedding = *api
                ._library
                .get(b"pangu_acl_embedding_prepare\0")
                .map_err(|e| invalid(e.to_string()))?;
            let block: Block = *api
                ._library
                .get(b"pangu_acl_model_layer_prepare\0")
                .map_err(|e| invalid(e.to_string()))?;
            let sequence: Sequence = *api
                ._library
                .get(b"pangu_acl_sequence_prepare\0")
                .map_err(|e| invalid(e.to_string()))?;
            let mut cache = BTreeMap::new();
            let mut hashes = BTreeMap::new();
            let c = &plan.checkpoint;
            let embedding = upload(
                &s,
                dir,
                c,
                &Binding {
                    name: program.embedding.clone(),
                    transform: Transform::Identity,
                },
                &mut cache,
                &mut hashes,
            )?;
            let ids = s.allocate(8)?;
            let positions = s.allocate(8)?;
            let append = s.allocate(8)?;
            let table = s.allocate(context * 8)?;
            let mask = s.allocate(context * 4)?;
            let mut hidden = s.allocate(4096)?;
            let mut ops = vec![];
            let mut op = 0;
            api.check(embed(s.ptr, embedding, ids, hidden, 248320, 2048, &mut op))?;
            ops.push(op);
            let slots = context.div_ceil(program.page_tokens) * program.page_tokens;
            let mut states = vec![];
            for item in &program.blocks {
                let mut bindings = vec![];
                for b in &item.bindings {
                    bindings.push(upload(&s, dir, c, b, &mut cache, &mut hashes)?);
                }
                let output = s.allocate(4096)?;
                let mut a = 0;
                let mut b = 0;
                let mut op = 0;
                let full = item.kind == pangu_model::Layer::Full;
                api.check(block(
                    s.ptr,
                    i32::from(full),
                    bindings.as_ptr(),
                    bindings.len() as u64,
                    hidden,
                    output,
                    positions,
                    append,
                    table,
                    mask,
                    slots as i64,
                    context as i64,
                    &mut a,
                    &mut b,
                    &mut op,
                ))?;
                states.push(Region {
                    handle: a,
                    bytes: if full { slots * 1024 } else { 6144 * 3 * 2 },
                });
                states.push(Region {
                    handle: b,
                    bytes: if full {
                        slots * 1024
                    } else {
                        16 * 128 * 128 * 4
                    },
                });
                ops.push(op);
                hidden = output;
            }
            let gamma = upload(&s, dir, c, &program.final_norm, &mut cache, &mut hashes)?;
            let normalized = s.allocate(4096)?;
            let logits = s.allocate(248320 * 2)?;
            let mut norm = 0;
            api.check((api.norm_prepare)(
                s.ptr, hidden, gamma, normalized, 1, 2048, &mut norm,
            ))?;
            ops.push(norm);
            let mut head = 0;
            api.check((api.prepare)(
                s.ptr, normalized, embedding, logits, 1, 248320, 2048, &mut head,
            ))?;
            ops.push(head);
            let mut root = 0;
            api.check(sequence(s.ptr, ops.as_ptr(), ops.len() as u64, &mut root))?;
            if hashes.len() != plan.checkpoint.weights.len() {
                return Err(invalid("incomplete native weight binding"));
            }
            let digest = sha(hashes
                .iter()
                .map(|(k, v)| format!("{k}:{v}\n"))
                .collect::<String>()
                .as_bytes());
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            write(&s, ids, &1i64.to_le_bytes())?;
            write(&s, positions, &0i64.to_le_bytes())?;
            write(&s, append, &0i64.to_le_bytes())?;
            write(
                &s,
                table,
                &(0..context)
                    .flat_map(|i| (i as i64).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut mask_values = vec![f32::NEG_INFINITY; context];
            mask_values[0] = 0.;
            write(
                &s,
                mask,
                &mask_values
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut graph = 0;
            api.check((api.capture)(s.ptr, root, &mut graph))?;
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            api.check((api.execute)(s.ptr, root))?;
            let eager = s.read(logits, 248320)?;
            let expected = states
                .iter()
                .map(|r| read(&s, r))
                .collect::<Result<Vec<_>>>()?;
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            api.check((api.replay)(s.ptr, graph))?;
            if s.read(logits, 248320)? != eager {
                return Err(invalid("native startup graph logits mismatch"));
            }
            for (r, expected) in states.iter().zip(&expected) {
                if read(&s, r)? != *expected {
                    return Err(invalid("native startup graph state mismatch"));
                }
            }
            let mut engine = Self {
                session: s,
                states,
                ids,
                positions,
                append,
                mask,
                logits,
                graph,
                context,
                frontier: 0,
                mask_values,
                program_key: program.key,
                weight_digest: digest,
            };
            engine.reset()?;
            Ok(engine)
        }
    }
    pub fn context(&self) -> usize {
        self.context
    }
    pub fn reset(&mut self) -> Result<()> {
        for r in &self.states {
            write(&self.session, r.handle, &vec![0; r.bytes])?;
        }
        self.frontier = 0;
        self.mask_values.fill(f32::NEG_INFINITY);
        Ok(())
    }
    /// Consume one token and return the greedy next-token ID. No compile/capture/allocation on device.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        if token >= 248320 || self.frontier >= self.context {
            return Err(invalid("native token or context limit"));
        }
        write(&self.session, self.ids, &(token as i64).to_le_bytes())?;
        write(
            &self.session,
            self.positions,
            &(self.frontier as i64).to_le_bytes(),
        )?;
        write(
            &self.session,
            self.append,
            &(self.frontier as i64).to_le_bytes(),
        )?;
        self.mask_values[self.frontier] = 0.;
        write(
            &self.session,
            self.mask,
            &self
                .mask_values
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.session
            .api
            .check(unsafe { (self.session.api.replay)(self.session.ptr, self.graph) })?;
        let values = self.session.read(self.logits, 248320)?;
        let mut best = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for (id, value) in values.into_iter().enumerate() {
            let logit = float(value);
            if !logit.is_finite() {
                return Err(invalid("nonfinite native logits"));
            }
            if logit > best_logit {
                best = id as u32;
                best_logit = logit;
            }
        }
        self.frontier += 1;
        Ok(best)
    }
}
