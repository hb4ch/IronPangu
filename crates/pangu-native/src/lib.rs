//! Fixed-lane native graph engine. Devices and all resources stay on one thread.
mod activation;
mod ffi;
mod memory;
pub use memory::{MemoryProfile, StartupOptions};
mod parallel;
mod sampling;
use ffi::{Api, Session, SessionPtr};
use pangu_compiler::{
    bound::Plan,
    checkpoint::{Checkpoint, Dtype, sha},
    lower::{Binding, Transform},
};
use pangu_model::{Result, invalid};
pub use parallel::{EngineSet, with_serving_engine};
pub use sampling::Sampling;
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type Embedding = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, i64, *mut u64) -> i32;
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
    i64,
    u64,
    *mut u64,
    *mut u64,
    *mut u64,
) -> i32;
type Sequence = unsafe extern "C" fn(SessionPtr, *const u64, u64, *mut u64) -> i32;
struct Region {
    kv: bool,
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
    if s.world > 1
        && w.shape.len() == 2
        && !b.name.ends_with("embed_tokens.weight")
        && !b.name.contains("conv1d")
    {
        if w.shape[0] % s.world as usize != 0 {
            return Err(invalid("weight rows not divisible by TP"));
        }
        let shard = bytes.len() / s.world as usize;
        bytes = bytes[s.rank as usize * shard..(s.rank as usize + 1) * shard].to_vec();
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
    with_engine_options(
        plan,
        dir,
        library,
        device,
        &StartupOptions {
            max_model_len: context,
            max_num_seqs: 1,
            memory_utilization: 0.9,
            profile_path: None,
        },
        run,
    )
}
/// Profile the requested context and enforce its memory budget before serving.
pub fn with_engine_options<R>(
    plan: &Plan,
    dir: &Path,
    library: &Path,
    device: i32,
    options: &StartupOptions,
    run: impl FnOnce(&mut Engine<'_>) -> Result<R>,
) -> Result<R> {
    options.validate()?;
    if plan.spec.prefill.tp != 1 {
        return Err(invalid("use coordinated serving engine for TP"));
    }
    let api = Api::load(library)?;
    let plan = pangu_compiler::lower::specialize_context(plan, options.max_model_len)?;
    let mut engine = match Engine::prepare(&api, &plan, dir, device, options, None) {
        Ok(engine) => engine,
        Err(e) => {
            memory::save_failure(options, &e.to_string());
            return Err(e);
        }
    };
    run(&mut engine)
}
pub struct Engine<'a> {
    session: Session<'a>,
    states: Vec<Region>,
    ids: u64,
    positions: u64,
    append: u64,
    mask: u64,
    samplers: Vec<sampling::Sampler>,
    sampler_logits: Vec<u64>,
    logits: u64,
    active: u64,
    batch: usize,
    slots_per_lane: usize,
    graph: u64,
    context: usize,
    frontiers: Vec<usize>,
    mask_values: Vec<f32>,
    pub memory_profile: MemoryProfile,
    pub program_key: String,
    pub weight_digest: String,
}
impl<'a> Engine<'a> {
    fn prepare(
        api: &'a Api,
        plan: &Plan,
        dir: &Path,
        device: i32,
        options: &StartupOptions,
        parallel: Option<&parallel::ParallelConfig>,
    ) -> Result<Self> {
        let context = options.max_model_len;
        let batch = options.max_num_seqs;
        let program = pangu_compiler::lower::lower_batch(plan, context, batch)?;
        let mut s = Session::open(api, device)?;
        if let Some(parallel) = parallel {
            s.init_parallel(parallel)?;
        }
        let mut profile = MemoryProfile::start(
            &s,
            options,
            plan.checkpoint.weights.values().map(|w| w.bytes).sum(),
            plan.checkpoint
                .weights
                .iter()
                .map(|(name, w)| {
                    if s.world > 1
                        && w.shape.len() == 2
                        && !name.ends_with("embed_tokens.weight")
                        && !name.contains("conv1d")
                    {
                        w.bytes / s.world as u64
                    } else {
                        w.bytes
                    }
                })
                .sum(),
        )?;
        unsafe {
            let embed: Embedding = *api
                ._library
                .get(b"pangu_acl_embedding_batch_prepare\0")
                .map_err(|e| invalid(e.to_string()))?;
            let block: Block = *api
                ._library
                .get(b"pangu_acl_model_layer_batch_prepare\0")
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
            // Upload every unique binding before operator preparation to isolate weight memory.
            for block in &program.blocks {
                for binding in &block.bindings {
                    upload(&s, dir, c, binding, &mut cache, &mut hashes)?;
                }
            }
            upload(&s, dir, c, &program.final_norm, &mut cache, &mut hashes)?;
            profile.resident_weight_bytes = profile.record(&s, "weights_loaded")?.buffer_bytes;
            let ids = s.allocate(batch * 8)?;
            let active = s.allocate(batch)?;
            write(&s, active, &vec![1; batch])?;
            let positions = s.allocate(batch * 8)?;
            let append = s.allocate(batch * 8)?;
            let table = s.allocate(batch * context * 8)?;
            let mask = s.allocate(batch * context * 4)?;
            // Values remain live through the consuming layer, including its
            // residual path. Adjacent input/output intervals must not alias.
            let hidden_handles = activation::bind_hidden(&s, program.blocks.len(), batch * 4096)?;
            let mut hidden = hidden_handles[0];
            let mut ops = vec![];
            let mut op = 0;
            api.check(embed(
                s.ptr,
                embedding,
                ids,
                hidden,
                248320,
                2048,
                batch as i64,
                &mut op,
            ))?;
            ops.push(op);
            let slots_per_lane = context.div_ceil(program.page_tokens) * program.page_tokens + 1;
            let slots = batch * slots_per_lane;
            let mut states = vec![];
            for (layer_index, item) in program.blocks.iter().enumerate() {
                let mut bindings = vec![];
                for b in &item.bindings {
                    bindings.push(upload(&s, dir, c, b, &mut cache, &mut hashes)?);
                }
                let output = hidden_handles[layer_index + 1];
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
                    batch as i64,
                    active,
                    &mut a,
                    &mut b,
                    &mut op,
                ))?;
                states.push(Region {
                    kv: full,
                    handle: a,
                    bytes: if full {
                        slots * 1024
                    } else {
                        batch * 6144 * 3 * 2
                    },
                });
                states.push(Region {
                    kv: full,
                    handle: b,
                    bytes: if full {
                        slots * 1024
                    } else {
                        batch * 16 * 128 * 128 * 4
                    },
                });
                if full {
                    profile.kv_cache_bytes += (slots * 2048) as u64;
                } else {
                    profile.recurrent_and_conv_bytes +=
                        (batch * (6144 * 3 * 2 + 16 * 128 * 128 * 4)) as u64;
                }
                ops.push(op);
                hidden = output;
            }
            let gamma = upload(&s, dir, c, &program.final_norm, &mut cache, &mut hashes)?;
            let normalized = hidden_handles[program.blocks.len() + 1];
            let logits = s.allocate(batch * 248320 * 2)?;
            let mut norm = 0;
            api.check((api.norm_prepare)(
                s.ptr,
                hidden,
                gamma,
                normalized,
                batch as i64,
                2048,
                &mut norm,
            ))?;
            ops.push(norm);
            let mut head = 0;
            api.check((api.prepare)(
                s.ptr,
                normalized,
                embedding,
                logits,
                batch as i64,
                248320,
                2048,
                &mut head,
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
            profile.record(&s, "operators_prepared")?;
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            write(
                &s,
                ids,
                &(0..batch)
                    .flat_map(|_| 1i64.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            write(&s, positions, &vec![0; batch * 8])?;
            write(
                &s,
                append,
                &(0..batch)
                    .flat_map(|lane| ((lane * slots_per_lane) as i64).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            write(
                &s,
                table,
                &(0..batch * context)
                    .flat_map(|i| {
                        ((i / context * slots_per_lane + i % context) as i64).to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )?;
            let mut mask_values = vec![f32::NEG_INFINITY; batch * context];
            for lane in 0..batch {
                mask_values[lane * context] = 0.;
            }
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
            let eager = s.read(logits, batch * 248320)?;
            let expected = states
                .iter()
                .map(|r| read(&s, r))
                .collect::<Result<Vec<_>>>()?;
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            api.check((api.replay)(s.ptr, graph))?;
            if s.read(logits, batch * 248320)? != eager {
                return Err(invalid("native startup graph logits mismatch"));
            }
            for (r, expected) in states.iter().zip(&expected) {
                if read(&s, r)? != *expected {
                    return Err(invalid("native startup graph state mismatch"));
                }
            }
            profile.startup_logits_sha256 = sha(&eager
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>());
            profile.startup_state_sha256 = expected.iter().map(|bytes| sha(bytes)).collect();
            profile.record(&s, "model_graph_qualified")?;
            let mut sampler_logits = Vec::new();
            let mut samplers = Vec::new();
            for _ in 0..batch {
                let row = s.allocate(248320 * 2)?;
                // Capture-time sampler warmup must not read uninitialized logits.
                write(&s, row, &vec![0; 248320 * 2])?;
                samplers.push(sampling::Sampler::prepare(&s, row, context)?);
                sampler_logits.push(row);
            }
            profile.record(&s, "sampler_graphs_qualified")?;
            // Exercise the actual maximum attention width and final position. Cache
            // contents are synthetic; this validates footprint/replay, not long-text quality.
            write(
                &s,
                positions,
                &(0..batch)
                    .flat_map(|_| ((context - 1) as i64).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            write(
                &s,
                append,
                &(0..batch)
                    .flat_map(|lane| ((lane * slots_per_lane + context - 1) as i64).to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            write(&s, mask, &vec![0; batch * context * 4])?;
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            api.check((api.execute)(s.ptr, root))?;
            let full_eager = s.read(logits, batch * 248320)?;
            if full_eager.iter().any(|&v| !float(v).is_finite()) {
                return Err(invalid("full-context profile produced nonfinite logits"));
            }
            for r in &states {
                write(&s, r.handle, &vec![0; r.bytes])?;
            }
            api.check((api.replay)(s.ptr, graph))?;
            if s.read(logits, batch * 248320)? != full_eager {
                return Err(invalid("full-context profile eager/graph mismatch"));
            }
            for sampler in &mut samplers {
                sampler.configure(
                    &s,
                    Sampling {
                        temperature: 1.,
                        top_k: 20,
                        ..Sampling::default()
                    },
                    42,
                    &[1],
                    &[],
                )?;
                sampler.draw(&s)?;
            }
            profile.full_context_logits_sha256 = sha(&full_eager
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>());
            profile.record(&s, "full_context_dry_run")?;
            let mut engine = Self {
                session: s,
                states,
                ids,
                positions,
                append,
                mask,
                samplers,
                sampler_logits,
                logits,
                active,
                batch,
                slots_per_lane,
                graph,
                context,
                frontiers: vec![0; batch],
                mask_values,
                memory_profile: profile,
                program_key: program.key,
                weight_digest: digest,
            };
            engine.reset()?;
            // Validate inactive lanes after nonzero state exists, not only after reset.
            if batch > 1 {
                engine.consume_batch(
                    &(0..batch)
                        .map(|lane| (lane, lane as u32 + 1))
                        .collect::<Vec<_>>(),
                )?;
                let before = engine
                    .states
                    .iter()
                    .map(|r| read(&engine.session, r))
                    .collect::<Result<Vec<_>>>()?;
                engine.consume_batch(&[(0, 42)])?;
                for (r, old) in engine.states.iter().zip(&before) {
                    let now = read(&engine.session, r)?;
                    let stride = r.bytes / batch;
                    let real = stride - if r.kv { 1024 } else { 0 };
                    for lane in 1..batch {
                        if now[lane * stride..lane * stride + real]
                            != old[lane * stride..lane * stride + real]
                        {
                            return Err(invalid("inactive graph lane state changed"));
                        }
                    }
                }
                engine
                    .memory_profile
                    .record(&engine.session, "inactive_lanes_qualified")?;
                engine.reset()?;
            }
            engine.memory_profile.finish(&engine.session)?;
            Ok(engine)
        }
    }
    pub fn context(&self) -> usize {
        self.context
    }
    pub fn batch_capacity(&self) -> usize {
        self.batch
    }
    pub fn reset(&mut self) -> Result<()> {
        for lane in 0..self.batch {
            self.reset_slot(lane)?;
        }
        Ok(())
    }
    pub fn reset_slot(&mut self, lane: usize) -> Result<()> {
        if lane >= self.batch {
            return Err(invalid("invalid native slot"));
        }
        for r in &self.states {
            let bytes = r.bytes / self.batch;
            let zeros = vec![0u8; bytes];
            self.session.api.check(unsafe {
                (self.session.api.write)(
                    self.session.ptr,
                    r.handle,
                    (lane * bytes) as u64,
                    zeros.as_ptr().cast(),
                    bytes as u64,
                )
            })?;
        }
        self.frontiers[lane] = 0;
        self.mask_values[lane * self.context..(lane + 1) * self.context].fill(f32::NEG_INFINITY);
        Ok(())
    }
    /// Each selected lane consumes one token in the same fixed-shape ACL graph.
    /// Inactive lanes write only their private dummy KV row; recurrence is preserved in graph.
    pub fn consume_batch(&mut self, tokens: &[(usize, u32)]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut active = vec![0u8; self.batch];
        let mut ids = vec![1i64; self.batch];
        let mut positions = vec![0i64; self.batch];
        let mut append: Vec<i64> = (0..self.batch)
            .map(|i| ((i + 1) * self.slots_per_lane - 1) as i64)
            .collect();
        for &(lane, token) in tokens {
            if lane >= self.batch
                || active[lane] != 0
                || token >= 248320
                || self.frontiers[lane] >= self.context
            {
                return Err(invalid("invalid, duplicate or full native slot"));
            }
            active[lane] = 1;
            ids[lane] = token as i64;
            positions[lane] = self.frontiers[lane] as i64;
            append[lane] = (lane * self.slots_per_lane + self.frontiers[lane]) as i64;
        }
        for &(lane, _) in tokens {
            self.mask_values[lane * self.context + self.frontiers[lane]] = 0.;
        }
        // An unused padded lane still needs one finite attention entry to avoid NaNs.
        let mut masks = self.mask_values.clone();
        for lane in 0..self.batch {
            if self.frontiers[lane] == 0 && active[lane] == 0 {
                masks[lane * self.context] = 0.;
            }
        }
        write(&self.session, self.active, &active)?;
        for (handle, values) in [
            (self.ids, ids),
            (self.positions, positions),
            (self.append, append),
        ] {
            write(
                &self.session,
                handle,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        write(
            &self.session,
            self.mask,
            &masks
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.session
            .api
            .check(unsafe { (self.session.api.replay)(self.session.ptr, self.graph) })?;
        for &(lane, _) in tokens {
            self.frontiers[lane] += 1;
        }
        Ok(())
    }
    pub fn consume(&mut self, token: u32) -> Result<()> {
        self.consume_batch(&[(0, token)])
    }
    pub fn configure_slot(
        &mut self,
        lane: usize,
        settings: Sampling,
        seed: u64,
        prompt: &[u32],
        stop: &[u32],
    ) -> Result<()> {
        let sampler = self
            .samplers
            .get_mut(lane)
            .ok_or_else(|| invalid("invalid sampling slot"))?;
        sampler.configure(&self.session, settings, seed, prompt, stop)
    }
    pub fn configure_sampling(
        &mut self,
        settings: Sampling,
        seed: u64,
        prompt: &[u32],
        stop: &[u32],
    ) -> Result<()> {
        self.configure_slot(0, settings, seed, prompt, stop)
    }
    pub fn sample_slot(&mut self, lane: usize) -> Result<u32> {
        if lane >= self.batch || self.frontiers[lane] == 0 {
            return Err(invalid("cannot sample before prefill"));
        }
        type CopyRegion = unsafe extern "C" fn(SessionPtr, u64, u64, u64, u64, u64) -> i32;
        let copy: CopyRegion = unsafe {
            *self
                .session
                .api
                ._library
                .get(b"pangu_acl_copy_region\0")
                .map_err(|e| invalid(e.to_string()))?
        };
        self.session.api.check(unsafe {
            copy(
                self.session.ptr,
                self.logits,
                (lane * 248320 * 2) as u64,
                self.sampler_logits[lane],
                0,
                248320 * 2,
            )
        })?;
        self.samplers[lane].draw(&self.session)
    }
    pub fn sample(&mut self) -> Result<u32> {
        self.sample_slot(0)
    }
}
