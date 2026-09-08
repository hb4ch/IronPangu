//! Single-node tensor parallelism. All ranks replay the same graph schedule;
//! rank zero owns request sampling and broadcasts only model input tokens.
use crate::{
    Engine, StartupOptions,
    ffi::{Api, Session},
    memory,
};
use inferfabric_compiler::bound::Plan;
use inferfabric_model::{Result, invalid};
use std::{path::Path, sync::mpsc, thread::JoinHandle};
pub(crate) struct ParallelConfig {
    pub root: Vec<u8>,
    pub rank: u32,
    pub world: u32,
}
#[derive(Clone)]
enum Command {
    Reset(usize),
    Consume(Vec<(usize, u32)>),
    Stop,
}
struct Peer {
    tx: mpsc::Sender<Command>,
    rx: mpsc::Receiver<std::result::Result<(), String>>,
    thread: Option<JoinHandle<()>>,
}
impl Drop for Peer {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
pub struct EngineSet<'a, 'b> {
    primary: &'a mut Engine<'b>,
    peers: Vec<Peer>,
}
impl<'a, 'b> std::ops::Deref for EngineSet<'a, 'b> {
    type Target = Engine<'b>;
    fn deref(&self) -> &Self::Target {
        self.primary
    }
}
impl EngineSet<'_, '_> {
    pub fn configure_slot(
        &mut self,
        lane: usize,
        settings: crate::Sampling,
        seed: u64,
        prompt: &[u32],
        stop: &[u32],
    ) -> Result<()> {
        self.primary
            .configure_slot(lane, settings, seed, prompt, stop)
    }
    pub fn sample_slot(&mut self, lane: usize) -> Result<u32> {
        self.primary.sample_slot(lane)
    }

    fn dispatch(&mut self, command: Command) -> Result<()> {
        for peer in &self.peers {
            peer.tx
                .send(command.clone())
                .map_err(|_| invalid("TP peer stopped"))?;
        }
        let primary = match command {
            Command::Reset(lane) => self.primary.reset_slot(lane),
            Command::Consume(ref tokens) => self.primary.consume_batch(tokens),
            Command::Stop => Ok(()),
        };
        for peer in &self.peers {
            peer.rx
                .recv()
                .map_err(|_| invalid("TP peer stopped"))?
                .map_err(invalid)?;
        }
        primary
    }
    pub fn reset_slot(&mut self, lane: usize) -> Result<()> {
        self.dispatch(Command::Reset(lane))
    }
    pub fn consume_batch(&mut self, tokens: &[(usize, u32)]) -> Result<()> {
        self.dispatch(Command::Consume(tokens.to_vec()))
    }
}
fn with_serving_engine_inner<R>(
    plan: &Plan,
    dir: &Path,
    library: &Path,
    device: i32,
    options: &StartupOptions,
    run: impl FnOnce(&mut EngineSet<'_, '_>) -> Result<R>,
) -> Result<R> {
    let world = plan.spec.prefill.tp as u32;
    if world == 1 {
        return crate::with_engine_options(plan, dir, library, device, options, |engine| {
            run(&mut EngineSet {
                primary: engine,
                peers: vec![],
            })
        });
    }
    options.validate()?;
    let plan = inferfabric_compiler::lower::specialize_context(plan, options.max_model_len)?;
    let api = Api::load(library)?;
    // HCCL root creation requires an initialized ACL device context.
    let _bootstrap = Session::open(&api, device)?;
    type DeviceCount = unsafe extern "C" fn(*mut u32) -> i32;
    let mut count = 0;
    unsafe {
        let query: DeviceCount = *api
            ._library
            .get(b"inferfabric_acl_device_count\0")
            .map_err(|e| invalid(e.to_string()))?;
        api.check(query(&mut count))?;
    }
    if device < 0 || device as u32 + world > count {
        return Err(invalid("TP ranks exceed visible local devices"));
    }
    type RootSize = unsafe extern "C" fn() -> u64;
    type Root = unsafe extern "C" fn(*mut std::ffi::c_void, u64) -> i32;
    let mut root = unsafe {
        let size: RootSize = *api
            ._library
            .get(b"inferfabric_acl_tp_root_size\0")
            .map_err(|e| invalid(e.to_string()))?;
        vec![0u8; size() as usize]
    };
    unsafe {
        let fill: Root = *api
            ._library
            .get(b"inferfabric_acl_tp_root\0")
            .map_err(|e| invalid(e.to_string()))?;
        api.check(fill(root.as_mut_ptr().cast(), root.len() as u64))?;
    }
    let mut peers = Vec::new();
    for rank in 1..world {
        let (tx, commands) = mpsc::channel();
        let (results, rx) = mpsc::channel();
        let plan = plan.clone();
        let dir = dir.to_owned();
        let library = library.to_owned();
        let root = root.clone();
        let mut options = options.clone();
        if let Some(path) = &mut options.profile_path {
            *path = path.with_extension(format!("rank{rank}.json"));
        }
        let thread = std::thread::Builder::new()
            .name(format!("inferfabric-tp-{rank}"))
            .spawn(move || {
                let result: Result<()> = (|| {
                    let api = Api::load(&library)?;
                    let parallel = ParallelConfig { root, rank, world };
                    let mut engine = Engine::prepare(
                        &api,
                        &plan,
                        &dir,
                        device + rank as i32,
                        &options,
                        Some(&parallel),
                    )?;
                    eprintln!(
                        "TP rank={rank} ready weights={} peak={}",
                        engine.memory_profile.resident_weight_bytes,
                        engine.memory_profile.observed_peak_bytes
                    );
                    if results.send(Ok(())).is_err() {
                        return Ok(());
                    }
                    while let Ok(command) = commands.recv() {
                        let result = match command {
                            Command::Stop => break,
                            Command::Reset(lane) => engine.reset_slot(lane),
                            Command::Consume(tokens) => engine.consume_batch(&tokens),
                        };
                        let failed = result.is_err();
                        if results.send(result.map_err(|e| e.to_string())).is_err() || failed {
                            break;
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = result {
                    memory::save_failure(&options, &format!("{e}"));
                    let _ = results.send(Err(format!("{e}")));
                }
            })?;
        peers.push(Peer {
            tx,
            rx,
            thread: Some(thread),
        });
    }
    let parallel = ParallelConfig {
        root,
        rank: 0,
        world,
    };
    let mut engine = match Engine::prepare(&api, &plan, dir, device, options, Some(&parallel)) {
        Ok(engine) => engine,
        Err(e) => {
            memory::save_failure(options, &e.to_string());
            return Err(e);
        }
    };
    for peer in &peers {
        peer.rx
            .recv()
            .map_err(|_| invalid("TP startup peer stopped"))?
            .map_err(invalid)?;
    }
    eprintln!(
        "TP world={world} devices={device}..{} captured matmul output shards + HCCL allgather",
        device + world as i32 - 1
    );
    run(&mut EngineSet {
        primary: &mut engine,
        peers,
    })
}

/// Coordinate rank lifetime and record failures even when device/bootstrap validation fails.
pub fn with_serving_engine<R>(
    plan: &Plan,
    dir: &Path,
    library: &Path,
    device: i32,
    options: &StartupOptions,
    run: impl FnOnce(&mut EngineSet<'_, '_>) -> Result<R>,
) -> Result<R> {
    let result = with_serving_engine_inner(plan, dir, library, device, options, run);
    if let Err(ref error) = result {
        memory::save_failure(options, &format!("{error}"));
    }
    result
}
