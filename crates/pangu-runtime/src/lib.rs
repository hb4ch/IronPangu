//! Device contracts, fail-closed Ascend placeholder and deterministic orchestration mock.
use pangu_ir::*;
use pangu_model::{Error, Layer, Result, Role, Spec, invalid};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, SyncSender},
    },
    thread::{self, JoinHandle},
};

pub type Trace = Arc<Mutex<Vec<String>>>;
pub fn trace(log: &Trace, event: impl Into<String>) {
    log.lock().expect("trace mutex poisoned").push(event.into());
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Validate,
    Compile,
    Prepare,
    Warmup,
    Capture,
    ValidateReplay,
    Reset,
    Ready,
    Failed,
    Closed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferHandle(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphHandle(pub u64);
#[derive(Debug, Clone)]
pub struct NativeBundle {
    pub backend: BackendKind,
    pub identity: String,
}
#[derive(Debug, Clone)]
pub enum WeightSource {
    MetadataOnly,
    Safetensors(PathBuf),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridState {
    pub consumed: usize,
    /// Mock markers only. NPU transport must use typed device regions instead.
    pub kv: Vec<Vec<u64>>,
    pub recurrent: Vec<u64>,
    pub conv: Vec<Vec<u64>>,
}
impl HybridState {
    pub fn empty(spec: &Spec) -> Self {
        let full = spec
            .model
            .layers
            .iter()
            .filter(|&&x| x == Layer::Full)
            .count();
        let delta = spec.model.layers.len() - full;
        Self {
            consumed: 0,
            kv: vec![vec![]; full],
            recurrent: vec![0; delta],
            conv: vec![vec![]; delta],
        }
    }
    pub fn validate(&self, spec: &Spec) -> Result<()> {
        let empty = Self::empty(spec);
        if self.consumed > spec.max_context()
            || self.kv.len() != empty.kv.len()
            || self.recurrent.len() != empty.recurrent.len()
            || self.conv.len() != empty.conv.len()
            || self.kv.iter().any(|v| v.len() != self.consumed)
            || self
                .conv
                .iter()
                .any(|v| v.len() != self.consumed.min(spec.model.conv_width))
        {
            return Err(invalid("hybrid state schema/frontier mismatch"));
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub struct StateBinding {
    pub slot: usize,
    pub generation: u64,
    pub pages: Vec<usize>,
}
#[derive(Debug, Clone)]
pub struct StepInput {
    pub request: u64,
    pub epoch: u64,
    pub tokens: Vec<u32>,
    pub state: HybridState,
    pub binding: StateBinding,
}
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub request: u64,
    pub epoch: u64,
    pub next_token: u32,
    pub state: HybridState,
}
#[derive(Debug, Clone)]
pub struct DeviceRegion {
    pub buffer: BufferHandle,
    pub offset: usize,
    pub bytes: usize,
}

/// Completion is synchronous at this boundary: returned outputs and released resources
/// are safe to consume/reuse. An NPU implementation may use async device work internally.
pub trait Backend: Send {
    fn kind(&self) -> BackendKind;
    fn compile_kernels(&mut self, _artifact: &Artifact) -> Result<NativeBundle> {
        Err(Error::NotImplemented("Ascend kernel compilation/loading"))
    }
    fn prepare(&mut self, _plan: &RankPlan, _bundle: &NativeBundle) -> Result<()> {
        Err(Error::NotImplemented("Ascend resource preparation"))
    }
    fn upload_weights(&mut self, _source: &WeightSource) -> Result<()> {
        Err(Error::NotImplemented("Ascend weight upload"))
    }
    fn allocate(&mut self, _bytes: usize) -> Result<BufferHandle> {
        Err(Error::NotImplemented("Ascend allocation"))
    }
    fn release(&mut self, _buffer: BufferHandle) -> Result<()> {
        Err(Error::NotImplemented("Ascend buffer release"))
    }
    fn state_region(
        &mut self,
        _buffer: BufferHandle,
        _offset: usize,
        _bytes: usize,
    ) -> Result<DeviceRegion> {
        Err(Error::NotImplemented("Ascend state region"))
    }
    fn warmup(&mut self, _spec: &Spec) -> Result<()> {
        Err(Error::NotImplemented("Ascend warmup"))
    }
    fn capture(&mut self, _bucket: Bucket) -> Result<GraphHandle> {
        Err(Error::NotImplemented("ACL Graph capture"))
    }
    fn validate_graph(&mut self, _graph: GraphHandle, _bucket: Bucket) -> Result<()> {
        Err(Error::NotImplemented("ACL Graph validation"))
    }
    fn reset_scratch(&mut self) -> Result<()> {
        Err(Error::NotImplemented("Ascend scratch reset"))
    }
    fn execute(
        &mut self,
        _plan: &RankPlan,
        _spec: &Spec,
        _inputs: &[StepInput],
        _graph: Option<GraphHandle>,
    ) -> Result<Vec<StepOutput>> {
        Err(Error::NotImplemented("Ascend prefill/decode"))
    }
    fn communicate(&mut self, _instruction: &Instruction) -> Result<()> {
        Err(Error::NotImplemented("HCCL communication"))
    }
    fn synchronize(&mut self) -> Result<()> {
        Err(Error::NotImplemented("ACL synchronization"))
    }
    fn close(&mut self) -> Result<()> {
        Err(Error::NotImplemented("Ascend cleanup"))
    }
}
pub struct AscendBackend;
impl Backend for AscendBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Ascend
    }
}

#[derive(Default)]
pub struct MockBackend {
    next: u64,
    buffers: BTreeMap<u64, usize>,
    graphs: BTreeMap<u64, Bucket>,
}
impl MockBackend {
    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }
}
impl Backend for MockBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Mock
    }
    fn compile_kernels(&mut self, a: &Artifact) -> Result<NativeBundle> {
        Ok(NativeBundle {
            backend: self.kind(),
            identity: a.key.clone(),
        })
    }
    fn prepare(&mut self, _: &RankPlan, b: &NativeBundle) -> Result<()> {
        if b.backend != self.kind() {
            return Err(invalid("native backend mismatch"));
        }
        Ok(())
    }
    fn upload_weights(&mut self, s: &WeightSource) -> Result<()> {
        match s {
            WeightSource::MetadataOnly => Ok(()),
            _ => Err(Error::NotImplemented("mock tensor weights")),
        }
    }
    fn allocate(&mut self, bytes: usize) -> Result<BufferHandle> {
        let id = self.id();
        self.buffers.insert(id, bytes);
        Ok(BufferHandle(id))
    }
    fn release(&mut self, b: BufferHandle) -> Result<()> {
        self.buffers
            .remove(&b.0)
            .ok_or_else(|| invalid("unknown buffer"))?;
        Ok(())
    }
    fn state_region(
        &mut self,
        b: BufferHandle,
        offset: usize,
        bytes: usize,
    ) -> Result<DeviceRegion> {
        let n = self
            .buffers
            .get(&b.0)
            .ok_or_else(|| invalid("unknown buffer"))?;
        if offset.checked_add(bytes).is_none_or(|end| end > *n) {
            return Err(invalid("region bounds"));
        }
        Ok(DeviceRegion {
            buffer: b,
            offset,
            bytes,
        })
    }
    fn warmup(&mut self, _: &Spec) -> Result<()> {
        Ok(())
    }
    fn capture(&mut self, b: Bucket) -> Result<GraphHandle> {
        let id = self.id();
        self.graphs.insert(id, b);
        Ok(GraphHandle(id))
    }
    fn validate_graph(&mut self, g: GraphHandle, b: Bucket) -> Result<()> {
        if self.graphs.get(&g.0) != Some(&b) {
            return Err(invalid("unknown graph"));
        }
        Ok(())
    }
    fn reset_scratch(&mut self) -> Result<()> {
        Ok(())
    }
    fn execute(
        &mut self,
        plan: &RankPlan,
        s: &Spec,
        inputs: &[StepInput],
        graph: Option<GraphHandle>,
    ) -> Result<Vec<StepOutput>> {
        // Structural execution only. The real backend must interleave kernels and
        // communication in this order; runtime never issues collectives ahead of it.
        for instruction in &plan.instructions {
            match instruction {
                Instruction::Communication { .. } => self.communicate(instruction)?,
                Instruction::Fence => self.synchronize()?,
                Instruction::Kernel { .. } => {}
            }
        }
        if let Some(g) = graph {
            let b = self
                .graphs
                .get(&g.0)
                .ok_or_else(|| invalid("unknown graph"))?;
            if inputs.len() > b.batch
                || inputs
                    .iter()
                    .any(|i| i.tokens.len() != 1 || i.state.consumed + 1 > b.context)
            {
                return Err(invalid("graph capacity"));
            }
        }
        inputs
            .iter()
            .map(|input| {
                let mut state = input.state.clone();
                state.validate(s)?;
                if input.tokens.is_empty()
                    || state
                        .consumed
                        .checked_add(input.tokens.len())
                        .is_none_or(|n| n > s.max_context())
                {
                    return Err(Error::Capacity("context".into()));
                }
                for &token in &input.tokens {
                    if token as usize >= s.model.vocab {
                        return Err(invalid("token outside vocabulary"));
                    }
                    let marker = (token as u64)
                        .wrapping_mul(31)
                        .wrapping_add(state.consumed as u64 + 1);
                    for (layer, kv) in state.kv.iter_mut().enumerate() {
                        kv.push(marker.wrapping_add(layer as u64));
                    }
                    for (layer, r) in state.recurrent.iter_mut().enumerate() {
                        *r = r.wrapping_mul(33).wrapping_add(marker + layer as u64);
                    }
                    for conv in &mut state.conv {
                        conv.push(marker);
                        if conv.len() > s.model.conv_width {
                            conv.remove(0);
                        }
                    }
                    state.consumed += 1;
                }
                let last = u64::from(*input.tokens.last().ok_or_else(|| invalid("empty tokens"))?);
                let next_token = ((last + state.consumed as u64 + 1) % s.model.vocab as u64) as u32;
                Ok(StepOutput {
                    request: input.request,
                    epoch: input.epoch,
                    next_token,
                    state,
                })
            })
            .collect()
    }
    fn communicate(&mut self, _: &Instruction) -> Result<()> {
        Ok(())
    }
    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
    fn close(&mut self) -> Result<()> {
        self.graphs.clear();
        self.buffers.clear();
        Ok(())
    }
}

pub struct Runtime {
    backend: Box<dyn Backend>,
    artifact: Artifact,
    plan: RankPlan,
    pub stage: Stage,
    graphs: BTreeMap<Bucket, GraphHandle>,
    log: Trace,
    buffer: Option<BufferHandle>,
}
impl Runtime {
    pub fn new(backend: Box<dyn Backend>, artifact: Artifact, plan: RankPlan, log: Trace) -> Self {
        Self {
            backend,
            artifact,
            plan,
            stage: Stage::Validate,
            graphs: BTreeMap::new(),
            log,
            buffer: None,
        }
    }
    fn enter(&mut self, stage: Stage, fail: Option<Stage>) -> Result<()> {
        self.stage = stage;
        trace(&self.log, format!("startup {:?} {stage:?}", self.plan.rank));
        if fail == Some(stage) {
            return Err(Error::Backend(format!("injected failure at {stage:?}")));
        }
        Ok(())
    }
    pub fn start(&mut self, fail: Option<Stage>) -> Result<()> {
        if self.stage != Stage::Validate {
            return Err(invalid("startup may only run once"));
        }
        let result = self.start_inner(fail);
        if result.is_err() {
            self.stage = Stage::Failed;
            let _ = self.backend.close();
            trace(&self.log, "startup Failed resources_closed");
        }
        result
    }
    fn start_inner(&mut self, fail: Option<Stage>) -> Result<()> {
        self.enter(Stage::Validate, fail)?;
        self.artifact.spec.validate()?;
        if self.artifact != pangu_compiler::compile(&self.artifact.spec, &self.artifact.target)?
            || !self.artifact.ranks.contains(&self.plan)
        {
            return Err(invalid("invalid executable artifact/rank"));
        }
        if self.artifact.target.backend != self.backend.kind() {
            return Err(invalid("artifact/backend mismatch; no automatic fallback"));
        }
        self.enter(Stage::Compile, fail)?;
        let bundle = self.backend.compile_kernels(&self.artifact)?;
        self.enter(Stage::Prepare, fail)?;
        self.backend.prepare(&self.plan, &bundle)?;
        self.backend.upload_weights(&WeightSource::MetadataOnly)?;
        self.buffer = Some(self.backend.allocate(4096)?);
        self.enter(Stage::Warmup, fail)?;
        self.backend.warmup(&self.artifact.spec)?;
        if self.plan.rank.role == Role::Decode {
            let buckets: Vec<_> = self
                .artifact
                .spec
                .batch_buckets
                .iter()
                .flat_map(|&batch| {
                    self.artifact
                        .spec
                        .context_buckets
                        .iter()
                        .map(move |&context| Bucket { batch, context })
                })
                .collect();
            for b in buckets {
                self.enter(Stage::Capture, fail)?;
                let g = self.backend.capture(b)?;
                trace(&self.log, format!("capture {:?} {b:?}", self.plan.rank));
                self.enter(Stage::ValidateReplay, fail)?;
                self.backend.validate_graph(g, b)?;
                let input = StepInput {
                    binding: StateBinding {
                        slot: 0,
                        generation: 0,
                        pages: vec![0],
                    },
                    request: 0,
                    epoch: 0,
                    tokens: vec![1],
                    state: HybridState::empty(&self.artifact.spec),
                };
                self.backend
                    .execute(&self.plan, &self.artifact.spec, &[input], Some(g))?;
                self.backend.synchronize()?;
                self.graphs.insert(b, g);
            }
        }
        self.enter(Stage::Reset, fail)?;
        self.backend.reset_scratch()?;
        self.enter(Stage::Ready, fail)
    }
    pub fn step(&mut self, inputs: &[StepInput]) -> Result<Vec<StepOutput>> {
        if self.stage != Stage::Ready {
            return Err(invalid("worker not ready"));
        }
        if inputs.is_empty() {
            return Err(invalid("empty step"));
        }
        let mut slots = std::collections::BTreeSet::new();
        let mut pages = std::collections::BTreeSet::new();
        for input in inputs {
            if !slots.insert(input.binding.slot)
                || input.binding.pages.iter().any(|p| !pages.insert(*p))
                || input.binding.pages.len()
                    < (input.state.consumed + input.tokens.len())
                        .div_ceil(self.artifact.spec.page_tokens)
            {
                return Err(invalid("invalid/aliased step state bindings"));
            }
            trace(
                &self.log,
                format!(
                    "metadata {:?} id={} epoch={} slot={} generation={} pages={:?}",
                    self.plan.rank,
                    input.request,
                    input.epoch,
                    input.binding.slot,
                    input.binding.generation,
                    input.binding.pages
                ),
            );
        }
        let graph = if self.plan.rank.role == Role::Decode {
            let context = inputs
                .iter()
                .map(|i| i.state.consumed + i.tokens.len())
                .max()
                .unwrap_or(0);
            let batch = self
                .artifact
                .spec
                .batch_buckets
                .iter()
                .copied()
                .find(|&n| n >= inputs.len())
                .ok_or_else(|| Error::Capacity("batch".into()))?;
            let context = self
                .artifact
                .spec
                .context_buckets
                .iter()
                .copied()
                .find(|&n| n >= context)
                .ok_or_else(|| Error::Capacity("context".into()))?;
            trace(
                &self.log,
                format!(
                    "replay {:?} batch={batch} context={context}",
                    self.plan.rank
                ),
            );
            Some(
                *self
                    .graphs
                    .get(&Bucket { batch, context })
                    .ok_or_else(|| invalid("missing startup graph"))?,
            )
        } else {
            trace(
                &self.log,
                format!("prefill {:?} requests={}", self.plan.rank, inputs.len()),
            );
            None
        };
        for instruction in &self.plan.instructions {
            if matches!(instruction, Instruction::Communication { .. }) {
                trace(
                    &self.log,
                    format!("comm {:?} {instruction:?}", self.plan.rank),
                );
            }
        }
        trace(
            &self.log,
            format!(
                "execute_plan {:?} instructions={}",
                self.plan.rank,
                self.plan.instructions.len()
            ),
        );
        let outputs = self
            .backend
            .execute(&self.plan, &self.artifact.spec, inputs, graph)?;
        self.backend.synchronize()?;
        if outputs.len() != inputs.len() {
            return Err(invalid("backend result count"));
        }
        for (input, output) in inputs.iter().zip(&outputs) {
            output.state.validate(&self.artifact.spec)?;
            if output.request != input.request
                || output.epoch != input.epoch
                || output.next_token as usize >= self.artifact.spec.model.vocab
                || output.state.consumed != input.state.consumed + input.tokens.len()
            {
                return Err(invalid("backend request/frontier mismatch"));
            }
        }
        Ok(outputs)
    }
    pub fn close(&mut self) -> Result<()> {
        if self.stage == Stage::Closed {
            return Ok(());
        }
        self.backend.synchronize()?;
        // Backend close destroys graphs before graph-referenced buffers.
        self.backend.close()?;
        self.buffer = None;
        self.graphs.clear();
        self.stage = Stage::Closed;
        trace(&self.log, format!("closed {:?}", self.plan.rank));
        Ok(())
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        if self.stage != Stage::Closed {
            let _ = self.backend.close();
        }
    }
}

enum Command {
    Step(Vec<StepInput>, SyncSender<Result<Vec<StepOutput>>>),
    Stop,
}
struct Worker {
    tx: SyncSender<Command>,
    join: Option<JoinHandle<()>>,
}
pub struct WorkerGroup {
    workers: Vec<Worker>,
}
impl WorkerGroup {
    pub fn mock(artifact: &Artifact, role: Role, log: Trace) -> Result<Self> {
        if artifact.target.backend != BackendKind::Mock {
            return Err(invalid("mock workers require mock artifact"));
        }
        let mut group = Self { workers: vec![] };
        for plan in artifact.ranks.iter().filter(|r| r.rank.role == role) {
            let (tx, rx) = mpsc::sync_channel(2);
            let (ready_tx, ready_rx) = mpsc::sync_channel(1);
            let (a, p, l) = (artifact.clone(), plan.clone(), log.clone());
            let join = thread::spawn(move || {
                let mut runtime = Runtime::new(Box::<MockBackend>::default(), a, p, l);
                let result = runtime.start(None);
                let failed = result.is_err();
                let _ = ready_tx.send(result);
                if failed {
                    return;
                }
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Command::Step(inputs, reply) => {
                            let _ = reply.send(runtime.step(&inputs));
                        }
                        Command::Stop => break,
                    }
                }
                let _ = runtime.close();
            });
            group.workers.push(Worker {
                tx,
                join: Some(join),
            });
            ready_rx
                .recv()
                .map_err(|e| Error::Backend(e.to_string()))??;
        }
        if group.workers.is_empty() {
            return Err(invalid("empty worker group"));
        }
        Ok(group)
    }
    pub fn step(&self, inputs: &[StepInput]) -> Result<Vec<StepOutput>> {
        let mut replies = Vec::new();
        for worker in &self.workers {
            let (tx, rx) = mpsc::sync_channel(1);
            worker
                .tx
                .send(Command::Step(inputs.to_vec(), tx))
                .map_err(|e| Error::Backend(e.to_string()))?;
            replies.push(rx);
        }
        let mut leader = None;
        for rx in replies {
            let output = rx.recv().map_err(|e| Error::Backend(e.to_string()))??;
            if leader.is_none() {
                leader = Some(output);
            }
        }
        leader.ok_or_else(|| invalid("no leader"))
    }
}
impl Drop for WorkerGroup {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(Command::Stop);
        }
        for w in &mut self.workers {
            if let Some(j) = w.join.take() {
                let _ = j.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn artifact() -> Artifact {
        let spec = pangu_dsl::parse(include_str!("../../../examples/qwen35-2b.pangu")).unwrap();
        pangu_compiler::compile(&spec, &Target::mock()).unwrap()
    }
    fn runtime(a: Artifact, log: Trace) -> Runtime {
        let plan = a
            .ranks
            .iter()
            .find(|r| r.rank.role == Role::Decode)
            .unwrap()
            .clone();
        Runtime::new(Box::<MockBackend>::default(), a, plan, log)
    }
    #[test]
    fn readiness_graph_buckets_and_no_lazy_capture() {
        let a = artifact();
        let log = Arc::new(Mutex::new(vec![]));
        let mut r = runtime(a.clone(), log.clone());
        let input = StepInput {
            binding: StateBinding {
                slot: 0,
                generation: 1,
                pages: vec![0],
            },
            request: 1,
            epoch: 1,
            tokens: vec![1],
            state: HybridState::empty(&a.spec),
        };
        assert!(r.step(std::slice::from_ref(&input)).is_err());
        r.start(None).unwrap();
        assert_eq!(
            r.graphs.len(),
            a.spec.batch_buckets.len() * a.spec.context_buckets.len()
        );
        let startup_count = log.lock().unwrap().len();
        let first = r.step(std::slice::from_ref(&input)).unwrap().remove(0);
        let second = StepInput {
            binding: StateBinding {
                slot: 1,
                generation: 3,
                pages: vec![1],
            },
            request: 2,
            epoch: 3,
            tokens: vec![first.next_token],
            state: first.state,
        };
        r.step(&[second, input]).unwrap();
        assert!(
            log.lock().unwrap()[startup_count..]
                .iter()
                .all(|x| !x.contains("capture") && !x.contains("startup"))
        );
        assert!(r.start(None).is_err());
        r.close().unwrap();
        assert_eq!(r.stage, Stage::Closed);
    }
    #[test]
    fn every_startup_failure_prevents_readiness() {
        for stage in [
            Stage::Validate,
            Stage::Compile,
            Stage::Prepare,
            Stage::Warmup,
            Stage::Capture,
            Stage::ValidateReplay,
            Stage::Reset,
            Stage::Ready,
        ] {
            let mut r = runtime(artifact(), Arc::new(Mutex::new(vec![])));
            assert!(r.start(Some(stage)).is_err());
            assert_eq!(r.stage, Stage::Failed);
            assert!(r.step(&[]).is_err());
        }
    }
    #[test]
    fn ascend_fails_closed() {
        let a = artifact();
        let plan = a.ranks[0].clone();
        let mut r = Runtime::new(
            Box::new(AscendBackend),
            a.clone(),
            plan,
            Arc::new(Mutex::new(vec![])),
        );
        assert!(matches!(r.start(None), Err(Error::Invalid(_))));
        let mut target = Target::mock();
        target.backend = BackendKind::Ascend;
        let a = pangu_compiler::compile(&a.spec, &target).unwrap();
        let plan = a.ranks[0].clone();
        let mut r = Runtime::new(
            Box::new(AscendBackend),
            a,
            plan,
            Arc::new(Mutex::new(vec![])),
        );
        assert!(matches!(r.start(None), Err(Error::NotImplemented(_))));
    }
    #[test]
    fn chunked_mock_state_matches_monolithic() {
        let a = artifact();
        let plan = &a.ranks[0];
        let mut b = MockBackend::default();
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        let input = StepInput {
            binding: StateBinding {
                slot: 0,
                generation: 1,
                pages: vec![0],
            },
            request: 1,
            epoch: 1,
            tokens: tokens.clone(),
            state: HybridState::empty(&a.spec),
        };
        let full = b.execute(plan, &a.spec, &[input], None).unwrap().remove(0);
        let mut state = HybridState::empty(&a.spec);
        let mut next = 0;
        for chunk in tokens.chunks(3) {
            let input = StepInput {
                binding: StateBinding {
                    slot: 0,
                    generation: 1,
                    pages: vec![0],
                },
                request: 1,
                epoch: 1,
                tokens: chunk.to_vec(),
                state,
            };
            let out = b.execute(plan, &a.spec, &[input], None).unwrap().remove(0);
            state = out.state;
            next = out.next_token;
        }
        assert_eq!(state, full.state);
        assert_eq!(next, full.next_token);
    }
    #[test]
    fn cached_binary_still_captures_each_start_and_rejects_aliases() {
        let original = artifact();
        let decoded = pangu_compiler::decode(&pangu_compiler::encode(&original).unwrap()).unwrap();
        let log = Arc::new(Mutex::new(vec![]));
        for a in [original, decoded] {
            let mut r = runtime(a.clone(), log.clone());
            r.start(None).unwrap();
            let input = StepInput {
                request: 1,
                epoch: 1,
                tokens: vec![1],
                state: HybridState::empty(&a.spec),
                binding: StateBinding {
                    slot: 0,
                    generation: 1,
                    pages: vec![0],
                },
            };
            assert!(r.step(&[input.clone(), input]).is_err());
            r.close().unwrap();
        }
        assert_eq!(
            log.lock()
                .unwrap()
                .iter()
                .filter(|s| s.starts_with("capture "))
                .count(),
            32
        );
    }
}
