//! Bounded-channel mock serving loop. Device computations are intentionally synthetic.
pub mod live;

use inferfabric_ir::Artifact;
use inferfabric_model::{Error, Result, Role, invalid};
use inferfabric_runtime::{HybridState, StepInput, Trace, WorkerGroup, trace};
use inferfabric_transfer::{InMemoryTransport, Lease, Manifest, ResourcePool};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{Arc, Mutex, mpsc},
    thread,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub submit_at: usize,
    #[serde(default)]
    pub cancel_at: Option<usize>,
    #[serde(default)]
    pub eos: Option<u32>,
}
#[derive(Debug, Clone, Copy)]
pub struct Capacity {
    pub pages: usize,
    pub slots: usize,
}
impl Default for Capacity {
    fn default() -> Self {
        Self {
            pages: 4096,
            slots: 8,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub outputs: BTreeMap<u64, Vec<u32>>,
    pub errors: BTreeMap<u64, String>,
    pub trace: Vec<String>,
    pub remaining_allocations: usize,
    pub pending_transfers: usize,
}
#[derive(Debug)]
enum Phase {
    Queued,
    Prefill,
    Transfer { ready_tick: usize },
    Decode,
    Done,
}
struct Job {
    request: Request,
    phase: Phase,
    state: HybridState,
    p: Option<Lease>,
    d: Option<Lease>,
    next: u32,
    output: Vec<u32>,
}
impl Job {
    fn done(&self) -> bool {
        matches!(self.phase, Phase::Done)
    }
}

/// CLI-to-scheduler input and scheduler-to-worker queues are all bounded.
pub fn run_mock(artifact: Artifact, requests: Vec<Request>, capacity: Capacity) -> Result<Report> {
    let (tx, rx) = mpsc::sync_channel(1);
    let join = thread::spawn(move || {
        let requests = rx.recv().map_err(|e| Error::Backend(e.to_string()))?;
        run(artifact, requests, capacity)
    });
    tx.send(requests)
        .map_err(|e| Error::Backend(e.to_string()))?;
    join.join()
        .map_err(|_| Error::Backend("scheduler thread panicked".into()))?
}
fn release(
    job: &mut Job,
    pool: &mut ResourcePool,
    transport: &mut InMemoryTransport,
) -> Result<()> {
    if let Some(p) = job.p.take() {
        pool.retire(&p)?;
    }
    if let Some(d) = job.d.take() {
        transport.abort(&d)?;
    }
    job.phase = Phase::Done;
    Ok(())
}
fn run(artifact: Artifact, requests: Vec<Request>, capacity: Capacity) -> Result<Report> {
    artifact.spec.validate()?;
    let spec = &artifact.spec;
    let mut ids = BTreeSet::new();
    for r in &requests {
        if !ids.insert(r.id)
            || r.prompt.is_empty()
            || r.max_new_tokens == 0
            || r.prompt.iter().any(|&t| t as usize >= spec.model.vocab)
            || r.prompt
                .len()
                .checked_add(r.max_new_tokens)
                .is_none_or(|n| n > spec.max_context())
            || r.cancel_at.is_some_and(|t| t < r.submit_at)
            || r.submit_at > 100_000
        {
            return Err(invalid("invalid request fixture"));
        }
    }
    let log: Trace = Arc::new(Mutex::new(vec![]));
    let p_workers = WorkerGroup::mock(&artifact, Role::Prefill, log.clone())?;
    let d_workers = WorkerGroup::mock(&artifact, Role::Decode, log.clone())?;
    trace(&log, "admission Ready transport=in-memory backend=mock");
    let mut p_pool = ResourcePool::new(capacity.pages, capacity.slots, spec.page_tokens)?;
    let mut transport = InMemoryTransport::new(
        spec.clone(),
        artifact.key.clone(),
        capacity.pages,
        capacity.slots,
    )?;
    let mut jobs: Vec<Job> = requests
        .into_iter()
        .map(|request| Job {
            request,
            phase: Phase::Queued,
            state: HybridState::empty(spec),
            p: None,
            d: None,
            next: 0,
            output: vec![],
        })
        .collect();
    let mut errors = BTreeMap::new();
    let mut prefill_queue = VecDeque::new();
    let mut tick = 0usize;
    // Every productive iteration consumes a token or crosses a bounded transfer boundary.
    let max_ticks = jobs.iter().map(|j| j.request.submit_at).max().unwrap_or(0)
        + jobs
            .iter()
            .map(|j| j.request.prompt.len() + j.request.max_new_tokens + 4)
            .sum::<usize>()
        + 10;
    while jobs.iter().any(|j| !j.done()) {
        if tick > max_ticks {
            return Err(Error::Backend("scheduler made no progress".into()));
        }
        for (index, job) in jobs.iter_mut().enumerate() {
            if job.done() || job.request.submit_at > tick {
                continue;
            }
            if job.request.cancel_at == Some(tick) {
                trace(&log, format!("tick={tick} cancel id={}", job.request.id));
                release(job, &mut p_pool, &mut transport)?;
                errors.insert(job.request.id, "cancelled".into());
                continue;
            }
            if matches!(job.phase, Phase::Queued) {
                if job.request.prompt.len().div_ceil(spec.page_tokens) > capacity.pages {
                    errors.insert(job.request.id, "prefill capacity".into());
                    job.phase = Phase::Done;
                    continue;
                }
                match p_pool.reserve(job.request.id, 1, job.request.prompt.len()) {
                    Ok(lease) => {
                        job.p = Some(lease);
                        job.phase = Phase::Prefill;
                        prefill_queue.push_back(index);
                        trace(&log, format!("tick={tick} submit id={}", job.request.id));
                    }
                    Err(Error::Capacity(_)) => {}
                    Err(e) => return Err(e),
                }
            }
        }
        // Completion visibility and all-rank readiness precede decode admission/ACK.
        for job in &mut jobs {
            if matches!(job.phase,Phase::Transfer{ready_tick} if ready_tick<=tick) {
                let lease = job.d.as_ref().ok_or_else(|| invalid("missing D lease"))?;
                transport.complete(lease)?;
                let manifest = transport.commit(lease)?;
                job.state = manifest.state;
                job.next = manifest.next_token;
                transport.acknowledge(lease)?;
                if let Some(p) = job.p.take() {
                    p_pool.retire(&p)?;
                }
                job.output.push(job.next);
                job.phase = Phase::Decode;
                trace(
                    &log,
                    format!(
                        "tick={tick} pd_commit_ack id={} consumed={} first={}",
                        job.request.id, job.state.consumed, job.next
                    ),
                );
                if job.output.len() == job.request.max_new_tokens
                    || job.request.eos == Some(job.next)
                {
                    release(job, &mut p_pool, &mut transport)?;
                    trace(&log, format!("tick={tick} finish id={}", job.request.id));
                }
            }
        }
        let mut decode_indices = Vec::new();
        let mut inputs = Vec::new();
        for (index, job) in jobs.iter_mut().enumerate() {
            if !matches!(job.phase, Phase::Decode) {
                continue;
            }
            if inputs.len()
                >= *spec
                    .batch_buckets
                    .last()
                    .ok_or_else(|| invalid("no batch buckets"))?
            {
                break;
            }
            let lease = job
                .d
                .as_ref()
                .ok_or_else(|| invalid("missing decode lease"))?;
            match transport.pool.ensure(lease, job.state.consumed + 1) {
                Ok(l) => job.d = Some(l),
                Err(Error::Capacity(_)) => {
                    errors.insert(job.request.id, "decode KV capacity".into());
                    release(job, &mut p_pool, &mut transport)?;
                    continue;
                }
                Err(e) => return Err(e),
            }
            inputs.push(StepInput {
                binding: job
                    .d
                    .as_ref()
                    .ok_or_else(|| invalid("missing D binding"))?
                    .binding(),
                request: job.request.id,
                epoch: 1,
                tokens: vec![job.next],
                state: job.state.clone(),
            });
            decode_indices.push(index);
        }
        if !inputs.is_empty() {
            trace(
                &log,
                format!(
                    "tick={tick} decode_batch ids={:?}",
                    inputs.iter().map(|i| i.request).collect::<Vec<_>>()
                ),
            );
            let outputs = d_workers.step(&inputs)?;
            for (index, output) in decode_indices.into_iter().zip(outputs) {
                let job = &mut jobs[index];
                job.state = output.state;
                job.next = output.next_token;
                job.output.push(job.next);
                if job.output.len() == job.request.max_new_tokens
                    || job.request.eos == Some(job.next)
                {
                    release(job, &mut p_pool, &mut transport)?;
                    trace(&log, format!("tick={tick} finish id={}", job.request.id));
                }
            }
        }
        let mut budget = spec.token_budget;
        let mut inputs = Vec::new();
        let mut indices = Vec::new();
        let turns = prefill_queue.len();
        for _ in 0..turns {
            let index = prefill_queue
                .pop_front()
                .ok_or_else(|| invalid("prefill queue"))?;
            let job = &jobs[index];
            if !matches!(job.phase, Phase::Prefill) {
                continue;
            }
            let remaining = job.request.prompt.len() - job.state.consumed;
            if remaining > 0 && budget > 0 {
                let chunk = remaining.min(budget);
                inputs.push(StepInput {
                    binding: job
                        .p
                        .as_ref()
                        .ok_or_else(|| invalid("missing P binding"))?
                        .binding(),
                    request: job.request.id,
                    epoch: 1,
                    tokens: job.request.prompt[job.state.consumed..job.state.consumed + chunk]
                        .to_vec(),
                    state: job.state.clone(),
                });
                indices.push(index);
                budget -= chunk;
            }
            prefill_queue.push_back(index);
            if budget == 0 {
                break;
            }
        }
        if !inputs.is_empty() {
            trace(
                &log,
                format!(
                    "tick={tick} prefill_budget used={}",
                    spec.token_budget - budget
                ),
            );
            for input in &inputs {
                trace(
                    &log,
                    format!(
                        "tick={tick} chunk id={} start={} len={}",
                        input.request,
                        input.state.consumed,
                        input.tokens.len()
                    ),
                );
            }
            for (index, output) in indices.into_iter().zip(p_workers.step(&inputs)?) {
                jobs[index].state = output.state;
                jobs[index].next = output.next_token;
            }
        }
        for job in &mut jobs {
            if !matches!(job.phase, Phase::Prefill)
                || job.state.consumed != job.request.prompt.len()
            {
                continue;
            }
            if job.state.consumed.div_ceil(spec.page_tokens) > capacity.pages {
                errors.insert(job.request.id, "PD capacity".into());
                release(job, &mut p_pool, &mut transport)?;
                continue;
            }
            match transport.reserve(job.request.id, 1, job.state.consumed) {
                Ok(lease) => {
                    transport.transfer(Manifest {
                        lease: lease.clone(),
                        model_key: artifact.key.clone(),
                        state: job.state.clone(),
                        next_token: job.next,
                    })?;
                    job.d = Some(lease);
                    job.phase = Phase::Transfer {
                        ready_tick: tick + 1,
                    };
                    trace(
                        &log,
                        format!(
                            "tick={tick} pd_transfer id={} kv={} recurrent={} conv={}",
                            job.request.id,
                            job.state.kv.len(),
                            job.state.recurrent.len(),
                            job.state.conv.len()
                        ),
                    );
                }
                Err(Error::Capacity(_)) => {}
                Err(e) => return Err(e),
            }
        }
        // Mock worker steps and transfers are complete/cancelled before this fence.
        p_pool.completion_fence();
        transport.completion_fence();
        tick += 1;
    }
    drop(p_workers);
    drop(d_workers);
    let remaining_allocations = p_pool.allocations() + transport.pool.allocations();
    let pending_transfers = transport.pending();
    let outputs = jobs.into_iter().map(|j| (j.request.id, j.output)).collect();
    let events = log
        .lock()
        .map_err(|_| Error::Backend("trace lock".into()))?
        .clone();
    Ok(Report {
        outputs,
        errors,
        trace: events,
        remaining_allocations,
        pending_transfers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferfabric_ir::Target;
    fn artifact() -> Artifact {
        let mut s = inferfabric_dsl::parse(include_str!("../../../examples/qwen35-2b.inferfabric"))
            .unwrap();
        s.page_tokens = 2;
        s.context_buckets = vec![8, 32, 128];
        inferfabric_compiler::compile(&s, &Target::mock()).unwrap()
    }
    fn request(id: u64, at: usize, length: usize, new: usize) -> Request {
        Request {
            id,
            prompt: (1..=length as u32).collect(),
            max_new_tokens: new,
            submit_at: at,
            cancel_at: None,
            eos: None,
        }
    }
    #[test]
    fn continuous_batching_and_chunk_boundaries() {
        let a = artifact();
        let report = run_mock(
            a.clone(),
            vec![request(1, 0, 7, 12), request(2, 2, 3, 4)],
            Capacity::default(),
        )
        .unwrap();
        assert!(report.errors.is_empty());
        assert_eq!(report.outputs[&1].len(), 12);
        assert_eq!(report.outputs[&2].len(), 4);
        assert_eq!(report.remaining_allocations, 0);
        assert_eq!(report.pending_transfers, 0);
        let position = |needle: &str| {
            report
                .trace
                .iter()
                .position(|s| s.contains(needle))
                .unwrap()
        };
        assert!(position("pd_commit_ack id=2") < position("finish id=1"));
        assert!(
            report
                .trace
                .iter()
                .any(|s| s.contains("chunk id=1 start=4 len=3"))
        );
        assert!(
            report
                .trace
                .iter()
                .any(|s| s.contains("decode_batch ids=[1, 2]"))
        );
        let solo = run_mock(a, vec![request(1, 0, 7, 12)], Capacity::default()).unwrap();
        assert_eq!(report.outputs[&1], solo.outputs[&1]);
    }
    #[test]
    fn cancellation_during_transfer_and_slot_reuse() {
        let mut cancelled = request(1, 0, 2, 4);
        cancelled.cancel_at = Some(1);
        let report = run_mock(
            artifact(),
            vec![cancelled, request(2, 2, 2, 3)],
            Capacity { pages: 8, slots: 1 },
        )
        .unwrap();
        assert_eq!(report.errors[&1], "cancelled");
        assert!(report.outputs[&1].is_empty());
        assert_eq!(report.outputs[&2].len(), 3);
        assert_eq!(report.remaining_allocations, 0);
        assert_eq!(report.pending_transfers, 0);
    }
    #[test]
    fn capacity_failure_and_eos_reclaim_resources() {
        let a = artifact();
        let report = run_mock(
            a.clone(),
            vec![request(1, 0, 3, 3)],
            Capacity { pages: 1, slots: 1 },
        )
        .unwrap();
        assert!(report.errors.contains_key(&1));
        assert_eq!(report.remaining_allocations, 0);
        let mut eos = request(2, 0, 2, 5);
        eos.eos = Some(5);
        let report = run_mock(a.clone(), vec![eos], Capacity::default()).unwrap();
        assert_eq!(report.outputs[&2], vec![5]);
        assert_eq!(report.remaining_allocations, 0);
        let report = run_mock(
            a,
            vec![request(3, 0, 2, 5)],
            Capacity { pages: 1, slots: 1 },
        )
        .unwrap();
        assert!(report.errors.contains_key(&3));
        assert_eq!(report.remaining_allocations, 0);
        assert_eq!(report.pending_transfers, 0);
    }
    #[test]
    fn parallel_plans_trace_mock_communications() {
        let mut a = artifact();
        a.spec.prefill.tp = 2;
        a.spec.decode.tp = 2;
        a.spec.prefill.cp = 2;
        a.spec.prefill.sp = true;
        let a = inferfabric_compiler::compile(&a.spec, &a.target).unwrap();
        let report = run_mock(a, vec![request(1, 0, 3, 3)], Capacity::default()).unwrap();
        for op in ["ReduceScatter", "AllGather", "AllReduce", "Send", "Receive"] {
            assert!(report.trace.iter().any(|x| x.contains(op)), "missing {op}");
        }
        assert_eq!(report.remaining_allocations, 0);
    }
    #[test]
    fn pd_duplicates_and_stale_epochs() {
        let a = artifact();
        let mut t = InMemoryTransport::new(a.spec.clone(), a.key.clone(), 8, 1).unwrap();
        let l = t.reserve(1, 1, 1).unwrap();
        let manifest = Manifest {
            lease: l.clone(),
            model_key: a.key.clone(),
            state: HybridState::empty(&a.spec),
            next_token: 1,
        };
        t.transfer(manifest.clone()).unwrap();
        assert!(t.commit(&l).is_err());
        t.transfer(manifest.clone()).unwrap();
        t.complete(&l).unwrap();
        assert_eq!(t.commit(&l).unwrap(), manifest);
        t.acknowledge(&l).unwrap();
        t.acknowledge(&l).unwrap();
        t.transfer(manifest).unwrap();
        t.finish(&l).unwrap();
        t.completion_fence();
        let new = t.reserve(1, 2, 1).unwrap();
        assert_eq!(new.slot, l.slot);
        assert_eq!(t.complete(&l), Err(Error::Stale));
        t.abort(&new).unwrap();
        t.completion_fence();
        assert_eq!(t.pool.allocations(), 0);
        assert_eq!(t.pending(), 0);
    }
    #[test]
    fn malformed_hybrid_state_cannot_commit() {
        let a = artifact();
        let mut t = InMemoryTransport::new(a.spec.clone(), a.key.clone(), 8, 1).unwrap();
        let lease = t.reserve(1, 1, 1).unwrap();
        let mut manifest = Manifest {
            lease: lease.clone(),
            model_key: "wrong".into(),
            state: HybridState::empty(&a.spec),
            next_token: 1,
        };
        assert!(t.transfer(manifest.clone()).is_err());
        manifest.model_key = a.key;
        manifest.state.recurrent.clear();
        assert!(t.transfer(manifest).is_err());
        assert!(t.commit(&lease).is_err());
        t.abort(&lease).unwrap();
        t.completion_fence();
        assert_eq!(t.pool.allocations(), 0);
    }
    #[test]
    fn tp_and_sp_plans_preserve_mock_outputs() {
        let a = artifact();
        let expected = run_mock(a.clone(), vec![request(1, 0, 5, 4)], Capacity::default())
            .unwrap()
            .outputs;
        for sp in [false, true] {
            let mut spec = a.spec.clone();
            spec.prefill.tp = 2;
            spec.decode.tp = 2;
            spec.prefill.sp = sp;
            let plan = inferfabric_compiler::compile(&spec, &a.target).unwrap();
            let report = run_mock(plan, vec![request(1, 0, 5, 4)], Capacity::default()).unwrap();
            assert_eq!(report.outputs, expected);
            assert_eq!(report.remaining_allocations, 0);
        }
    }
}
