//! Persistent scheduling over explicitly selected mock workers.
use crate::{Capacity, Request};
use pangu_ir::Artifact;
use pangu_model::{Error, Result, Role, invalid};
use pangu_runtime::{HybridState, StepInput, Trace, WorkerGroup};
use pangu_transfer::{InMemoryTransport, Lease, Manifest, ResourcePool};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    Length,
    Stop,
    Abort,
    Error(String),
}
#[derive(Debug, Clone)]
pub struct Event {
    pub request: u64,
    pub token: Option<u32>,
    pub finish: Option<Finish>,
}
struct Job {
    request: Request,
    state: HybridState,
    prefill: Option<Lease>,
    decode: Option<Lease>,
    next: u32,
    generated: usize,
    stop_tokens: Vec<u32>,
}
pub struct LiveScheduler {
    artifact: Artifact,
    prefill: WorkerGroup,
    decode: WorkerGroup,
    p_pool: ResourcePool,
    transport: InMemoryTransport,
    jobs: BTreeMap<u64, Job>,
    queue: VecDeque<u64>,
    capacity: Capacity,
    epoch: u64,
    pub trace: Trace,
}
impl LiveScheduler {
    pub fn mock(artifact: Artifact, capacity: Capacity) -> Result<Self> {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let prefill = WorkerGroup::mock(&artifact, Role::Prefill, trace.clone())?;
        let decode = WorkerGroup::mock(&artifact, Role::Decode, trace.clone())?;
        let p_pool = ResourcePool::new(capacity.pages, capacity.slots, artifact.spec.page_tokens)?;
        let transport = InMemoryTransport::new(
            artifact.spec.clone(),
            artifact.key.clone(),
            capacity.pages,
            capacity.slots,
        )?;
        Ok(Self {
            artifact,
            prefill,
            decode,
            p_pool,
            transport,
            jobs: BTreeMap::new(),
            queue: VecDeque::new(),
            capacity,
            epoch: 0,
            trace,
        })
    }
    pub fn is_idle(&self) -> bool {
        self.jobs.is_empty()
    }
    pub fn allocations(&self) -> usize {
        self.p_pool.allocations() + self.transport.pool.allocations()
    }
    pub fn submit(&mut self, mut request: Request, stop_tokens: Vec<u32>) -> Result<()> {
        let spec = &self.artifact.spec;
        if self.jobs.contains_key(&request.id)
            || request.prompt.is_empty()
            || request.max_new_tokens == 0
            || request
                .prompt
                .iter()
                .any(|&t| t as usize >= spec.model.vocab)
            || request
                .prompt
                .len()
                .checked_add(request.max_new_tokens)
                .is_none_or(|n| n > spec.max_context())
        {
            return Err(invalid("invalid live request"));
        }
        if self.jobs.len() >= self.capacity.slots
            || request.prompt.len().div_ceil(spec.page_tokens) > self.capacity.pages
        {
            return Err(Error::Capacity("live admission".into()));
        }
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| invalid("request epoch overflow"))?;
        let lease = self
            .p_pool
            .reserve(request.id, self.epoch, request.prompt.len())?;
        request.submit_at = 0;
        request.cancel_at = None;
        self.queue.push_back(request.id);
        self.jobs.insert(
            request.id,
            Job {
                state: HybridState::empty(spec),
                request,
                prefill: Some(lease),
                decode: None,
                next: 0,
                generated: 0,
                stop_tokens,
            },
        );
        Ok(())
    }
    fn remove(&mut self, id: u64) -> Result<()> {
        if let Some(job) = self.jobs.remove(&id) {
            if let Some(lease) = job.prefill {
                self.p_pool.retire(&lease)?;
            }
            if let Some(lease) = job.decode {
                self.transport.finish(&lease)?;
            }
        }
        self.queue.retain(|&value| value != id);
        // Mock execution and transfers complete synchronously. Hardware needs actual fences.
        self.p_pool.completion_fence();
        self.transport.completion_fence();
        Ok(())
    }
    pub fn cancel(&mut self, id: u64) -> Result<Option<Event>> {
        let existed = self.jobs.contains_key(&id);
        self.remove(id)?;
        Ok(existed.then_some(Event {
            request: id,
            token: None,
            finish: Some(Finish::Abort),
        }))
    }
    fn emit(&mut self, id: u64, events: &mut Vec<Event>) -> Result<()> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or_else(|| invalid("missing live request"))?;
        job.generated += 1;
        let finish = if job.request.eos == Some(job.next) || job.stop_tokens.contains(&job.next) {
            Some(Finish::Stop)
        } else if job.generated == job.request.max_new_tokens {
            Some(Finish::Length)
        } else {
            None
        };
        events.push(Event {
            request: id,
            token: Some(job.next),
            finish: finish.clone(),
        });
        if finish.is_some() {
            self.remove(id)?;
        }
        Ok(())
    }
    pub fn step(&mut self) -> Result<Vec<Event>> {
        let mut events = Vec::new();
        let mut inputs = Vec::new();
        let batch = *self
            .artifact
            .spec
            .batch_buckets
            .last()
            .ok_or_else(|| invalid("no buckets"))?;
        let ids: Vec<_> = self
            .jobs
            .iter()
            .filter(|(_, j)| j.decode.is_some())
            .map(|(&id, _)| id)
            .take(batch)
            .collect();
        for id in ids {
            let job = self.jobs.get_mut(&id).unwrap();
            match self
                .transport
                .pool
                .ensure(job.decode.as_ref().unwrap(), job.state.consumed + 1)
            {
                Ok(lease) => {
                    inputs.push(StepInput {
                        request: id,
                        epoch: lease.epoch,
                        tokens: vec![job.next],
                        state: job.state.clone(),
                        binding: lease.binding(),
                    });
                    job.decode = Some(lease);
                }
                Err(Error::Capacity(message)) => {
                    self.remove(id)?;
                    events.push(Event {
                        request: id,
                        token: None,
                        finish: Some(Finish::Error(message)),
                    });
                }
                Err(error) => return Err(error),
            }
        }
        if !inputs.is_empty() {
            for output in self.decode.step(&inputs)? {
                let job = self.jobs.get_mut(&output.request).unwrap();
                job.state = output.state;
                job.next = output.next_token;
                self.emit(output.request, &mut events)?;
            }
        }
        inputs.clear();
        let mut budget = self.artifact.spec.token_budget;
        for _ in 0..self.queue.len() {
            if budget == 0 {
                break;
            }
            let id = self.queue.pop_front().unwrap();
            let job = self.jobs.get(&id).unwrap();
            if let Some(lease) = &job.prefill {
                let start = job.state.consumed;
                let chunk = (job.request.prompt.len() - start).min(budget);
                if chunk > 0 {
                    inputs.push(StepInput {
                        request: id,
                        epoch: lease.epoch,
                        tokens: job.request.prompt[start..start + chunk].to_vec(),
                        state: job.state.clone(),
                        binding: lease.binding(),
                    });
                    budget -= chunk;
                }
                self.queue.push_back(id);
            }
        }
        if !inputs.is_empty() {
            for output in self.prefill.step(&inputs)? {
                let job = self.jobs.get_mut(&output.request).unwrap();
                job.state = output.state;
                job.next = output.next_token;
            }
        }
        let ready: Vec<_> = self
            .jobs
            .iter()
            .filter(|(_, j)| j.prefill.is_some() && j.state.consumed == j.request.prompt.len())
            .map(|(&id, _)| id)
            .collect();
        for id in ready {
            let job = self.jobs.get_mut(&id).unwrap();
            let lease = match self.transport.reserve(
                id,
                job.prefill.as_ref().unwrap().epoch,
                job.state.consumed,
            ) {
                Ok(lease) => lease,
                Err(Error::Capacity(_)) => continue,
                Err(error) => return Err(error),
            };
            job.decode = Some(lease.clone());
            self.transport.transfer(Manifest {
                lease: lease.clone(),
                model_key: self.artifact.key.clone(),
                state: job.state.clone(),
                next_token: job.next,
            })?;
            self.transport.complete(&lease)?;
            let manifest = self.transport.commit(&lease)?;
            self.transport.acknowledge(&lease)?;
            job.state = manifest.state;
            job.next = manifest.next_token;
            self.p_pool.retire(&job.prefill.take().unwrap())?;
            self.p_pool.completion_fence();
            self.queue.retain(|&value| value != id);
            self.emit(id, &mut events)?;
        }
        let mut trace = self.trace.lock().unwrap();
        if trace.len() > 4096 {
            let excess = trace.len() - 4096;
            trace.drain(..excess);
        }
        Ok(events)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_cancellation_and_pd_match_fixture() {
        let spec = pangu_dsl::parse(include_str!("../../../examples/qwen35-2b.pangu")).unwrap();
        let artifact = pangu_compiler::compile(&spec, &pangu_ir::Target::mock()).unwrap();
        let request = Request {
            id: 1,
            prompt: vec![1, 2, 3, 4, 5],
            max_new_tokens: 5,
            eos: None,
            submit_at: 0,
            cancel_at: None,
        };
        let expected =
            crate::run_mock(artifact.clone(), vec![request.clone()], Capacity::default()).unwrap();
        let mut live = LiveScheduler::mock(artifact, Capacity::default()).unwrap();
        live.submit(request.clone(), vec![]).unwrap();
        assert!(live.step().unwrap().is_empty());
        live.submit(
            Request {
                id: 2,
                ..request.clone()
            },
            vec![],
        )
        .unwrap();
        assert_eq!(live.cancel(2).unwrap().unwrap().finish, Some(Finish::Abort));
        let mut tokens = Vec::new();
        while !live.is_idle() {
            tokens.extend(live.step().unwrap().into_iter().filter_map(|e| e.token));
        }
        assert_eq!(tokens, expected.outputs[&1]);
        assert_eq!(live.allocations(), 0);
        live.submit(request, vec![]).unwrap();
        live.cancel(1).unwrap();
        assert_eq!(live.allocations(), 0);
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;
    #[test]
    fn new_request_joins_running_decode_without_recapture() {
        let spec = pangu_dsl::parse(include_str!("../../../examples/qwen35-2b.pangu")).unwrap();
        let artifact = pangu_compiler::compile(&spec, &pangu_ir::Target::mock()).unwrap();
        let mut engine = LiveScheduler::mock(artifact, Capacity::default()).unwrap();
        let captures = engine
            .trace
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.starts_with("capture "))
            .count();
        let first = Request {
            id: 1,
            prompt: vec![1],
            max_new_tokens: 8,
            submit_at: 0,
            cancel_at: None,
            eos: None,
        };
        engine.submit(first.clone(), vec![]).unwrap();
        let initial = engine.step().unwrap();
        assert_eq!(initial.len(), 1);
        assert!(initial[0].finish.is_none());
        engine.submit(Request { id: 2, ..first }, vec![]).unwrap();
        let joined = engine.step().unwrap();
        assert!(joined.iter().any(|e| e.request == 1 && e.token.is_some()));
        assert!(joined.iter().any(|e| e.request == 2 && e.token.is_some()));
        assert!(joined.iter().all(|e| e.finish.is_none()));
        assert_eq!(
            engine
                .trace
                .lock()
                .unwrap()
                .iter()
                .filter(|s| s.starts_with("capture "))
                .count(),
            captures
        );
        engine.cancel(1).unwrap();
        engine.cancel(2).unwrap();
        assert!(engine.is_idle());
        assert_eq!(engine.allocations(), 0);
    }
}
