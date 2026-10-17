//! Continuous admission over resident histories; invocations remain single-token.
use super::{Active, Job, error};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::mpsc;
use vllm_llm::{Error, FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest};

pub(super) fn validate(req: &GenerateRequest) -> vllm_llm::Result<()> {
    let p = &req.sampling_params;
    if p.temperature != 0.0
        || p.min_tokens != 0
        || p.presence_penalty != 0.0
        || p.frequency_penalty != 0.0
        || p.repetition_penalty != 1.0
    {
        return Err(Error::Unsupported { message: "program ABI v1 supports greedy sampling without penalties or min_tokens; non-greedy qualification is pending".into() });
    }
    Ok(())
}
pub(super) trait ResidentEngine {
    fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()>;
    fn select(&mut self, slot: u32) -> inferfabric_model::Result<()>;
    fn token(
        &mut self,
        token: u32,
        position: u32,
        sample: bool,
    ) -> inferfabric_model::Result<Option<u32>>;
}
impl ResidentEngine for inferfabric_native::program::Engine {
    fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()> {
        self.reset_resident(slot)
    }
    fn select(&mut self, slot: u32) -> inferfabric_model::Result<()> {
        self.select_resident(slot)
    }
    fn token(
        &mut self,
        token: u32,
        position: u32,
        sample: bool,
    ) -> inferfabric_model::Result<Option<u32>> {
        self.token(token, position, sample)
    }
}
struct Running {
    job: Job,
    position: u32,
    generated: u32,
    next: Option<u32>,
}
fn cancelled(job: &Job) -> bool {
    job.cancel.load(Ordering::Acquire) || job.tx.is_closed()
}
fn release(slot: &mut Option<Running>, active: &Active) {
    if let Some(r) = slot.take() {
        active.lock().unwrap().remove(&r.job.req.request_id);
    }
}
fn advance(engine: &mut impl ResidentEngine, r: &mut Running) -> inferfabric_model::Result<bool> {
    let prompt = &r.job.req.prompt_token_ids;
    let prefill = (r.position as usize) < prompt.len();
    let token = if prefill {
        prompt[r.position as usize]
    } else {
        r.next
            .ok_or_else(|| inferfabric_model::invalid("missing decode input"))?
    };
    let sample = r.position as usize + 1 >= prompt.len();
    let next = engine.token(token, r.position, sample)?;
    r.position += 1;
    if cancelled(&r.job) {
        return Ok(true);
    }
    if !sample {
        return Ok(false);
    }
    let token = next.ok_or_else(|| inferfabric_model::invalid("missing program output token"))?;
    r.next = Some(token);
    r.generated += 1;
    let p = &r.job.req.sampling_params;
    let finish = if p.eos_token_id == Some(token) {
        Some(FinishReason::stop_eos())
    } else if p.stop_token_ids.contains(&token) {
        Some(FinishReason::Stop(Some(
            vllm_engine_core_client::protocol::output::StopReason::TokenId(token),
        )))
    } else if r.generated >= p.max_tokens {
        Some(FinishReason::Length)
    } else {
        None
    };
    let terminal = finish.is_some();
    let output = GenerateOutput {
        request_id: r.job.req.request_id.clone(),
        prompt_info: (r.generated == 1).then(|| GeneratePromptInfo {
            prompt_token_ids: prompt.clone().into(),
            prompt_logprobs: None,
        }),
        token_ids: vec![token],
        logprobs: None,
        finish_reason: finish,
        cached_token_count: 0,
        kv_transfer_params: None,
    };
    Ok(r.job.tx.try_send(Ok(output)).is_err() || terminal)
}
pub(super) fn run(
    engine: &mut impl ResidentEngine,
    rx: &mut mpsc::Receiver<Job>,
    active: &Active,
    stop: &Arc<AtomicBool>,
    residents: usize,
    token_budget: usize,
) -> inferfabric_model::Result<()> {
    if residents == 0 || residents > 64 || token_budget == 0 {
        return Err(inferfabric_model::invalid(
            "invalid program scheduler capacity",
        ));
    }
    let mut slots: Vec<Option<Running>> = (0..residents).map(|_| None).collect();
    let (mut cursor, mut max_active, mut mixed_ticks, mut consumed) = (0, 0, 0, 0u64);
    let result: inferfabric_model::Result<()> = (|| {
        while !stop.load(Ordering::Acquire) {
            for (index, slot) in slots.iter_mut().enumerate() {
                if slot.as_ref().is_some_and(|r| cancelled(&r.job)) {
                    release(slot, active);
                }
                while slot.is_none() {
                    let job = match rx.try_recv() {
                        Ok(job) => job,
                        Err(_) => break,
                    };
                    if cancelled(&job) {
                        active.lock().unwrap().remove(&job.req.request_id);
                        continue;
                    }
                    // Install before fallible reset so failures reach this request.
                    // Every prior native invocation has completed on its own stream.
                    *slot = Some(Running {
                        job,
                        position: 0,
                        generated: 0,
                        next: None,
                    });
                    engine.reset(index as u32)?;
                }
            }
            let count = slots.iter().flatten().count();
            if count > max_active {
                max_active = count;
                eprintln!(
                    "program scheduler active={count} capacity={residents} token_budget={token_budget} invocation_tokens=1"
                );
            }
            if count == 0 {
                if rx.is_closed() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            let start = cursor;
            let (mut tokens, mut prefill, mut decode) = (0, 0, 0);
            for offset in 0..residents {
                if tokens == token_budget || stop.load(Ordering::Acquire) {
                    break;
                }
                let index = (start + offset) % residents;
                let Some(r) = slots[index].as_mut() else {
                    continue;
                };
                if cancelled(&r.job) {
                    release(&mut slots[index], active);
                    continue;
                }
                if (r.position as usize) < r.job.req.prompt_token_ids.len() {
                    prefill += 1;
                } else {
                    decode += 1;
                }
                engine.select(index as u32)?;
                let finished = advance(engine, r)?;
                tokens += 1;
                consumed += 1;
                cursor = (index + 1) % residents;
                if finished {
                    release(&mut slots[index], active);
                }
            }
            if prefill > 0 && decode > 0 {
                mixed_ticks += 1;
            }
        }
        Ok(())
    })();
    rx.close();
    let message = result
        .as_ref()
        .err()
        .map(ToString::to_string)
        .unwrap_or_else(|| "program worker stopped".into());
    for slot in &mut slots {
        if let Some(r) = slot.as_ref() {
            let _ = r.job.tx.try_send(Err(error(message.clone())));
        }
        release(slot, active);
    }
    while let Ok(job) = rx.try_recv() {
        let _ = job.tx.try_send(Err(error(message.clone())));
    }
    active.lock().unwrap().clear();
    eprintln!(
        "program scheduler stopped max_active={max_active} mixed_ticks={mixed_ticks} consumed_tokens={consumed} token_budget={token_budget} invocation_tokens=1"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    fn request(id: &str, prompt: Vec<u32>, outputs: u32) -> GenerateRequest {
        GenerateRequest {
            request_id: id.into(),
            prompt_token_ids: prompt,
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0,"max_tokens":outputs}),
            )
            .unwrap(),
            mm_features: None,
            arrival_time: None,
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        }
    }
    #[derive(Default)]
    struct Fake {
        selected: usize,
        position: [u32; 2],
        sum: [u32; 2],
        trace: Vec<(usize, u32)>,
        resets: Vec<u32>,
        fail_at: Option<usize>,
        cancel_at: Option<(usize, Arc<AtomicBool>)>,
    }
    impl ResidentEngine for Fake {
        fn reset(&mut self, slot: u32) -> inferfabric_model::Result<()> {
            self.resets.push(slot);
            self.position[slot as usize] = 0;
            self.sum[slot as usize] = 0;
            Ok(())
        }
        fn select(&mut self, slot: u32) -> inferfabric_model::Result<()> {
            self.selected = slot as usize;
            Ok(())
        }
        fn token(
            &mut self,
            token: u32,
            position: u32,
            sample: bool,
        ) -> inferfabric_model::Result<Option<u32>> {
            if self.fail_at == Some(self.trace.len()) {
                return Err(inferfabric_model::invalid("injected rank failure"));
            }
            if let Some((at, flag)) = &self.cancel_at {
                if *at == self.trace.len() {
                    flag.store(true, Ordering::Release);
                }
            }
            assert_eq!(position, self.position[self.selected]);
            self.position[self.selected] += 1;
            self.sum[self.selected] += token;
            self.trace.push((self.selected, token));
            Ok(sample.then_some(self.sum[self.selected]))
        }
    }
    fn job(
        req: GenerateRequest,
        active: &Active,
    ) -> (Job, mpsc::Receiver<vllm_llm::Result<GenerateOutput>>) {
        let (tx, rx) = mpsc::channel(32);
        let cancel = Arc::new(AtomicBool::new(false));
        active
            .lock()
            .unwrap()
            .insert(req.request_id.clone(), cancel.clone());
        (Job { req, tx, cancel }, rx)
    }
    #[test]
    fn interleaves_long_prefill_with_decode_and_reuses_only_finished_slot() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, mut ar) = job(request("a", vec![1; 6], 1), &active);
        let (b, mut br) = job(request("b", vec![10], 2), &active);
        let (c, mut cr) = job(request("c", vec![20], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake::default();
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            1,
        )
        .unwrap();
        assert_eq!(
            engine.trace,
            vec![
                (0, 1),
                (1, 10),
                (0, 1),
                (1, 10),
                (0, 1),
                (1, 20),
                (0, 1),
                (0, 1),
                (0, 1)
            ]
        );
        assert_eq!(engine.resets, vec![0, 1, 1]);
        assert_eq!(ar.try_recv().unwrap().unwrap().token_ids, vec![6]);
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![10]);
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![20]);
        assert_eq!(cr.try_recv().unwrap().unwrap().token_ids, vec![20]);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn rank_failure_reaches_all_residents_and_queued_requests() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let mut receivers = Vec::new();
        for id in ["a", "b", "c"] {
            let (j, r) = job(request(id, vec![1; 3], 2), &active);
            tx.try_send(j).unwrap();
            receivers.push(r);
        }
        let mut engine = Fake {
            fail_at: Some(1),
            ..Default::default()
        };
        assert!(
            run(
                &mut engine,
                &mut rx,
                &active,
                &Arc::new(AtomicBool::new(false)),
                2,
                2
            )
            .is_err()
        );
        for mut r in receivers {
            assert!(r.try_recv().unwrap().is_err());
        }
        assert!(active.lock().unwrap().is_empty());
        assert!(tx.is_closed());
    }
    #[test]
    fn queued_cancellation_and_disconnect_do_not_consume_a_slot() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, _ar) = job(request("cancelled", vec![1], 1), &active);
        a.cancel.store(true, Ordering::Release);
        let (b, br) = job(request("disconnected", vec![2], 1), &active);
        drop(br);
        let (c, mut cr) = job(request("live", vec![3], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake::default();
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            2,
        )
        .unwrap();
        assert_eq!(engine.trace, vec![(0, 3)]);
        assert_eq!(engine.resets, vec![0]);
        assert!(cr.try_recv().unwrap().unwrap().finish_reason.is_some());
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn active_cancellation_during_native_step_does_not_emit_or_reset_peer() {
        let active = Arc::new(Mutex::new(Default::default()));
        let (tx, mut rx) = mpsc::channel(4);
        let (a, mut ar) = job(request("a", vec![1; 4], 1), &active);
        let flag = a.cancel.clone();
        let (b, mut br) = job(request("b", vec![2; 4], 1), &active);
        let (c, mut cr) = job(request("c", vec![3], 1), &active);
        for j in [a, b, c] {
            tx.try_send(j).unwrap();
        }
        drop(tx);
        let mut engine = Fake {
            cancel_at: Some((0, flag)),
            ..Default::default()
        };
        run(
            &mut engine,
            &mut rx,
            &active,
            &Arc::new(AtomicBool::new(false)),
            2,
            2,
        )
        .unwrap();
        assert_eq!(engine.resets, vec![0, 1, 0]);
        assert!(ar.try_recv().is_err());
        assert_eq!(br.try_recv().unwrap().unwrap().token_ids, vec![8]);
        assert_eq!(cr.try_recv().unwrap().unwrap().token_ids, vec![3]);
        assert!(active.lock().unwrap().is_empty());
    }
    #[test]
    fn shutdown_during_native_step_closes_active_and_queued_streams() {
        let active = Arc::new(Mutex::new(Default::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, mut rx) = mpsc::channel(4);
        let mut receivers = Vec::new();
        for id in ["a", "b", "queued"] {
            let (j, r) = job(request(id, vec![1; 3], 2), &active);
            tx.try_send(j).unwrap();
            receivers.push(r);
        }
        let mut engine = Fake {
            cancel_at: Some((0, stop.clone())),
            ..Default::default()
        };
        run(&mut engine, &mut rx, &active, &stop, 2, 2).unwrap();
        assert_eq!(engine.trace.len(), 1);
        for mut r in receivers {
            assert!(r.try_recv().unwrap().is_err());
        }
        assert!(active.lock().unwrap().is_empty());
        assert!(tx.is_closed());
    }
    #[test]
    fn initial_program_capabilities_reject_unimplemented_sampling() {
        let mut request = GenerateRequest {
            request_id: "test".into(),
            prompt_token_ids: vec![1],
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0,"max_tokens":2}),
            )
            .unwrap(),
            mm_features: None,
            arrival_time: None,
            cache_salt: None,
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        };
        assert!(validate(&request).is_ok());
        request.sampling_params.temperature = 0.7;
        assert!(validate(&request).is_err());
        request.sampling_params.temperature = 0.;
        request.sampling_params.min_tokens = 1;
        assert!(validate(&request).is_err());
        request.sampling_params.min_tokens = 0;
        request.sampling_params.repetition_penalty = 1.1;
        assert!(validate(&request).is_err());
    }
}
