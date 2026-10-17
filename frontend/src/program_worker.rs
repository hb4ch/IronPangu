//! Persistent ABI v1 baseline. One active request; no packed-prefill claim.
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
fn cancelled(job: &Job, stop: &AtomicBool) -> bool {
    stop.load(Ordering::Acquire) || job.cancel.load(Ordering::Acquire) || job.tx.is_closed()
}
pub(super) fn run(
    engine: &mut inferfabric_native::program::Engine,
    rx: &mut mpsc::Receiver<Job>,
    active: &Active,
    stop: &Arc<AtomicBool>,
) -> inferfabric_model::Result<()> {
    let mut failure = None;
    while !stop.load(Ordering::Acquire) {
        let job = match rx.try_recv() {
            Ok(job) => job,
            Err(mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            Err(mpsc::error::TryRecvError::Disconnected) => break,
        };
        let result = execute(engine, &job, stop);
        active.lock().unwrap().remove(&job.req.request_id);
        if let Err(e) = result {
            let _ = job.tx.try_send(Err(error(e.to_string())));
            failure = Some(e);
            break;
        }
    }
    rx.close();
    while let Ok(job) = rx.try_recv() {
        let _ = job.tx.try_send(Err(error("program worker stopped")));
    }
    active.lock().unwrap().clear();
    if let Some(e) = failure {
        Err(e)
    } else {
        Ok(())
    }
}
fn execute(
    engine: &mut inferfabric_native::program::Engine,
    job: &Job,
    stop: &AtomicBool,
) -> inferfabric_model::Result<()> {
    if cancelled(job, stop) {
        return Ok(());
    }
    engine.reset()?;
    let mut pos = 0;
    let mut next = None;
    for &token in &job.req.prompt_token_ids {
        if cancelled(job, stop) {
            return Ok(());
        }
        next = engine.token(
            token,
            pos,
            pos as usize + 1 == job.req.prompt_token_ids.len(),
        )?;
        pos += 1;
    }
    let p = &job.req.sampling_params;
    for generated in 0..p.max_tokens {
        if cancelled(job, stop) {
            return Ok(());
        }
        let token =
            next.ok_or_else(|| inferfabric_model::invalid("missing program output token"))?;
        let finish = if p.eos_token_id == Some(token) {
            Some(FinishReason::stop_eos())
        } else if p.stop_token_ids.contains(&token) {
            Some(FinishReason::Stop(Some(
                vllm_engine_core_client::protocol::output::StopReason::TokenId(token),
            )))
        } else if generated + 1 == p.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        let terminal = finish.is_some();
        let output = GenerateOutput {
            request_id: job.req.request_id.clone(),
            prompt_info: (generated == 0).then(|| GeneratePromptInfo {
                prompt_token_ids: job.req.prompt_token_ids.clone().into(),
                prompt_logprobs: None,
            }),
            token_ids: vec![token],
            logprobs: None,
            finish_reason: finish,
            cached_token_count: 0,
            kv_transfer_params: None,
        };
        if job.tx.try_send(Ok(output)).is_err() || terminal {
            break;
        }
        next = engine.token(token, pos, true)?;
        pos += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_program_capabilities_reject_unimplemented_sampling() {
        let mut request = GenerateRequest {
            request_id: "test".into(), prompt_token_ids: vec![1],
            sampling_params: serde_json::from_value(serde_json::json!({"temperature":0,"max_tokens":2})).unwrap(),
            mm_features: None, arrival_time: None, cache_salt: None, trace_headers: None, priority: 0,
            data_parallel_rank: None, reasoning_parser_kwargs: None, lora_request: None,
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
