use futures::Stream;
use std::{
    path::PathBuf,
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
};
use tokio::sync::{mpsc, oneshot};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_llm::{
    BackendMetadata, Error, FinishReason, GenerateOutput, GenerateOutputStream, GeneratePromptInfo,
    GenerateRequest, GenerationBackend, Result,
};
fn error(message: impl Into<String>) -> Error {
    Error::Backend {
        message: message.into(),
    }
}
struct Job {
    req: GenerateRequest,
    tx: mpsc::Sender<Result<GenerateOutput>>,
    cancel: Arc<AtomicBool>,
}
type Active = Arc<Mutex<Option<(String, Arc<AtomicBool>)>>>;
pub struct Backend {
    tx: mpsc::Sender<Job>,
    active: Active,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    finished: tokio::sync::Mutex<Option<oneshot::Receiver<()>>>,
    context: usize,
}
struct Guard {
    healthy: Arc<AtomicBool>,
    finished: Option<oneshot::Sender<()>>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.healthy.store(false, Ordering::Release);
        if let Some(done) = self.finished.take() {
            let _ = done.send(());
        }
    }
}
impl Backend {
    pub async fn native(
        plan: pangu_compiler::bound::Plan,
        dir: PathBuf,
        library: PathBuf,
        device: i32,
    ) -> anyhow::Result<Arc<Self>> {
        let context = plan.spec.context_buckets[0];
        let (tx, mut rx) = mpsc::channel::<Job>(1);
        let (ready_tx, ready_rx) = oneshot::channel();
        let (done, finished) = oneshot::channel();
        let active: Active = Arc::new(Mutex::new(None));
        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(false));
        let (worker_active, worker_stop, worker_health) =
            (active.clone(), stop.clone(), healthy.clone());
        std::thread::Builder::new()
            .name("pangu-native".into())
            .spawn(move || {
                let _guard = Guard {
                    healthy: worker_health.clone(),
                    finished: Some(done),
                };
                let mut ready = Some(ready_tx);
                let result =
                    pangu_native::with_engine(&plan, &dir, &library, device, context, |engine| {
                        eprintln!(
                            "native program {} weights {} context {}",
                            engine.program_key, engine.weight_digest, context
                        );
                        worker_health.store(true, Ordering::Release);
                        if ready.take().unwrap().send(Ok(())).is_err() {
                            return Ok(());
                        }
                        while !worker_stop.load(Ordering::Acquire) {
                            // Polling permits shutdown without another command or an async runtime on this thread.
                            let job = match rx.try_recv() {
                                Ok(job) => job,
                                Err(mpsc::error::TryRecvError::Empty) => {
                                    std::thread::sleep(std::time::Duration::from_millis(2));
                                    continue;
                                }
                                Err(mpsc::error::TryRecvError::Disconnected) => break,
                            };
                            let result = generate(engine, &job, &worker_stop);
                            *worker_active.lock().unwrap() = None;
                            if let Err(e) = result {
                                let _ = job.tx.try_send(Err(error(e.to_string())));
                                return Err(e); // An ACL failure invalidates serving readiness.
                            }
                        }
                        Ok(())
                    });
                if let Some(ready) = ready {
                    let _ = ready.send(Err(result
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "native startup stopped".into())));
                } else if let Err(e) = result {
                    eprintln!("native engine stopped: {e}");
                }
            })?;
        ready_rx
            .await
            .map_err(|_| anyhow::anyhow!("native startup thread stopped"))?
            .map_err(anyhow::Error::msg)?;
        Ok(Arc::new(Self {
            tx,
            active,
            stop,
            healthy,
            finished: tokio::sync::Mutex::new(Some(finished)),
            context,
        }))
    }
}
fn generate(
    engine: &mut pangu_native::Engine<'_>,
    job: &Job,
    stop: &AtomicBool,
) -> pangu_model::Result<()> {
    let cancelled =
        || job.cancel.load(Ordering::Acquire) || stop.load(Ordering::Acquire) || job.tx.is_closed();
    if cancelled() {
        return Ok(());
    }
    engine.reset()?;
    let mut next = 0;
    for &token in &job.req.prompt_token_ids {
        if cancelled() {
            return Ok(());
        }
        next = engine.step(token)?;
    }
    let p = &job.req.sampling_params;
    for i in 0..p.max_tokens {
        if cancelled() {
            return Ok(());
        }
        let finish = if p.eos_token_id == Some(next) {
            Some(FinishReason::stop_eos())
        } else if p.stop_token_ids.contains(&next) {
            Some(FinishReason::Stop(Some(
                vllm_engine_core_client::protocol::output::StopReason::TokenId(next),
            )))
        } else if i + 1 == p.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        let terminal = finish.is_some();
        let output = GenerateOutput {
            request_id: job.req.request_id.clone(),
            prompt_info: (i == 0).then(|| GeneratePromptInfo {
                prompt_token_ids: job.req.prompt_token_ids.clone().into(),
                prompt_logprobs: None,
            }),
            token_ids: vec![next],
            logprobs: None,
            finish_reason: finish,
            cached_token_count: 0,
            kv_transfer_params: None,
        };
        if job.tx.try_send(Ok(output)).is_err() || terminal {
            return Ok(());
        }
        next = engine.step(next)?;
    }
    Ok(())
}
struct Output {
    rx: mpsc::Receiver<Result<GenerateOutput>>,
    cancel: Arc<AtomicBool>,
}
impl Stream for Output {
    type Item = Result<GenerateOutput>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}
impl Drop for Backend {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}
#[async_trait::async_trait]
impl GenerationBackend for Backend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            max_model_len: self.context as u32,
            model_dtype: ModelDtype::BFloat16,
            version: "0.25.1 / IronPangu native experimental single-request".into(),
            healthy: self.healthy.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire),
        }
    }
    async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        if !self.metadata().healthy {
            return Err(error("native engine is not ready"));
        }
        crate::backend::validate(&req)?;
        validate_capacity(&req, self.context)?;
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err(error(
                "native experimental backend allows one active request",
            ));
        }
        let id = req.request_id.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel(self.context + 1);
        self.tx
            .try_send(Job {
                req,
                tx,
                cancel: cancel.clone(),
            })
            .map_err(|_| error("native engine stopped or busy"))?;
        *active = Some((id.clone(), cancel.clone()));
        Ok(GenerateOutputStream::from_stream(id, Output { rx, cancel }))
    }
    async fn abort(&self, ids: &[String]) -> Result<()> {
        if let Some((id, cancel)) = self.active.lock().unwrap().as_ref()
            && ids.contains(id)
        {
            cancel.store(true, Ordering::Release);
        }
        Ok(())
    }
    async fn shutdown(&self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(done) = self.finished.lock().await.take() {
            done.await
                .map_err(|_| error("native shutdown thread failed"))?;
        }
        Ok(())
    }
}
fn validate_capacity(req: &GenerateRequest, context: usize) -> Result<()> {
    if req.prompt_token_ids.is_empty()
        || req.prompt_token_ids.iter().any(|&id| id >= 248320)
        || req.sampling_params.max_tokens == 0
        || req
            .prompt_token_ids
            .len()
            .checked_add(req.sampling_params.max_tokens as usize)
            .is_none_or(|n| n > context)
    {
        return Err(Error::Unsupported {
            message: format!(
                "native mode requires valid text tokens, a nonempty prompt, positive max_tokens, and prompt + max_tokens <= {context}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_rejects_invalid_ids_and_capacity_without_device_work() {
        let mut req = GenerateRequest {
            request_id: "check".into(),
            prompt_token_ids: vec![1; 112],
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0,"max_tokens":16}),
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
        assert!(validate_capacity(&req, 128).is_ok());
        req.prompt_token_ids.push(1);
        assert!(validate_capacity(&req, 128).is_err());
        req.prompt_token_ids = vec![248320];
        assert!(validate_capacity(&req, 128).is_err());
        req.prompt_token_ids.clear();
        assert!(validate_capacity(&req, 128).is_err());
        req.prompt_token_ids.push(1);
        req.sampling_params.max_tokens = 0;
        assert!(validate_capacity(&req, 128).is_err());
    }
}
