#[path = "native_worker.rs"]
mod worker;
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
    BackendMetadata, Error, GenerateOutput, GenerateOutputStream, GenerateRequest,
    GenerationBackend, Result,
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
type Active = Arc<Mutex<std::collections::BTreeMap<String, Arc<AtomicBool>>>>;
pub struct Backend {
    tx: mpsc::Sender<Job>,
    active: Active,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    finished: tokio::sync::Mutex<Option<oneshot::Receiver<()>>>,
    context: usize,
    defaults: vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams,
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
        options: pangu_native::StartupOptions,
        max_num_batched_tokens: usize,
    ) -> anyhow::Result<Arc<Self>> {
        options.validate()?;
        anyhow::ensure!(
            max_num_batched_tokens > 0,
            "max-num-batched-tokens must be positive"
        );
        let defaults = crate::native_sampling::defaults(&dir)?;
        eprintln!(
            "native sampling defaults: temperature={} top_k={} top_p={} presence_penalty={}",
            defaults.temperature, defaults.top_k, defaults.top_p, defaults.presence_penalty
        );
        let context = options.max_model_len;
        let (tx, mut rx) = mpsc::channel::<Job>(options.max_num_seqs * 4);
        let (ready_tx, ready_rx) = oneshot::channel();
        let (done, finished) = oneshot::channel();
        let active: Active = Arc::new(Mutex::new(std::collections::BTreeMap::new()));
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
                    pangu_native::with_serving_engine(&plan, &dir, &library, device, &options, |engine| {
                        eprintln!(
                            "native program {} weights {} context {}",
                            engine.program_key, engine.weight_digest, context
                        );
                        let memory=&engine.memory_profile;
                        eprintln!("HBM profile passed: context={} weights={} non_weight_observed={} KV={} temporary={} workspace={} peak={} remaining_budget={} bytes",
                            context,memory.resident_weight_bytes,memory.non_weight_observed_bytes,memory.kv_cache_bytes,memory.temporary_bytes,memory.workspace_bytes,memory.observed_peak_bytes,memory.remaining_budget_bytes);
                        worker_health.store(true, Ordering::Release);
                        if ready.take().unwrap().send(Ok(())).is_err() {
                            return Ok(());
                        }
                        worker::run(engine, &mut rx, &worker_active, &worker_stop, max_num_batched_tokens)
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
            defaults,
        }))
    }
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
    fn default_sampling_params(
        &self,
    ) -> Option<vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams> {
        Some(self.defaults.clone())
    }
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            max_model_len: self.context as u32,
            model_dtype: ModelDtype::BFloat16,
            version: "0.25.1 / IronPangu native continuous batching".into(),
            healthy: self.healthy.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire),
        }
    }
    async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        if !self.metadata().healthy {
            return Err(error("native engine is not ready"));
        }
        crate::native_sampling::validate(&req)?;
        validate_capacity(&req, self.context)?;
        let mut active = self.active.lock().unwrap();
        if active.contains_key(&req.request_id) {
            return Err(error("duplicate active request id"));
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
        active.insert(id.clone(), cancel.clone());
        Ok(GenerateOutputStream::from_stream(id, Output { rx, cancel }))
    }
    async fn abort(&self, ids: &[String]) -> Result<()> {
        let active = self.active.lock().unwrap();
        for id in ids {
            if let Some(cancel) = active.get(id) {
                cancel.store(true, Ordering::Release);
            }
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
