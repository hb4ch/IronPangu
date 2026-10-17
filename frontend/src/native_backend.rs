#[path = "native_worker.rs"]
mod worker;
#[path = "program_worker.rs"]
mod program_worker;
#[path = "program_cached.rs"]
mod program_cached;
#[cfg(test)]
#[path = "program_backpressure_test.rs"]
mod program_backpressure_test;
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
fn program_defaults() -> anyhow::Result<vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams> {
    Ok(serde_json::from_value(serde_json::json!({"temperature":0.0,"top_p":1.0,"top_k":0,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}))?)
}
pub struct Backend {
    tx: mpsc::Sender<Job>,
    active: Active,
    stop: Arc<AtomicBool>,
    healthy: Arc<AtomicBool>,
    finished: tokio::sync::Mutex<Option<oneshot::Receiver<()>>>,
    context: usize,
    vocab: u32,
    program_mode: bool,
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
        plan: inferfabric_compiler::bound::Plan,
        dir: PathBuf,
        library: PathBuf,
        device: i32,
        options: inferfabric_native::StartupOptions,
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
            .name("inferfabric-native".into())
            .spawn(move || {
                let _guard = Guard {
                    healthy: worker_health.clone(),
                    finished: Some(done),
                };
                let mut ready = Some(ready_tx);
                let result =
                    inferfabric_native::with_serving_engine(&plan, &dir, &library, device, &options, |engine| {
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
            vocab: 248320,
            program_mode: false,
            defaults,
        }))
    }

    pub async fn program(bundle: PathBuf, dir: PathBuf, library: PathBuf, options: crate::program_args::ProgramArgs) -> anyhow::Result<Arc<Self>> {
        let defaults = program_defaults()?;
        let manifest = inferfabric_native::program::manifest(&bundle)?;
        let chunk = inferfabric_native::program::prefill_capacity(&manifest)? as usize;
        let mut options = options;
        if options.max_num_batched_tokens.is_none() {
            options.max_num_batched_tokens = Some(options.max_num_seqs.unwrap_or(manifest.resident_sequences as usize).checked_mul(chunk).ok_or_else(||anyhow::anyhow!("token budget overflow"))?);
        }
        let (residents, context, token_budget) = options.resolve(manifest.resident_sequences as usize, manifest.variants[0].max_context as usize)?;
        eprintln!("program scheduler configured max_num_seqs={residents} max_num_batched_tokens={token_budget} max_model_len={context}; compiled_prefill_tokens={chunk}, decode_tokens=1");
        let vocab = manifest.vocab_size;
        let tokenizer_hash=if options.prefix_cache_slots.is_some() {
            let mut bytes=Vec::new();
            for name in ["tokenizer.json","tokenizer_config.json"] {
                let value=std::fs::read(dir.join(name))?;
                bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());bytes.extend(value);
            }
            inferfabric_compiler::checkpoint::sha(&bytes)
        } else {String::new()};
        let (tx, mut rx) = mpsc::channel::<Job>(residents * 4);
        let (ready_tx, ready_rx) = oneshot::channel();
        let (done, finished) = oneshot::channel();
        let active: Active = Arc::new(Mutex::new(std::collections::BTreeMap::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let healthy = Arc::new(AtomicBool::new(false));
        let (a, s, h) = (active.clone(), stop.clone(), healthy.clone());
        std::thread::Builder::new().name("inferfabric-program".into()).spawn(move || {
            let _guard = Guard { healthy: h.clone(), finished: Some(done) };
            match inferfabric_native::program::Engine::open(&bundle, &dir, &library).and_then(|engine|
                program_cached::CachedProgram::new(engine,&manifest,&options,residents,context,token_budget,tokenizer_hash)) {
                Ok(mut engine) => {
                    h.store(true, Ordering::Release);
                    if ready_tx.send(Ok(())).is_ok() {
                        if let Err(e) = program_worker::run(&mut engine, &mut rx, &a, &s, residents, token_budget) {
                            eprintln!("program worker failed: {e}");
                        }
                        h.store(false, Ordering::Release);
                    }
                }
                Err(e) => { let _ = ready_tx.send(Err(e.to_string())); }
            }
        })?;
        ready_rx.await.map_err(|_| anyhow::anyhow!("program startup thread stopped"))?.map_err(anyhow::Error::msg)?;
        Ok(Arc::new(Self { tx, active, stop, healthy, finished: tokio::sync::Mutex::new(Some(finished)), context, vocab, program_mode: true, defaults }))
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
            version: if self.program_mode { "0.25.1 / InferFabric resident request scheduler; single-token greedy invocations" } else { "0.25.1 / InferFabric native continuous batching" }.into(),
            healthy: self.healthy.load(Ordering::Acquire) && !self.stop.load(Ordering::Acquire),
        }
    }
    async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        if !self.metadata().healthy {
            return Err(Error::Unavailable { message: "native engine is not ready".into() });
        }
        if self.program_mode {crate::native_sampling::validate_with_cache_salt(&req,true)?;}
        else {crate::native_sampling::validate(&req)?;}
        if self.program_mode { program_worker::validate(&req)?; }
        validate_capacity_for(&req, self.context, self.vocab)?;
        let mut active = self.active.lock().unwrap();
        if active.contains_key(&req.request_id) {
            return Err(error("duplicate active request id"));
        }
        let id = req.request_id.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        // Program scheduling applies per-request backpressure. Its output queue
        // must not scale with a 64K (or larger future) context envelope.
        let (tx, rx) = mpsc::channel(if self.program_mode { 64 } else { self.context + 1 });
        self.tx
            .try_send(Job {
                req,
                tx,
                cancel: cancel.clone(),
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => Error::Overloaded { message: "native admission queue is full; retry later".into() },
                mpsc::error::TrySendError::Closed(_) => Error::Unavailable { message: "native engine has stopped".into() },
            })?;
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
#[cfg(test)]
fn validate_capacity(req: &GenerateRequest, context: usize) -> Result<()> {
    validate_capacity_for(req, context, 248320)
}
fn validate_capacity_for(req: &GenerateRequest, context: usize, vocab: u32) -> Result<()> {
    if req.prompt_token_ids.is_empty()
        || req.prompt_token_ids.iter().any(|&id| id >= vocab)
        || req.sampling_params.stop_token_ids.iter().chain(req.sampling_params.eos_token_id.iter()).any(|&id| id >= vocab)
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
    fn request(id: &str) -> GenerateRequest {
        GenerateRequest {
            request_id: id.into(), prompt_token_ids: vec![1],
            sampling_params: serde_json::from_value(serde_json::json!({"temperature":0,"max_tokens":16})).unwrap(),
            mm_features: None, arrival_time: None, cache_salt: None, trace_headers: None,
            priority: 0, data_parallel_rank: None, reasoning_parser_kwargs: None, lora_request: None,
        }
    }
    #[tokio::test]
    async fn admission_distinguishes_full_closed_and_unhealthy_without_leaking_ids() {
        let (tx, mut rx) = mpsc::channel(1);
        let backend = Backend {
            tx, active: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
            stop: Arc::new(AtomicBool::new(false)), healthy: Arc::new(AtomicBool::new(true)),
            finished: tokio::sync::Mutex::new(None), context: 128, vocab: 248320,
            program_mode: true, defaults: program_defaults().unwrap(),
        };
        let first = backend.generate(request("first")).await.unwrap();
        assert!(matches!(backend.generate(request("retry")).await, Err(Error::Overloaded { .. })));
        assert_eq!(backend.active.lock().unwrap().len(), 1);
        assert!(!backend.active.lock().unwrap().contains_key("retry"));
        let accepted = rx.recv().await.unwrap();
        drop(first);
        assert!(accepted.cancel.load(Ordering::Acquire));
        backend.active.lock().unwrap().remove("first");
        let second = backend.generate(request("retry")).await.unwrap();
        assert!(backend.active.lock().unwrap().contains_key("retry"));
        drop(second);
        drop(rx);
        assert!(matches!(backend.generate(request("closed")).await, Err(Error::Unavailable { .. })));
        assert!(!backend.active.lock().unwrap().contains_key("closed"));
        backend.healthy.store(false, Ordering::Release);
        assert!(matches!(backend.generate(request("unhealthy")).await, Err(Error::Unavailable { .. })));
        assert!(!backend.active.lock().unwrap().contains_key("unhealthy"));
    }
    #[test]
    fn program_defaults_match_unsigned_protocol() {
        let defaults = program_defaults().unwrap();
        assert_eq!(defaults.temperature, 0.);
        assert_eq!(defaults.top_k, 0);
        assert_eq!(defaults.repetition_penalty, 1.);
    }
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
