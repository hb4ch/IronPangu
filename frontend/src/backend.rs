use futures::Stream;
use inferfabric_scheduler::{
    Capacity, Request,
    live::{Event, Finish, LiveScheduler},
};
use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use tokio::sync::{mpsc, oneshot};
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_llm::{
    BackendMetadata, Error, GenerateOutput, GenerateOutputStream, GeneratePromptInfo,
    GenerateRequest, GenerationBackend, Result,
};

fn error(message: impl Into<String>) -> Error {
    Error::Backend {
        message: message.into(),
    }
}
enum Command {
    Submit(
        Box<GenerateRequest>,
        mpsc::Sender<Result<GenerateOutput>>,
        Arc<AtomicBool>,
        oneshot::Sender<Result<()>>,
    ),
    Abort(Vec<String>, oneshot::Sender<Result<()>>),
    Shutdown(oneshot::Sender<()>),
}
struct Output {
    id: String,
    prompt: Option<Arc<[u32]>>,
    tx: mpsc::Sender<Result<GenerateOutput>>,
    cancelled: Arc<AtomicBool>,
}
struct HealthGuard(Arc<AtomicBool>);
impl Drop for HealthGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
pub struct Backend {
    tx: mpsc::Sender<Command>,
    healthy: Arc<AtomicBool>,
    max_model_len: u32,
}
impl Backend {
    pub async fn mock(artifact: inferfabric_ir::Artifact) -> anyhow::Result<Arc<Self>> {
        let max_model_len = artifact.spec.max_context() as u32;
        let (tx, mut rx) = mpsc::channel(64);
        let (ready_tx, ready_rx) = oneshot::channel();
        let healthy = Arc::new(AtomicBool::new(false));
        let alive = healthy.clone();
        std::thread::Builder::new()
            .name("inferfabric-scheduler".into())
            .spawn(move || {
                let _health = HealthGuard(alive.clone());
                let mut scheduler = match LiveScheduler::mock(artifact, Capacity::default()) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("{e}")));
                        return;
                    }
                };
                alive.store(true, Ordering::Release);
                let _ = ready_tx.send(Ok(()));
                let mut outputs: BTreeMap<u64, Output> = BTreeMap::new();
                let mut next_id = 0u64;
                loop {
                    let first = if scheduler.is_idle() {
                        rx.blocking_recv()
                    } else {
                        rx.try_recv().ok()
                    };
                    let mut commands = first.into_iter().collect::<Vec<_>>();
                    for _ in 0..63 {
                        match rx.try_recv() {
                            Ok(c) => commands.push(c),
                            Err(_) => break,
                        }
                    }
                    let mut stop = None;
                    for cmd in commands {
                        match cmd {
                            Command::Shutdown(done) => {
                                stop = Some(done);
                                break;
                            }
                            Command::Abort(ids, done) => {
                                for output in outputs.values() {
                                    if ids.contains(&output.id) {
                                        output.cancelled.store(true, Ordering::Release);
                                    }
                                }
                                let _ = done.send(Ok(()));
                            }
                            Command::Submit(req, tx, cancelled, done) => {
                                if outputs.values().any(|o| o.id == req.request_id) {
                                    let _ = done.send(Err(error("duplicate live request ID")));
                                    continue;
                                }
                                next_id += 1;
                                let request = Request {
                                    id: next_id,
                                    prompt: req.prompt_token_ids.clone(),
                                    max_new_tokens: req.sampling_params.max_tokens as usize,
                                    submit_at: 0,
                                    cancel_at: None,
                                    eos: req.sampling_params.eos_token_id,
                                };
                                match scheduler.submit(request, req.sampling_params.stop_token_ids)
                                {
                                    Ok(()) => {
                                        outputs.insert(
                                            next_id,
                                            Output {
                                                id: req.request_id,
                                                prompt: Some(req.prompt_token_ids.into()),
                                                tx,
                                                cancelled,
                                            },
                                        );
                                        if done.send(Ok(())).is_err() {
                                            outputs
                                                .get(&next_id)
                                                .unwrap()
                                                .cancelled
                                                .store(true, Ordering::Release);
                                        }
                                    }
                                    Err(e) => {
                                        let _ = done.send(Err(error(format!("{e}"))));
                                    }
                                }
                            }
                        }
                    }
                    let ids: Vec<_> = outputs
                        .iter()
                        .filter(|(_, o)| {
                            stop.is_some()
                                || o.cancelled.load(Ordering::Acquire)
                                || o.tx.is_closed()
                        })
                        .map(|(&id, _)| id)
                        .collect();
                    for id in ids {
                        let _ = scheduler.cancel(id);
                        if let Some(mut output) = outputs.remove(&id) {
                            send(
                                &mut output,
                                Event {
                                    request: id,
                                    token: None,
                                    finish: Some(Finish::Abort),
                                },
                            );
                        }
                    }
                    if let Some(done) = stop {
                        alive.store(false, Ordering::Release);
                        drop(scheduler);
                        let _ = done.send(());
                        return;
                    }
                    if rx.is_closed() && scheduler.is_idle() {
                        break;
                    }
                    if !scheduler.is_idle() {
                        match scheduler.step() {
                            Ok(events) => {
                                for event in events {
                                    let id = event.request;
                                    let finished = event.finish.is_some();
                                    let delivered = outputs
                                        .get_mut(&id)
                                        .is_some_and(|output| send(output, event));
                                    if finished || !delivered {
                                        let _ = scheduler.cancel(id);
                                        outputs.remove(&id);
                                    }
                                }
                            }
                            Err(e) => {
                                for output in outputs.values() {
                                    let _ = output
                                        .tx
                                        .try_send(Err(error(format!("scheduler failed: {e}"))));
                                }
                                for id in outputs.keys() {
                                    let _ = scheduler.cancel(*id);
                                }
                                break;
                            }
                        }
                    }
                }
                alive.store(false, Ordering::Release);
            })?;
        ready_rx.await?.map_err(anyhow::Error::msg)?;
        Ok(Arc::new(Self {
            tx,
            healthy,
            max_model_len,
        }))
    }
}
fn send(output: &mut Output, event: Event) -> bool {
    let finish_reason = event.finish.map(|f| match f {
        Finish::Length => vllm_llm::FinishReason::Length,
        Finish::Stop => vllm_llm::FinishReason::stop_eos(),
        Finish::Abort => vllm_llm::FinishReason::Abort,
        Finish::Error(_) => vllm_llm::FinishReason::Error,
    });
    output
        .tx
        .try_send(Ok(GenerateOutput {
            request_id: output.id.clone(),
            prompt_info: output
                .prompt
                .take()
                .map(|prompt_token_ids| GeneratePromptInfo {
                    prompt_token_ids,
                    prompt_logprobs: None,
                }),
            token_ids: event.token.into_iter().collect(),
            logprobs: None,
            finish_reason,
            cached_token_count: 0,
            kv_transfer_params: None,
        }))
        .is_ok()
}
struct OutputStream {
    rx: mpsc::Receiver<Result<GenerateOutput>>,
    cancelled: Arc<AtomicBool>,
}
impl Stream for OutputStream {
    type Item = Result<GenerateOutput>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}
impl Drop for OutputStream {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
#[async_trait::async_trait]
impl GenerationBackend for Backend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            max_model_len: self.max_model_len,
            model_dtype: ModelDtype::BFloat16,
            version: "0.25.1 / InferFabric synthetic mock".into(),
            healthy: self.healthy.load(Ordering::Acquire),
        }
    }
    async fn generate(&self, req: GenerateRequest) -> Result<GenerateOutputStream> {
        if !self.metadata().healthy {
            return Err(error("engine is not ready"));
        }
        validate(&req)?;
        let id = req.request_id.clone();
        let (tx, rx) = mpsc::channel(64);
        let cancelled = Arc::new(AtomicBool::new(false));
        let (done, result) = oneshot::channel();
        self.tx
            .send(Command::Submit(Box::new(req), tx, cancelled.clone(), done))
            .await
            .map_err(|_| error("engine stopped"))?;
        result
            .await
            .map_err(|_| error("engine stopped during admission"))??;
        Ok(GenerateOutputStream::from_stream(
            id,
            OutputStream { rx, cancelled },
        ))
    }
    async fn abort(&self, ids: &[String]) -> Result<()> {
        let (done, result) = oneshot::channel();
        self.tx
            .send(Command::Abort(ids.to_vec(), done))
            .await
            .map_err(|_| error("engine stopped"))?;
        result.await.map_err(|_| error("engine stopped"))?
    }
    async fn shutdown(&self) -> Result<()> {
        if !self.healthy.load(Ordering::Acquire) {
            return Ok(());
        }
        let (done, result) = oneshot::channel();
        self.tx
            .send(Command::Shutdown(done))
            .await
            .map_err(|_| error("engine stopped"))?;
        result.await.map_err(|_| error("engine stopped"))
    }
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::Unsupported {
        message: message.into(),
    }
}
pub(crate) fn validate(req: &GenerateRequest) -> Result<()> {
    let p = &req.sampling_params;
    if req.mm_features.is_some()
        || req.lora_request.is_some()
        || req.priority != 0
        || req.data_parallel_rank.is_some()
        || req.cache_salt.is_some()
        || req.reasoning_parser_kwargs.is_some()
    {
        return Err(unsupported(
            "only text requests without LoRA, priority, or engine routing overrides are supported",
        ));
    }
    if p.temperature != 0.0
        || p.top_p != 1.0
        || p.top_k != 0
        || p.min_tokens != 0
        || p.min_p != 0.0
        || p.frequency_penalty != 0.0
        || p.presence_penalty != 0.0
        || p.repetition_penalty != 1.0
        || p.thinking_token_budget.is_some()
        || p.logprobs.is_some()
        || p.prompt_logprobs.is_some()
        || p.repetition_detection.is_some()
        || p.logit_bias.is_some()
        || p.allowed_token_ids.is_some()
        || p.bad_words_token_ids.is_some()
        || p.structured_outputs.is_some()
        || p.logprob_token_ids.is_some()
        || p.extra_args.is_some()
    {
        return Err(unsupported(
            "this backend supports greedy generation only; set temperature=0 and omit advanced sampling options",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    fn request(id: &str) -> GenerateRequest {
        GenerateRequest {
            request_id: id.into(),
            prompt_token_ids: vec![1, 2, 3, 4, 5],
            sampling_params: serde_json::from_value(
                serde_json::json!({"temperature":0.0,"max_tokens":4}),
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
    #[tokio::test]
    async fn streams_terminal_metadata_and_shutdown() {
        let spec = inferfabric_dsl::parse(include_str!("../fixtures/mock.inferfabric")).unwrap();
        let artifact =
            inferfabric_compiler::compile(&spec, &inferfabric_ir::Target::mock()).unwrap();
        let backend = Backend::mock(artifact).await.unwrap();
        let mut stream = backend.generate(request("a")).await.unwrap();
        let mut count = 0;
        let mut prompts = 0;
        let mut terminal = false;
        while let Some(item) = stream.next().await {
            let output = item.unwrap();
            count += output.token_ids.len();
            prompts += usize::from(output.prompt_info.is_some());
            terminal = output.finished();
        }
        assert_eq!((count, prompts, terminal), (4, 1, true));
        assert!(stream.next().await.is_none());
        let mut invalid = request("bad");
        invalid.sampling_params.temperature = 1.0;
        assert!(backend.generate(invalid).await.is_err());
        backend.shutdown().await.unwrap();
        assert!(!backend.metadata().healthy);
        assert!(backend.generate(request("after")).await.is_err());
    }
    #[tokio::test]
    async fn early_stream_close_is_an_error() {
        let mut stream =
            GenerateOutputStream::from_stream("closed".into(), futures::stream::empty());
        assert!(stream.next().await.unwrap().is_err());
        assert!(stream.next().await.is_none());
    }
}
