mod backend;
mod native_backend;
mod native_smoke;
mod smoke;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use vllm_server::{Config, HttpListenerMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--native-smoke-test") {
        return native_smoke::run(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("http://127.0.0.1:18081"),
        )
        .await;
    }
    if args.get(1).is_some_and(|s| s == "--smoke-test") {
        return smoke::run(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("http://127.0.0.1:18080"),
        )
        .await;
    }
    anyhow::ensure!(
        args.len() >= 4,
        "usage: --mock DSL TOKENIZER [PORT] | --native DSL CHECKPOINT LIB DEVICE [PORT]"
    );
    let (engine, name, port_index): (Arc<dyn vllm_llm::GenerationBackend>, &str, usize) =
        match args[1].as_str() {
            "--mock" => {
                let spec = pangu_dsl::parse(&std::fs::read_to_string(&args[2])?)?;
                let artifact = pangu_compiler::compile(&spec, &pangu_ir::Target::mock())?;
                (backend::Backend::mock(artifact).await?, "ironpangu-mock", 4)
            }
            "--native" => {
                anyhow::ensure!(
                    args.len() >= 6,
                    "--native requires DSL CHECKPOINT LIB DEVICE [PORT]"
                );
                let spec = pangu_dsl::parse_checkpoint(&std::fs::read_to_string(&args[2])?)?;
                let checkpoint =
                    pangu_compiler::checkpoint::inspect(std::path::Path::new(&args[3]))?;
                let plan = pangu_compiler::bound::compile(&spec, checkpoint)?;
                (
                    native_backend::Backend::native(
                        plan,
                        args[3].clone().into(),
                        args[4].clone().into(),
                        args[5].parse()?,
                    )
                    .await?,
                    "ironpangu-qwen35",
                    6,
                )
            }
            _ => anyhow::bail!("select --mock or --native"),
        };
    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        signal.cancel();
    });
    let config = Config {
        transport_mode: vllm_engine_core_client::EngineCoreClientConfig::new_single(
            "unused-in-process",
        )
        .transport_mode,
        coordinator_mode: vllm_server::CoordinatorMode::None,
        model: args[3].clone(),
        served_model_name: vec![name.into()],
        listener_mode: HttpListenerMode::BindTcp {
            host: "127.0.0.1".into(),
            port: args
                .get(port_index)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(8000),
        },
        tool_call_parser: Default::default(),
        reasoning_parser: Default::default(),
        renderer: Default::default(),
        language_model_only: true,
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: Default::default(),
        max_logprobs: Some(0),
        api_server_options: Default::default(),
        cors: Default::default(),
        tls: None,
        api_keys: vec![],
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
        keep_alive_timeout: Duration::from_secs(5),
        profiler: None,
    };
    eprintln!("vLLM Rust frontend 0.25.1; {name}; loopback only");
    let result = vllm_server::serve_with_backend(
        config,
        shutdown,
        engine.clone() as Arc<dyn vllm_llm::GenerationBackend>,
    )
    .await;
    vllm_llm::GenerationBackend::shutdown(engine.as_ref()).await?;
    result
}
