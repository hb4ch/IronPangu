mod backend;
mod batching_smoke;
mod native_args;
mod native_backend;
mod native_sampling;
mod native_smoke;
mod smoke;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use vllm_server::{Config, HttpListenerMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--batching-smoke-test") {
        return batching_smoke::run(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("http://127.0.0.1:18083"),
        )
        .await;
    }
    if args.get(1).is_some_and(|s| s == "--sampling-smoke-test") {
        return native_smoke::sampling(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("http://127.0.0.1:18081"),
        )
        .await;
    }
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
        "usage: --mock DSL TOKENIZER [PORT] | --native DSL CHECKPOINT LIB DEVICE [PORT] [--max-model-len N] [--gpu-memory-utilization F] [--memory-profile FILE] [--profile-only] [--max-num-seqs N] [--max-num-batched-tokens N]"
    );
    let (engine, name, port): (Arc<dyn vllm_llm::GenerationBackend>, &str, u16) = match args[1]
        .as_str()
    {
        "--mock" => {
            let spec = inferfabric_dsl::parse(&std::fs::read_to_string(&args[2])?)?;
            let artifact = inferfabric_compiler::compile(&spec, &inferfabric_ir::Target::mock())?;
            (
                backend::Backend::mock(artifact).await?,
                "inferfabric-mock",
                args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(8000),
            )
        }
        "--native" => {
            anyhow::ensure!(
                args.len() >= 6,
                "--native requires DSL CHECKPOINT LIB DEVICE [PORT] [--max-model-len N] [--gpu-memory-utilization F] [--memory-profile FILE] [--profile-only] [--max-num-seqs N] [--max-num-batched-tokens N]"
            );
            let spec = inferfabric_dsl::parse_checkpoint(&std::fs::read_to_string(&args[2])?)?;
            let checkpoint =
                inferfabric_compiler::checkpoint::inspect(std::path::Path::new(&args[3]))?;
            let native_args =
                native_args::NativeArgs::parse(&args[6..], spec.max_context(), args[5].parse()?)?;
            let plan = inferfabric_compiler::bound::compile(&spec, checkpoint)?;
            let profile_path = native_args.startup.profile_path.clone();
            let engine = native_backend::Backend::native(
                plan,
                args[3].clone().into(),
                args[4].clone().into(),
                args[5].parse()?,
                native_args.startup,
                native_args.max_num_batched_tokens,
            )
            .await?;
            if native_args.profile_only {
                vllm_llm::GenerationBackend::shutdown(engine.as_ref()).await?;
                if let Some(path) = profile_path {
                    println!("{}", std::fs::read_to_string(path)?);
                }
                return Ok(());
            }
            (engine, "inferfabric-qwen35", native_args.port)
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
            port,
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
