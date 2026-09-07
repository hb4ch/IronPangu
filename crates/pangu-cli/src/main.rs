#[cfg(unix)]
mod npu_probe;
use pangu_ir::Target;
use pangu_model::{Result, invalid};
use pangu_scheduler::{Capacity, Request};
use std::{env, fs, path::Path};
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        #[cfg(unix)]
        Some("npu-model-probe") if args.len() == 7 => {
            let device = args[4].parse().map_err(|_| invalid("invalid device ID"))?;
            let tokens: Vec<u32> =
                serde_json::from_slice(&fs::read(&args[5])?).map_err(|e| invalid(e.to_string()))?;
            let report = npu_probe::run_model(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                device,
                &tokens,
            )?;
            pangu_compiler::write_atomic(
                Path::new(&args[6]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!(
                "NPU complete-model graph qualification passed; report={}",
                args[6]
            );
        }
        #[cfg(unix)]
        Some("npu-attention-probe") if args.len() == 5 => {
            let device = args[3].parse().map_err(|_| invalid("invalid device ID"))?;
            let report =
                npu_probe::run_attention(Path::new(&args[1]), Path::new(&args[2]), device)?;
            pangu_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU attention qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-layer-probe") if args.len() == 5 => {
            let device = args[3].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_layer(Path::new(&args[1]), Path::new(&args[2]), device)?;
            pangu_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU delta layer qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-conv-probe") if args.len() == 5 => {
            let device = args[3].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_conv(Path::new(&args[1]), Path::new(&args[2]), device)?;
            pangu_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU convolution qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-delta-probe") if args.len() == 4 => {
            let device = args[2].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_delta(Path::new(&args[1]), device)?;
            pangu_compiler::write_atomic(
                Path::new(&args[3]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU recurrence qualification passed; report={}", args[3]);
        }
        #[cfg(unix)]
        Some("npu-math-probe") if args.len() == 6 => {
            let device = args[4].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run(
                Path::new(&args[1]),
                Path::new(&args[2]),
                Path::new(&args[3]),
                device,
            )?;
            pangu_compiler::write_atomic(
                Path::new(&args[5]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU math qualification passed; report={}", args[5]);
        }
        Some("compile-checkpoint") if args.len() == 4 => {
            let spec = pangu_dsl::parse_checkpoint(&fs::read_to_string(&args[1])?)?;
            let checkpoint = pangu_compiler::checkpoint::inspect(Path::new(&args[2]))?;
            let plan = pangu_compiler::bound::compile(&spec, checkpoint)?;
            let bytes = serde_json::to_vec_pretty(&plan).map_err(|e| invalid(e.to_string()))?;
            pangu_compiler::write_atomic(Path::new(&args[3]), &bytes)?;
            println!(
                "typed_plan={} executable=false tensors={} nodes={} weight_bytes={} output={}",
                plan.key,
                plan.checkpoint.weights.len(),
                plan.nodes.len(),
                plan.text_weight_bytes,
                args[3]
            );
        }
        Some("compile") if args.len() == 3 => {
            let spec = pangu_dsl::parse(&fs::read_to_string(&args[1])?)?;
            let (artifact, hit) =
                pangu_compiler::cached(&spec, &Target::mock(), Path::new(".pangu-cache"))?;
            pangu_compiler::write_atomic(Path::new(&args[2]), &pangu_compiler::encode(&artifact)?)?;
            println!(
                "mock artifact={} cache_hit={hit} output={}",
                artifact.key, args[2]
            );
        }
        Some("inspect") if args.len() == 2 => {
            let artifact = pangu_compiler::decode(&fs::read(&args[1])?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&artifact).map_err(|e| invalid(e.to_string()))?
            );
        }
        Some("demo") if args.len() == 3 => {
            let spec = pangu_dsl::parse(&fs::read_to_string(&args[1])?)?;
            let requests: Vec<Request> =
                serde_json::from_slice(&fs::read(&args[2])?).map_err(|e| invalid(e.to_string()))?;
            let (artifact, hit) =
                pangu_compiler::cached(&spec, &Target::mock(), Path::new(".pangu-cache"))?;
            eprintln!("backend=mock numerical_inference=false cache_hit={hit}");
            let report = pangu_scheduler::run_mock(artifact, requests, Capacity::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| invalid(e.to_string()))?
            );
        }
        _ => {
            return Err(invalid(
                "usage: pangu npu-model-probe DSL CHECKPOINT LIBRARY DEVICE TOKENS_JSON REPORT | npu-delta-probe LIBRARY DEVICE REPORT | npu-conv-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-layer-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-attention-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-math-probe DSL CHECKPOINT LIBRARY DEVICE REPORT | compile-checkpoint MODEL.pangu CHECKPOINT_DIR OUTPUT.json | compile MODEL.pangu OUTPUT | inspect ARTIFACT | demo MODEL.pangu REQUESTS.json",
            ));
        }
    }
    Ok(())
}
