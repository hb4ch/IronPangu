mod model_inspect;
#[cfg(unix)]
mod npu_probe;
mod physical;
use inferfabric_ir::Target;
use inferfabric_model::{Result, invalid};
use inferfabric_scheduler::{Capacity, Request};
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
        Some("model-source") if args.len() == 3 => {
            let definition = inferfabric_dsl::definition(&fs::read_to_string(&args[1])?)?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[2]),
                &inferfabric_dsl::authoring::encode(&definition)?,
            )?;
            return Ok(());
        }
        Some("explain-model") => return model_inspect::run(&args),
        Some("plan" | "plan-debug" | "tune-plan" | "explain" | "execute") => {
            return physical::run(&args);
        }
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
            inferfabric_compiler::write_atomic(
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
            inferfabric_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU attention qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-layer-probe") if args.len() == 5 => {
            let device = args[3].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_layer(Path::new(&args[1]), Path::new(&args[2]), device)?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU delta layer qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-conv-probe") if args.len() == 5 => {
            let device = args[3].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_conv(Path::new(&args[1]), Path::new(&args[2]), device)?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[4]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU convolution qualification passed; report={}", args[4]);
        }
        #[cfg(unix)]
        Some("npu-delta-probe") if args.len() == 4 => {
            let device = args[2].parse().map_err(|_| invalid("invalid device ID"))?;
            let report = npu_probe::run_delta(Path::new(&args[1]), device)?;
            inferfabric_compiler::write_atomic(
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
            inferfabric_compiler::write_atomic(
                Path::new(&args[5]),
                &serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!("NPU math qualification passed; report={}", args[5]);
        }
        Some("inspect-openpangu" | "inspect-openpangu-checkpoint")
            if args.len() == 3
                || (args.len() == 4
                    && args[0] == "inspect-openpangu-checkpoint"
                    && args[3] == "--allow-stale-index-size") =>
        {
            let report = if args[0] == "inspect-openpangu" {
                inferfabric_compiler::openpangu::inspect_config(Path::new(&args[1]))?
            } else {
                inferfabric_compiler::openpangu::inspect_checkpoint_with_options(
                    Path::new(&args[1]),
                    args.len() == 4,
                )?
            };
            let bytes = serde_json::to_vec_pretty(&report).map_err(|e| invalid(e.to_string()))?;
            inferfabric_compiler::write_atomic(Path::new(&args[2]), &bytes)?;
            println!(
                "OpenPangu metadata audit: {} main layers; executable=false hardware_qualified=false; {}",
                report.layers.len(),
                args[2]
            );
        }
        Some("compile-token") if args.len() >= 4 =>
        {
            let (mut allow_stale, mut resident_sequences) = (false, None);
            let mut prefill_tokens = None;
            let mut prefill_variants = None;
            let mut shared_pages = false;
            let mut options = args[4..].iter();
            while let Some(option) = options.next() {
                match option.as_str() {
                    "--allow-stale-index-size" if !allow_stale => allow_stale = true,
                    "--shared-pages" if !shared_pages => shared_pages = true,
                    "--resident-sequences" if resident_sequences.is_none() => {
                        let count = options.next().ok_or_else(|| invalid("missing resident sequence count"))?
                            .parse::<usize>().map_err(|_| invalid("invalid resident sequence count"))?;
                        if !(1..=64).contains(&count) { return Err(invalid("resident sequences must be 1..64")); }
                        resident_sequences = Some(count);
                    }
                    "--prefill-variants" if prefill_variants.is_none() && prefill_tokens.is_none() => {
                        let counts=options.next().ok_or_else(||invalid("missing prefill variant list"))?
                            .split(',').map(|s|s.parse::<usize>().map_err(|_|invalid("invalid prefill variant count"))).collect::<Result<Vec<_>>>()?;
                        prefill_variants=Some(counts);
                    }
                    "--prefill-tokens" if prefill_tokens.is_none() && prefill_variants.is_none() => {
                        let count = options.next().ok_or_else(|| invalid("missing prefill token count"))?
                            .parse::<usize>().map_err(|_| invalid("invalid prefill token count"))?;
                        if !matches!(count, 1 | 2 | 4 | 8 | 16 | 32 | 4096) { return Err(invalid("prefill tokens must be 1,2,4,8,16,32 or 4096 (uncaptured)")); }
                        prefill_tokens = Some(count);
                    }
                    _ => return Err(invalid("unknown or duplicate compile-token option")),
                }
            }
            let inferfabric_dsl::authoring::ModelDefinition::Execution { parallel, graph } =
                inferfabric_dsl::definition(&fs::read_to_string(&args[1])?)?
            else {
                return Err(invalid("compile-token requires an explicit token graph"));
            };
            let inventory = inferfabric_compiler::openpangu::inspect_inventory(
                Path::new(&args[2]),
                allow_stale,
            )?;
            inferfabric_compiler::token::write_bundle_with_variants(
                &graph,
                &parallel,
                &inventory,
                Path::new(&args[3]),
                resident_sequences.unwrap_or(parallel.max_num_seqs),
                prefill_tokens.unwrap_or_else(|| if prefill_variants.is_some() { 1 } else { parallel.max_num_batched_tokens }),
                shared_pages,
                prefill_variants.as_deref().unwrap_or(&[]),
            )?;
            println!(
                "token graph nodes={} ranks={} output={}",
                graph.nodes.len(),
                parallel.tp,
                args[3]
            );
        }
        Some("plan-openpangu")
            if args.len() == 4 || (args.len() == 5 && args[4] == "--allow-stale-index-size") =>
        {
            let source = fs::read_to_string(&args[1])?;
            if matches!(
                inferfabric_dsl::definition(&source)?,
                inferfabric_dsl::authoring::ModelDefinition::Execution { .. }
            ) {
                return Err(invalid(
                    "plan-openpangu is placement-only; use compile-token MODEL CHECKPOINT OUTPUT_DIR for an executable graph",
                ));
            }
            let spec = inferfabric_dsl::openpangu::parse(&source)?;
            let audit = inferfabric_compiler::openpangu::inspect_checkpoint_with_options(
                Path::new(&args[2]),
                args.len() == 5,
            )?;
            let plan = inferfabric_compiler::openpangu::parallel::compile(&audit, &spec)?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[3]),
                &serde_json::to_vec_pretty(&plan).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!(
                "OpenPangu TP={} EP={} on {} ranks; {} experts/rank; executable=false",
                spec.tp,
                spec.ep,
                plan.ranks.len(),
                plan.experts_per_rank
            );
        }
        Some("bind-openpangu-checkpoint")
            if args.len() == 3 || (args.len() == 4 && args[3] == "--allow-stale-index-size") =>
        {
            let audit = inferfabric_compiler::openpangu::inspect_checkpoint_with_options(
                Path::new(&args[1]),
                args.len() == 4,
            )?;
            let contract = inferfabric_compiler::openpangu::binding::bind(&audit)?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[2]),
                &serde_json::to_vec_pretty(&contract).map_err(|e| invalid(e.to_string()))?,
            )?;
            println!(
                "Bound {} main-model tensors ({} bytes); excluded MTP={} bytes; executable=false",
                contract.bindings.len(),
                contract.bound_bytes,
                contract.mtp_bytes
            );
        }
        Some("compile-checkpoint") if args.len() == 4 => {
            let spec = inferfabric_dsl::parse_checkpoint(&fs::read_to_string(&args[1])?)?;
            let checkpoint = inferfabric_compiler::checkpoint::inspect(Path::new(&args[2]))?;
            let plan = inferfabric_compiler::bound::compile(&spec, checkpoint)?;
            let bytes = serde_json::to_vec_pretty(&plan).map_err(|e| invalid(e.to_string()))?;
            inferfabric_compiler::write_atomic(Path::new(&args[3]), &bytes)?;
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
            let spec = inferfabric_dsl::parse(&fs::read_to_string(&args[1])?)?;
            let (artifact, hit) = inferfabric_compiler::cached(
                &spec,
                &Target::mock(),
                Path::new(".inferfabric-cache"),
            )?;
            inferfabric_compiler::write_atomic(
                Path::new(&args[2]),
                &inferfabric_compiler::encode(&artifact)?,
            )?;
            println!(
                "mock artifact={} cache_hit={hit} output={}",
                artifact.key, args[2]
            );
        }
        Some("inspect") if args.len() == 2 => {
            let artifact = inferfabric_compiler::decode(&fs::read(&args[1])?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&artifact).map_err(|e| invalid(e.to_string()))?
            );
        }
        Some("demo") if args.len() == 3 => {
            let spec = inferfabric_dsl::parse(&fs::read_to_string(&args[1])?)?;
            let requests: Vec<Request> =
                serde_json::from_slice(&fs::read(&args[2])?).map_err(|e| invalid(e.to_string()))?;
            let (artifact, hit) = inferfabric_compiler::cached(
                &spec,
                &Target::mock(),
                Path::new(".inferfabric-cache"),
            )?;
            eprintln!("backend=mock numerical_inference=false cache_hit={hit}");
            let report = inferfabric_scheduler::run_mock(artifact, requests, Capacity::default())?;
            println!(
                "{}",
                serde_json::to_string_pretty(&report).map_err(|e| invalid(e.to_string()))?
            );
        }
        _ => {
            return Err(invalid(
                "usage: inferfabric compile-token MODEL CHECKPOINT OUTPUT_DIR [--allow-stale-index-size] [--resident-sequences N] [--prefill-tokens 1|2|4|8|16|32|4096 (4096 is uncaptured) | --prefill-variants 2,4,8,16,32] [--shared-pages] | plan-openpangu DSL CHECKPOINT OUTPUT.json [--allow-stale-index-size] | bind-openpangu-checkpoint CHECKPOINT_DIR OUTPUT.json [--allow-stale-index-size] | inspect-openpangu CONFIG.json OUTPUT.json | inspect-openpangu-checkpoint CHECKPOINT_DIR OUTPUT.json [--allow-stale-index-size] | explain-model TYPED-PLAN.json --html MODEL.html | plan GRAPH.json OUTPUT.ifplan [--dump-ir DIR] | explain PLAN.ifplan [--html OUTPUT.html] | execute PLAN.ifplan INPUTS.json REPORT.json | npu-model-probe DSL CHECKPOINT LIBRARY DEVICE TOKENS_JSON REPORT | npu-delta-probe LIBRARY DEVICE REPORT | npu-conv-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-layer-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-attention-probe CHECKPOINT LIBRARY DEVICE REPORT | npu-math-probe DSL CHECKPOINT LIBRARY DEVICE REPORT | compile-checkpoint MODEL.inferfabric CHECKPOINT_DIR OUTPUT.json | compile MODEL.inferfabric OUTPUT | inspect ARTIFACT | demo MODEL.inferfabric REQUESTS.json",
            ));
        }
    }
    Ok(())
}
