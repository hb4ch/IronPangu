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
                "usage: pangu compile MODEL.pangu OUTPUT | inspect ARTIFACT | demo MODEL.pangu REQUESTS.json",
            ));
        }
    }
    Ok(())
}
