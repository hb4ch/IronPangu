use inferfabric_model::{Result, invalid};
use inferfabric_plan::{self as plan, ir::LogicalGraph};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};
fn read(path: &str) -> Result<Vec<u8>> {
    if fs::metadata(path)?.len() > (plan::MAX_BYTES + 48) as u64 {
        return Err(invalid("input file exceeds 64 MiB limit"));
    }
    Ok(fs::read(path)?)
}
fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    inferfabric_compiler::write_atomic(
        path,
        &serde_json::to_vec_pretty(value).map_err(|e| invalid(e.to_string()))?,
    )
}
pub fn run(args: &[String]) -> Result<()> {
    match args[0].as_str() {
        "plan" if args.len() == 3 || (args.len() == 5 && args[3] == "--dump-ir") => {
            let graph: LogicalGraph =
                serde_json::from_slice(&read(&args[1])?).map_err(|e| invalid(e.to_string()))?;
            let c = plan::compile(graph)?;
            let binary = plan::encode(&c.physical)?;
            inferfabric_compiler::write_atomic(Path::new(&args[2]), &binary)?;
            if args.len() == 5 {
                let dir = Path::new(&args[4]);
                fs::create_dir_all(dir)?;
                write_json(&dir.join("00-logical.json"), &c.logical)?;
                write_json(&dir.join("01-typed.json"), &c.typed)?;
                write_json(&dir.join("02-optimized.json"), &c.optimized)?;
                write_json(&dir.join("03-physical.json"), &c.physical)?;
                write_json(
                    &dir.join("04-bundle-manifest.json"),
                    &plan::manifest(&binary)?,
                )?;
            }
            println!(
                "target={} steps={} arena_bytes={} folded={} removed={} binary={}",
                c.physical.target,
                c.physical.steps.len(),
                c.physical.arena_bytes,
                c.physical.folded_nodes.len(),
                c.physical.removed_nodes.len(),
                args[2]
            );
        }
        "explain" if args.len() == 2 || (args.len() == 4 && args[2] == "--html") => {
            let p = plan::decode(&read(&args[1])?)?;
            if args.len() == 4 {
                inferfabric_compiler::write_atomic(
                    Path::new(&args[3]),
                    plan::visualize::html(&p)?.as_bytes(),
                )?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&p).map_err(|e| invalid(e.to_string()))?
            );
        }
        "execute" if args.len() == 4 => {
            let p = plan::decode(&read(&args[1])?)?;
            // The executable path never opens source, resolves symbols, or calls compile.
            let input: Value =
                serde_json::from_slice(&read(&args[2])?).map_err(|e| invalid(e.to_string()))?;
            let invocations: Vec<BTreeMap<String, Vec<f32>>> =
                serde_json::from_value(input).map_err(|e| invalid(e.to_string()))?;
            if invocations.is_empty() || invocations.len() > 1024 {
                return Err(invalid("expected 1..1024 invocation binding maps"));
            }
            let mut executor = plan::Executor::new(p)?;
            let reports = invocations
                .iter()
                .map(|inputs| executor.run(inputs))
                .collect::<Result<Vec<_>>>()?;
            write_json(Path::new(&args[3]), &reports)?;
            println!(
                "CPU physical plan executed: invocations={} report={}",
                reports.len(),
                args[3]
            );
        }
        _ => {
            return Err(invalid(
                "usage: inferfabric plan GRAPH.json OUTPUT.ifplan [--dump-ir DIR] | explain PLAN.ifplan [--html OUTPUT.html] | execute PLAN.ifplan INPUTS.json REPORT.json",
            ));
        }
    }
    Ok(())
}
