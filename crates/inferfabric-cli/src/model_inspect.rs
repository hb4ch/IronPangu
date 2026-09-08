use inferfabric_model::{Result, invalid};
use std::{fs, path::Path};
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 || args[2] != "--html" {
        return Err(invalid(
            "usage: explain-model TYPED-PLAN.json --html MODEL.html",
        ));
    }
    if fs::metadata(&args[1])?.len() > 64 * 1024 * 1024 {
        return Err(invalid("model plan exceeds 64 MiB"));
    }
    let plan: inferfabric_compiler::bound::Plan =
        serde_json::from_slice(&fs::read(&args[1])?).map_err(|e| invalid(e.to_string()))?;
    let html = inferfabric_compiler::model_inspect::html(&plan)?;
    inferfabric_compiler::write_atomic(Path::new(&args[3]), html.as_bytes())?;
    println!(
        "{}: {} layers, {} operators, {} weights, {} state tensors; {}",
        plan.spec.model.name,
        plan.spec.model.layers.len(),
        plan.nodes.len(),
        plan.checkpoint.weights.len(),
        plan.states.len(),
        args[3]
    );
    Ok(())
}
