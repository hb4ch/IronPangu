//! Render a saved execution.json without checkpoint access or NPU allocation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: view_token_plan EXECUTION.json OUTPUT.html [PHYSICAL.json]".into());
    }
    let graph = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let physical = inferfabric_compiler::token_physical::lower(&graph, &[0, 1, 2, 3, 4, 5, 6, 7])?;
    let html = inferfabric_compiler::token_view::render_with_physical(&graph, &physical)?;
    std::fs::write(&args[1], html)?;
    if args.len() == 3 {
        std::fs::write(&args[2], serde_json::to_vec_pretty(&physical)?)?;
    }
    Ok(())
}
