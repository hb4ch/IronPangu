//! Render a saved execution.json without checkpoint access or NPU allocation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: view_token_plan EXECUTION.json OUTPUT.html".into());
    }
    let graph = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let html = inferfabric_compiler::token_view::render(&graph)?;
    std::fs::write(&args[1], html)?;
    Ok(())
}
