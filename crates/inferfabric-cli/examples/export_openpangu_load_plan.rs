//! Lower OpenPangu placement into the shared executable initialization IR.
use inferfabric_compiler::{openpangu, write_atomic};
use inferfabric_model::{Result, invalid};
use std::path::Path;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 && !(a.len() == 4 && a[3] == "--allow-stale-index-size") {
        return Err(invalid(
            "usage: export_openpangu_load_plan CHECKPOINT DSL OUTPUT_DIR [--allow-stale-index-size]",
        ));
    }
    let model = openpangu::inspect_checkpoint_with_options(Path::new(&a[0]), a.len() == 4)?;
    let spec = inferfabric_dsl::openpangu::parse(&std::fs::read_to_string(&a[1])?)?;
    let plan = openpangu::parallel::compile(&model, &spec)?;
    std::fs::create_dir_all(&a[2])?;
    for rank in &plan.ranks {
        let initialization = openpangu::parallel::initialization_plan(&plan, rank.rank)?;
        write_atomic(
            &Path::new(&a[2]).join(format!("rank-{}.bin", rank.rank)),
            &inferfabric_compiler::initialization::encode(&initialization)?,
        )?;
        write_atomic(
            &Path::new(&a[2]).join(format!("rank-{}.init.json", rank.rank)),
            &serde_json::to_vec_pretty(&initialization).unwrap(),
        )?;
    }
    write_atomic(
        &Path::new(&a[2]).join("plan.json"),
        &serde_json::to_vec_pretty(&plan).unwrap(),
    )?;
    Ok(())
}
