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
    let source = std::fs::read_to_string(&a[1])?;
    let definition = inferfabric_dsl::definition(&source)?;
    if let inferfabric_dsl::authoring::ModelDefinition::Execution { parallel, graph } = definition {
        let inventory = openpangu::inspect_inventory(Path::new(&a[0]), a.len() == 4)?;
        inferfabric_compiler::token::write_bundle(&graph, &parallel, &inventory, Path::new(&a[2]))?;
    } else {
        std::fs::create_dir_all(&a[2])?;
        let model = openpangu::inspect_checkpoint_with_options(Path::new(&a[0]), a.len() == 4)?;
        let spec = inferfabric_dsl::openpangu::parse(&source)?;
        let plan = openpangu::parallel::compile(&model, &spec)?;
        for rank in &plan.ranks {
            write_initialization(
                Path::new(&a[2]),
                &openpangu::parallel::initialization_plan(&plan, rank.rank)?,
            )?;
        }
        write_atomic(
            &Path::new(&a[2]).join("plan.json"),
            &serde_json::to_vec_pretty(&plan).unwrap(),
        )?;
        // A placement-only artifact cannot accidentally retain an old execution program.
        for name in ["execution.bin", "execution.json", "execution.dot"] {
            let path = Path::new(&a[2]).join(name);
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}
fn write_initialization(
    dir: &Path,
    initialization: &inferfabric_ir::initialization::InitializationPlan,
) -> Result<()> {
    write_atomic(
        &dir.join(format!("rank-{}.bin", initialization.rank)),
        &inferfabric_compiler::initialization::encode(initialization)?,
    )?;
    write_atomic(
        &dir.join(format!("rank-{}.init.json", initialization.rank)),
        &serde_json::to_vec_pretty(initialization).unwrap(),
    )
}
