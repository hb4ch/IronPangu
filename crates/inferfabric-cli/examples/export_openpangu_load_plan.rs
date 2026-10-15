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
    std::fs::create_dir_all(&a[2])?;
    if let inferfabric_dsl::authoring::ModelDefinition::Execution { parallel, graph } = definition {
        let inventory = openpangu::inspect_inventory(Path::new(&a[0]), a.len() == 4)?;
        let plans =
            inferfabric_compiler::token::initialization_plans(&graph, &parallel, &inventory)?;
        // Validate all ranks before publishing any artifact.
        let binary = graph.encode()?;
        for initialization in &plans {
            write_initialization(Path::new(&a[2]), initialization)?;
        }
        write_atomic(&Path::new(&a[2]).join("execution.bin"), &binary)?;
        write_atomic(
            &Path::new(&a[2]).join("execution.json"),
            &serde_json::to_vec_pretty(&graph).unwrap(),
        )?;
        write_atomic(
            &Path::new(&a[2]).join("execution.dot"),
            graph.dot()?.as_bytes(),
        )?;
        write_atomic(&Path::new(&a[2]).join("plan.json"), &serde_json::to_vec_pretty(&serde_json::json!({"format":"inferfabric.token-bundle.v1", "executable":true, "topology":parallel, "nodes":graph.nodes.len(), "checkpoint_metadata_sha256":inventory.metadata_sha256})).unwrap())?;
    } else {
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
