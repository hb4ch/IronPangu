//! Compile audited checkpoint slices to a bounded native loader manifest.
use inferfabric_compiler::{openpangu, write_atomic};
use inferfabric_model::{Result, invalid};
use std::path::Path;
fn string(out: &mut Vec<u8>, value: &str) -> Result<()> {
    if value.len() > 4096 {
        return Err(invalid("manifest string too long"));
    }
    out.extend((value.len() as u32).to_le_bytes());
    out.extend(value.as_bytes());
    Ok(())
}
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
        let mut out = b"IFLOAD01".to_vec();
        out.extend((rank.rank as u32).to_le_bytes());
        out.extend(rank.device.to_le_bytes());
        out.extend((rank.weights.len() as u32).to_le_bytes());
        out.extend(rank.stored_weight_bytes.to_le_bytes());
        out.extend(rank.prepared_weight_bytes.to_le_bytes());
        string(&mut out, &plan.checkpoint_metadata_sha256)?;
        for (name, t) in &rank.weights {
            string(&mut out, name)?;
            string(&mut out, &t.shard)?;
            string(&mut out, &t.dtype)?;
            string(&mut out, t.transform)?;
            out.extend(t.offset.to_le_bytes());
            out.extend(t.bytes.to_le_bytes());
            out.extend((t.local_shape.len() as u32).to_le_bytes());
            for dim in &t.local_shape {
                out.extend(dim.to_le_bytes());
            }
        }
        write_atomic(
            &Path::new(&a[2]).join(format!("rank-{}.bin", rank.rank)),
            &out,
        )?;
    }
    write_atomic(
        &Path::new(&a[2]).join("plan.json"),
        &serde_json::to_vec_pretty(&plan).unwrap(),
    )?;
    Ok(())
}
