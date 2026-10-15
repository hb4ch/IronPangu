//! Portable IFINIT01 encoding shared by every model's initialization lowering.
use inferfabric_ir::initialization::{InitializationPlan, Instruction};
use inferfabric_model::Result;
fn string(out: &mut Vec<u8>, value: &str) {
    out.extend((value.len() as u32).to_le_bytes());
    out.extend(value.as_bytes());
}
pub fn encode(plan: &InitializationPlan) -> Result<Vec<u8>> {
    plan.validate()?;
    let mut out = b"IFINIT01".to_vec();
    out.extend(plan.rank.to_le_bytes());
    out.extend(plan.device.to_le_bytes());
    string(&mut out, &plan.checkpoint_metadata_sha256);
    out.extend((plan.parameters.len() as u32).to_le_bytes());
    for p in &plan.parameters {
        out.extend(p.id.to_le_bytes());
        string(&mut out, &p.name);
        out.extend((p.dtype as u32).to_le_bytes());
        out.extend((p.shape.len() as u32).to_le_bytes());
        for d in &p.shape {
            out.extend(d.to_le_bytes());
        }
    }
    out.extend((plan.instructions.len() as u32).to_le_bytes());
    for op in &plan.instructions {
        match op {
            Instruction::LoadSlice {
                destination,
                shard,
                offset,
                bytes,
                source_dtype,
            } => {
                out.extend(1u32.to_le_bytes());
                out.extend(destination.to_le_bytes());
                string(&mut out, shard);
                out.extend(offset.to_le_bytes());
                out.extend(bytes.to_le_bytes());
                out.extend((*source_dtype as u32).to_le_bytes());
            }
        }
    }
    Ok(out)
}
