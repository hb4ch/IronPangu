use crate::{MAX_BYTES, ir::PhysicalPlan, verify};
use inferfabric_model::{Result, invalid};
use sha2::{Digest, Sha256};
const MAGIC: &[u8; 8] = b"IFPLAN01";
pub fn encode(plan: &PhysicalPlan) -> Result<Vec<u8>> {
    verify(plan)?;
    let payload = serde_json::to_vec(plan).map_err(|e| invalid(e.to_string()))?;
    if payload.len() > MAX_BYTES {
        return Err(invalid("plan binary exceeds 64 MiB"));
    }
    let mut bytes = MAGIC.to_vec();
    bytes.extend((payload.len() as u64).to_le_bytes());
    bytes.extend(Sha256::digest(&payload));
    bytes.extend(payload);
    Ok(bytes)
}
pub fn decode(bytes: &[u8]) -> Result<PhysicalPlan> {
    if bytes.len() < 48 || &bytes[..8] != MAGIC {
        return Err(invalid("invalid physical binary magic/header"));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    if len > MAX_BYTES as u64 || len != (bytes.len() - 48) as u64 {
        return Err(invalid("physical binary length mismatch"));
    }
    let digest = Sha256::digest(&bytes[48..]);
    if bytes[16..48] != digest[..] {
        return Err(invalid("physical binary checksum mismatch"));
    }
    let plan: PhysicalPlan =
        serde_json::from_slice(&bytes[48..]).map_err(|e| invalid(e.to_string()))?;
    verify(&plan)?;
    Ok(plan)
}
pub fn manifest(bytes: &[u8]) -> Result<serde_json::Value> {
    let plan = decode(bytes)?;
    Ok(
        serde_json::json!({"format":"IFPLAN01", "target":plan.target,
        "payload_bytes":bytes.len()-48, "sha256": bytes[16..48].iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "kernel_abi":"built-in CPU f32 v1", "native_device_objects":[],
        "requires_source":false, "requires_planner":false,
        "runtime_bindings":["input arrays", "owned state", "arena base"],
        "state_commit":"simultaneous after all steps succeed", "device_qualified":false}),
    )
}
