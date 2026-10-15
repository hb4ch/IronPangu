//! Export a bound floating-point checkpoint tensor for native qualification.
use inferfabric_compiler::{checkpoint::sha, openpangu, write_atomic};
use inferfabric_model::{Result, invalid};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 3 && !(a.len() == 4 && a[3] == "--allow-stale-index-size") {
        return Err(invalid(
            "usage: export_openpangu_tensor CHECKPOINT TENSOR OUTPUT [--allow-stale-index-size]",
        ));
    }
    let root = Path::new(&a[0]).canonicalize()?;
    let m = openpangu::inspect_checkpoint_with_options(&root, a.len() == 4)?;
    let b = openpangu::binding::bind(&m)?;
    let t = &b
        .bindings
        .get(&a[1])
        .ok_or_else(|| invalid("unknown main-model tensor"))?
        .tensor;
    if !matches!(t.dtype.as_str(), "BF16" | "F32")
        || t.bytes > 128 * 1024 * 1024
        || t.shape.is_empty()
        || t.shape.len() > 4
    {
        return Err(invalid("expected bounded BF16/F32 tensor with rank 1..4"));
    }
    let path = root.join(&t.shard).canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        return Err(invalid("shard escapes root"));
    }
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(t.offset))?;
    let mut bytes = vec![0; t.bytes as usize];
    f.read_exact(&mut bytes)?;
    let values: Vec<f32> = if t.dtype == "BF16" {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16))
            .collect()
    } else {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect()
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite tensor payload"));
    }
    let mut out = b"IFF32T01".to_vec();
    out.extend((t.shape.len() as u64).to_le_bytes());
    for d in &t.shape {
        out.extend(d.to_le_bytes());
    }
    for v in values {
        out.extend(v.to_le_bytes());
    }
    write_atomic(Path::new(&a[2]), &out)?;
    let report = serde_json::json!({"tensor":a[1],"shape":t.shape,"source_dtype":t.dtype,"source_sha256":sha(&bytes),"fixture_sha256":sha(&out),"checkpoint_metadata_sha256":b.checkpoint_metadata_sha256});
    write_atomic(
        Path::new(&format!("{}.json", a[2])),
        &serde_json::to_vec_pretty(&report).unwrap(),
    )?;
    println!("{report}");
    Ok(())
}
