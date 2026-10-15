//! Export one validated W8A8 projection for native qualification; never executes a model.
use inferfabric_compiler::{
    checkpoint::sha,
    openpangu::{self, TensorDescriptor},
    write_atomic,
};
use inferfabric_model::{Result, invalid};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
fn read(root: &Path, t: &TensorDescriptor) -> Result<Vec<u8>> {
    if t.bytes > 256 * 1024 * 1024 {
        return Err(invalid("probe tensor exceeds 256 MiB"));
    }
    let root = root.canonicalize()?;
    let path = root.join(&t.shard).canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        return Err(invalid("shard escapes root"));
    }
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(t.offset))?;
    let mut bytes = vec![0; t.bytes as usize];
    f.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn main() -> Result<()> {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    if a.len() != 3 && !(a.len() == 4 && a[3] == "--allow-stale-index-size") {
        return Err(invalid(
            "usage: export_openpangu_linear CHECKPOINT MODULE OUTPUT.bin [--allow-stale-index-size]",
        ));
    }
    let root = Path::new(&a[0]);
    let m = openpangu::inspect_checkpoint_with_options(root, a.len() == 4)?;
    let b = openpangu::binding::bind(&m)?;
    let w = b
        .bindings
        .get(&format!("{}.weight", a[1]))
        .ok_or_else(|| invalid("unknown projection"))?;
    let s = b
        .bindings
        .get(&format!("{}.weight_scale", a[1]))
        .ok_or_else(|| invalid("missing scale"))?;
    if w.tensor.dtype != "I8" || w.tensor.shape.len() != 2 {
        return Err(invalid("expected W8A8 matrix"));
    }
    let weights = read(root, &w.tensor)?;
    let scales = read(root, &s.tensor)?;
    let mut out = b"IFW8A801".to_vec();
    for d in &w.tensor.shape {
        out.extend(d.to_le_bytes());
    }
    out.extend(&weights);
    let values: Vec<f32> = match s.tensor.dtype.as_str() {
        "BF16" => scales
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
            .collect(),
        "F32" => scales
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect(),
        _ => return Err(invalid("unsupported scale dtype")),
    };
    for v in values {
        if !v.is_finite() || v <= 0.0 {
            return Err(invalid("invalid quantization scale"));
        }
        out.extend(v.to_le_bytes());
    }
    write_atomic(Path::new(&a[2]), &out)?;
    let report = serde_json::json!({"module":a[1],"shape":w.tensor.shape,"checkpoint_metadata_sha256":b.checkpoint_metadata_sha256,"weight_sha256":sha(&weights),"scale_sha256":sha(&scales),"fixture_sha256":sha(&out),"scope":"one real projection; synthetic inputs; not full-model qualification"});
    write_atomic(
        Path::new(&format!("{}.json", a[2])),
        &serde_json::to_vec_pretty(&report).unwrap(),
    )?;
    println!("{report}");
    Ok(())
}
