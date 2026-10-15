//! Export one complete MoE layer, with bounded staging and one metadata audit.
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
fn payload(root: &Path, t: &TensorDescriptor) -> Result<Vec<u8>> {
    if t.bytes > 128 * 1024 * 1024 {
        return Err(invalid("tensor exceeds layer-export staging bound"));
    }
    let p = root.join(&t.shard).canonicalize()?;
    if p.parent() != Some(root) {
        return Err(invalid("shard escapes checkpoint"));
    }
    let mut f = File::open(p)?;
    f.seek(SeekFrom::Start(t.offset))?;
    let mut bytes = vec![0; t.bytes as usize];
    f.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn floats(bytes: &[u8], dtype: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = match dtype {
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| f32::from_bits(u32::from(u16::from_le_bytes(*v)) << 16))
            .collect(),
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        _ => return Err(invalid("expected floating tensor")),
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite payload"));
    }
    Ok(values)
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 4 && !(a.len() == 5 && a[4] == "--allow-stale-index-size") {
        return Err(invalid(
            "usage: export_openpangu_moe CHECKPOINT LAYER OUTPUT_DIR REPORT.json [--allow-stale-index-size]",
        ));
    }
    let layer: usize = a[1].parse().map_err(|_| invalid("invalid layer"))?;
    let root = Path::new(&a[0]).canonicalize()?;
    let m = openpangu::inspect_checkpoint_with_options(&root, a.len() == 5)?;
    if !m.layers.get(layer).is_some_and(|l| l.feed_forward == "moe") {
        return Err(invalid("expected main-model MoE layer"));
    }
    let b = openpangu::binding::bind(&m)?;
    let output = Path::new(&a[2]);
    std::fs::create_dir_all(output)?;
    let prefix = format!("model.layers.{layer}.mlp.");
    let mut reports = vec![];
    for (name, binding) in &b.bindings {
        let Some(short) = name.strip_prefix(&prefix) else {
            continue;
        };
        if short.ends_with(".weight_scale") {
            continue;
        }
        let t = &binding.tensor;
        let bytes = payload(&root, t)?;
        let mut report = serde_json::json!({"tensor":name,"shape":t.shape,"source_dtype":t.dtype,"source_sha256":sha(&bytes)});
        let mut out = if t.dtype == "I8" {
            if t.shape.len() != 2 {
                return Err(invalid("expected quantized matrix"));
            }
            let scale = &b
                .bindings
                .get(&name.replace(".weight", ".weight_scale"))
                .ok_or_else(|| invalid("missing scale"))?
                .tensor;
            let scale_bytes = payload(&root, scale)?;
            let scales = floats(&scale_bytes, &scale.dtype)?;
            if scales.iter().any(|v| *v <= 0.) {
                return Err(invalid("invalid quantized scale"));
            }
            let mut out = b"IFW8A801".to_vec();
            for d in &t.shape {
                out.extend(d.to_le_bytes());
            }
            out.extend(&bytes);
            for v in scales {
                out.extend(v.to_le_bytes());
            }
            report["scale_sha256"] = sha(&scale_bytes).into();
            out
        } else {
            let values = floats(&bytes, &t.dtype)?;
            let mut out = b"IFF32T01".to_vec();
            out.extend((t.shape.len() as u64).to_le_bytes());
            for d in &t.shape {
                out.extend(d.to_le_bytes());
            }
            for v in values {
                out.extend(v.to_le_bytes());
            }
            out
        };
        let file = format!("{}.bin", short.strip_suffix(".weight").unwrap_or(short));
        report["file"] = file.clone().into();
        report["fixture_sha256"] = sha(&out).into();
        write_atomic(&output.join(file), &out)?;
        out.clear();
        reports.push(report);
    }
    let report = serde_json::json!({"layer":layer,"checkpoint_metadata_sha256":b.checkpoint_metadata_sha256,"fixtures":reports});
    write_atomic(
        Path::new(&a[3]),
        &serde_json::to_vec_pretty(&report).unwrap(),
    )?;
    println!("exported {} tensors for layer {}", reports.len(), layer);
    Ok(())
}
