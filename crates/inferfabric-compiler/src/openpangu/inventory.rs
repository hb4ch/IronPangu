use super::{MAX_METADATA, json};
use crate::checkpoint::sha;
use inferfabric_model::{Result, invalid};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};
#[derive(Debug, Clone, Serialize)]
pub struct TensorDescriptor {
    pub shard: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Debug, Serialize)]
pub struct Shard {
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub header_sha256: String,
}
#[derive(Debug, Serialize)]
pub struct Inventory {
    pub scope: &'static str,
    pub payload_verified: bool,
    pub stored_tensor_bytes: u64,
    pub index_declared_bytes: Option<u64>,
    pub index_size_matches: bool,
    pub bytes_by_dtype: BTreeMap<String, u64>,
    pub tensor_count: usize,
    pub tensors: BTreeMap<String, TensorDescriptor>,
    pub shards: BTreeMap<String, Shard>,
    pub index_sha256: Option<String>,
    pub auxiliary_sha256: BTreeMap<String, String>,
    pub metadata_sha256: String,
    pub capacity_note: &'static str,
}
fn dtype_bytes(s: &str) -> Result<u64> {
    match s {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Ok(1),
        "BF16" | "F16" | "I16" | "U16" => Ok(2),
        "F32" | "I32" | "U32" => Ok(4),
        "F64" | "I64" | "U64" => Ok(8),
        _ => Err(invalid(format!("unsupported stored dtype {s}"))),
    }
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("metadata arithmetic overflow"))
}
pub fn inspect(root: &Path, allow_stale_index_size: bool) -> Result<Inventory> {
    let mut index_hash = None;
    let mut expected_total = None;
    let map: Option<BTreeMap<String, String>> =
        if root.join("model.safetensors.index.json").exists() {
            let b = json::read(&json::local_file(root, "model.safetensors.index.json")?)?;
            let v = json::parse(&b)?;
            index_hash = Some(sha(&b));
            if let Some(n) = v.get("metadata").and_then(|m| m.get("total_size")) {
                expected_total = Some(
                    n.as_u64()
                        .ok_or_else(|| invalid("invalid index total_size"))?,
                )
            }
            let m: BTreeMap<String, String> = serde_json::from_value(v["weight_map"].clone())
                .map_err(|_| invalid("invalid weight_map"))?;
            if m.is_empty() || m.len() > 1_000_000 {
                return Err(invalid("invalid weight_map count"));
            }
            Some(m)
        } else {
            None
        };
    let names: BTreeSet<String> = map
        .as_ref()
        .map(|m| m.values().cloned().collect())
        .unwrap_or_else(|| BTreeSet::from(["model.safetensors".into()]));
    if names.len() > 4096 {
        return Err(invalid("too many shards"));
    }
    let mut result = Inventory {
        scope: "stored tensor inventory; no architecture binding or runtime placement",
        payload_verified: false,
        stored_tensor_bytes: 0,
        index_declared_bytes: expected_total,
        index_size_matches: true,
        bytes_by_dtype: BTreeMap::new(),
        tensor_count: 0,
        tensors: BTreeMap::new(),
        shards: BTreeMap::new(),
        index_sha256: index_hash,
        auxiliary_sha256: BTreeMap::new(),
        metadata_sha256: String::new(),
        capacity_note: "Stored checkpoint bytes are not peak HBM or a device-count estimate. Include rank replication, post-load transforms, staging, cache/state, activation, workspace, communication and graph reserves separately.",
    };
    let mut header_budget = 0;
    for name in names {
        if !name.ends_with(".safetensors") {
            return Err(invalid("index must name safetensors files"));
        }
        let mut f = File::open(json::local_file(root, &name)?)?;
        let size = f.metadata()?.len();
        let mut prefix = [0u8; 8];
        f.read_exact(&mut prefix)?;
        let n = u64::from_le_bytes(prefix);
        let base = add(n, 8)?;
        if n == 0 || n > MAX_METADATA || base > size {
            return Err(invalid("invalid safetensors header length"));
        }
        header_budget = add(header_budget, n)?;
        if header_budget > 256 * 1024 * 1024 {
            return Err(invalid("aggregate headers exceed 256 MiB"));
        }
        let mut header = vec![0; n as usize];
        f.read_exact(&mut header)?;
        let v = json::parse(&header)?;
        let obj = v
            .as_object()
            .ok_or_else(|| invalid("header must be object"))?;
        let mut ranges = vec![];
        for (key, t) in obj {
            if key == "__metadata__" {
                if !t
                    .as_object()
                    .is_some_and(|m| m.values().all(Value::is_string))
                {
                    return Err(invalid("invalid safetensors metadata"));
                }
                continue;
            }
            if key.is_empty() {
                return Err(invalid("empty tensor name"));
            }
            let dtype = t["dtype"]
                .as_str()
                .ok_or_else(|| invalid("missing dtype"))?;
            let shape: Vec<u64> =
                serde_json::from_value(t["shape"].clone()).map_err(|_| invalid("invalid shape"))?;
            if shape.len() > 16 {
                return Err(invalid("tensor rank exceeds 16"));
            }
            let bytes = shape.iter().try_fold(dtype_bytes(dtype)?, |n, d| {
                n.checked_mul(*d)
                    .ok_or_else(|| invalid("tensor size overflow"))
            })?;
            let offsets: [u64; 2] = serde_json::from_value(t["data_offsets"].clone())
                .map_err(|_| invalid("invalid offsets"))?;
            if offsets[1].checked_sub(offsets[0]) != Some(bytes) || add(base, offsets[1])? > size {
                return Err(invalid(format!("invalid tensor range {key}")));
            }
            if map.as_ref().is_some_and(|m| m.get(key) != Some(&name)) {
                return Err(invalid("index/header ownership mismatch"));
            }
            let tensor = TensorDescriptor {
                shard: name.clone(),
                dtype: dtype.into(),
                shape,
                offset: add(base, offsets[0])?,
                bytes,
            };
            if result.tensors.insert(key.clone(), tensor).is_some() {
                return Err(invalid("duplicate tensor across shards"));
            }
            if result.tensors.len() > 1_000_000 {
                return Err(invalid("too many tensors"));
            }
            ranges.push(offsets);
            result.stored_tensor_bytes = add(result.stored_tensor_bytes, bytes)?;
            let count = result.bytes_by_dtype.entry(dtype.into()).or_default();
            *count = add(*count, bytes)?;
        }
        ranges.sort();
        let mut cursor = 0;
        for [a, b] in ranges {
            if a != cursor {
                return Err(invalid("tensor ranges overlap or have holes"));
            }
            cursor = b;
        }
        if add(base, cursor)? != size {
            return Err(invalid("unindexed trailing shard bytes"));
        }
        result.shards.insert(
            name,
            Shard {
                file_bytes: size,
                header_bytes: n,
                header_sha256: sha(&header),
            },
        );
    }
    result.tensor_count = result.tensors.len();
    if result.tensor_count == 0 || map.as_ref().is_some_and(|m| m.len() != result.tensor_count) {
        return Err(invalid("checkpoint index size/count mismatch"));
    }
    result.index_size_matches = expected_total.is_none_or(|n| n == result.stored_tensor_bytes);
    if !result.index_size_matches && !allow_stale_index_size {
        return Err(invalid(format!(
            "checkpoint index total_size mismatch: declared {:?}, measured {}; metadata-only audit can explicitly allow stale index size",
            expected_total, result.stored_tensor_bytes
        )));
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ] {
        if root.join(name).exists() {
            let limit = if name == "tokenizer.json" {
                64 * 1024 * 1024
            } else {
                MAX_METADATA
            };
            let b = json::read_bounded(&json::local_file(root, name)?, limit)?;
            result.auxiliary_sha256.insert(name.into(), sha(&b));
        }
    }
    result.metadata_sha256 = sha(&serde_json::to_vec(&result).map_err(|e| invalid(e.to_string()))?);
    Ok(result)
}
