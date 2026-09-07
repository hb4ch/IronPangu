//! Checkpoint metadata validation. Tensor payloads are never read by this module.
use pangu_model::{Result, invalid};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Component, Path},
};

const MAX_JSON: u64 = 16 * 1024 * 1024;
pub const PREFIX: &str = "model.language_model.";
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F32,
}
impl Dtype {
    pub fn bytes(self) -> u64 {
        match self {
            Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Weight {
    pub shard: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute file offset, including the safetensors prefix and header.
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Shard {
    pub bytes: u64,
    pub header_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub config_sha256: String,
    pub index_sha256: String,
    pub shards: BTreeMap<String, Shard>,
    pub weights: BTreeMap<String, Weight>,
    pub excluded: BTreeMap<String, usize>,
    /// Metadata fingerprint only: changing payload bytes requires a separate digest at upload.
    pub metadata_sha256: String,
}
pub fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))
}
fn bounded(path: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    (&mut file).take(MAX_JSON + 1).read_to_end(&mut data)?;
    if data.len() as u64 > MAX_JSON {
        return Err(invalid("metadata exceeds 16 MiB"));
    }
    Ok(data)
}
fn local_file(root: &Path, name: &str) -> Result<std::path::PathBuf> {
    // Cross-platform validation, including Windows paths when compiled on Linux.
    if name.contains(['/', '\\', ':'])
        || name.is_empty()
        || !matches!(
            Path::new(name).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
    {
        return Err(invalid(format!("invalid checkpoint filename {name}")));
    }
    let path = root.join(name).canonicalize()?;
    if path.parent() != Some(root) {
        return Err(invalid("checkpoint symlink escapes directory"));
    }
    Ok(path)
}
/// Validate the deliberately narrow, versioned Qwen3.5-2B text math contract.
pub fn validate_config(c: &Value) -> Result<()> {
    let expected = serde_json::json!({
        "model_type":"qwen3_5", "architectures":["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings":true
    });
    for (k, v) in expected.as_object().unwrap() {
        if c.get(k) != Some(v) {
            return Err(invalid(format!("unsupported config {k}")));
        }
    }
    let t = &c["text_config"];
    let expected = serde_json::json!({
        "model_type":"qwen3_5_text", "hidden_size":2048,"intermediate_size":6144,
        "vocab_size":248320,"num_hidden_layers":24,"num_attention_heads":8,
        "num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,
        "linear_num_value_heads":16,"linear_key_head_dim":128,"linear_value_head_dim":128,
        "linear_conv_kernel_dim":4,"attention_bias":false,"attn_output_gate":true,
        "hidden_act":"silu","dtype":"bfloat16","mamba_ssm_dtype":"float32",
        "tie_word_embeddings":true,"mlp_only_layers":[],"full_attention_interval":4,
        "max_position_embeddings":262144
    });
    for (k, v) in expected.as_object().unwrap() {
        if t.get(k) != Some(v) {
            return Err(invalid(format!("unsupported text_config.{k}")));
        }
    }
    if t["rms_norm_eps"].as_f64() != Some(1e-6) || t["attention_dropout"].as_f64() != Some(0.0) {
        return Err(invalid("unsupported norm epsilon or attention dropout"));
    }
    let rope = &t["rope_parameters"];
    if rope["rope_type"] != "default"
        || rope["rope_theta"].as_f64() != Some(10000000.0)
        || rope["partial_rotary_factor"].as_f64() != Some(0.25)
        || rope["mrope_interleaved"] != true
        || rope["mrope_section"] != serde_json::json!([11, 11, 10])
    {
        return Err(invalid("unsupported RoPE configuration"));
    }
    let expected: Vec<_> = (0..24)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    if t["layer_types"] != serde_json::json!(expected) {
        return Err(invalid("unsupported layer schedule"));
    }
    Ok(())
}
pub fn schema() -> BTreeMap<String, (Dtype, Vec<usize>)> {
    let mut out = BTreeMap::new();
    let mut add = |name: String, dtype, shape: &[usize]| {
        out.insert(format!("{PREFIX}{name}"), (dtype, shape.to_vec()));
    };
    add("embed_tokens.weight".into(), Dtype::BF16, &[248320, 2048]);
    add("norm.weight".into(), Dtype::BF16, &[2048]);
    for i in 0..24 {
        let mut layer =
            |name: &str, dtype, shape: &[usize]| add(format!("layers.{i}.{name}"), dtype, shape);
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            layer(name, Dtype::BF16, &[2048]);
        }
        for name in ["mlp.gate_proj.weight", "mlp.up_proj.weight"] {
            layer(name, Dtype::BF16, &[6144, 2048]);
        }
        layer("mlp.down_proj.weight", Dtype::BF16, &[2048, 6144]);
        if i % 4 == 3 {
            for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                layer(name, Dtype::BF16, &[256]);
            }
            layer("self_attn.q_proj.weight", Dtype::BF16, &[4096, 2048]);
            for name in ["self_attn.k_proj.weight", "self_attn.v_proj.weight"] {
                layer(name, Dtype::BF16, &[512, 2048]);
            }
            layer("self_attn.o_proj.weight", Dtype::BF16, &[2048, 2048]);
        } else {
            layer("linear_attn.A_log", Dtype::F32, &[16]);
            layer("linear_attn.norm.weight", Dtype::F32, &[128]);
            layer("linear_attn.dt_bias", Dtype::BF16, &[16]);
            layer("linear_attn.conv1d.weight", Dtype::BF16, &[6144, 1, 4]);
            for name in [
                "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight",
            ] {
                layer(name, Dtype::BF16, &[16, 2048]);
            }
            layer("linear_attn.in_proj_qkv.weight", Dtype::BF16, &[6144, 2048]);
            for name in [
                "linear_attn.in_proj_z.weight",
                "linear_attn.out_proj.weight",
            ] {
                layer(name, Dtype::BF16, &[2048, 2048]);
            }
        }
    }
    out
}
/// Read only config, index, and bounded safetensors headers; validate every file range.
pub fn inspect(directory: &Path) -> Result<Checkpoint> {
    let root = directory.canonicalize()?;
    let config = bounded(&local_file(&root, "config.json")?)?;
    validate_config(&json(&config)?)?;
    let index_bytes = bounded(&local_file(&root, "model.safetensors.index.json")?)?;
    let index = json(&index_bytes)?;
    let map: BTreeMap<String, String> =
        serde_json::from_value(index["weight_map"].clone()).map_err(|e| invalid(e.to_string()))?;
    if map.is_empty() {
        return Err(invalid("empty weight map"));
    }
    let mut all = BTreeMap::new();
    let mut shards = BTreeMap::new();
    let mut total = 0u64;
    for name in map.values() {
        if shards.contains_key(name) {
            continue;
        }
        let path = local_file(&root, name)?;
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let n = u64::from_le_bytes(prefix);
        if n == 0 || n > MAX_JSON || n.checked_add(8).is_none_or(|end| end > file_size) {
            return Err(invalid("invalid safetensors header size"));
        }
        let mut bytes = vec![0; n as usize];
        file.read_exact(&mut bytes)?;
        let tensors = parse_header(&bytes, name, 8 + n, file_size)?;
        for (key, weight) in tensors {
            if map.get(&key) != Some(name) || all.insert(key, weight).is_some() {
                return Err(invalid("index/header mismatch or duplicate tensor"));
            }
        }
        total = total
            .checked_add(file_size - 8 - n)
            .ok_or_else(|| invalid("checkpoint size overflow"))?;
        shards.insert(
            name.clone(),
            Shard {
                bytes: file_size,
                header_sha256: sha(&bytes),
            },
        );
    }
    if all.len() != map.len() || index["metadata"]["total_size"].as_u64() != Some(total) {
        return Err(invalid("index size/count mismatch"));
    }
    let expected = schema();
    for (key, (dtype, shape)) in &expected {
        let w = all
            .get(key)
            .ok_or_else(|| invalid(format!("missing weight {key}")))?;
        if &w.dtype != dtype || &w.shape != shape {
            return Err(invalid(format!("weight dtype/shape mismatch: {key}")));
        }
    }
    let mut excluded = BTreeMap::new();
    let mut weights = BTreeMap::new();
    for (key, w) in all {
        if expected.contains_key(&key) {
            weights.insert(key, w);
        } else {
            let group = if key.starts_with("model.visual.") {
                "vision"
            } else if key.starts_with("mtp.") {
                "mtp"
            } else {
                return Err(invalid(format!("unexpected checkpoint tensor {key}")));
            };
            *excluded.entry(group.into()).or_insert(0) += 1;
        }
    }
    let mut result = Checkpoint {
        config_sha256: sha(&config),
        index_sha256: sha(&index_bytes),
        shards,
        weights,
        excluded,
        metadata_sha256: String::new(),
    };
    result.metadata_sha256 = sha(&serde_json::to_vec(&result).map_err(|e| invalid(e.to_string()))?);
    Ok(result)
}
fn parse_header(
    bytes: &[u8],
    shard: &str,
    base: u64,
    size: u64,
) -> Result<BTreeMap<String, Weight>> {
    let value = json(bytes)?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("header must be an object"))?;
    let mut out = BTreeMap::new();
    let mut ranges = vec![];
    for (key, v) in object {
        if key == "__metadata__" {
            continue;
        }
        let dtype: Dtype = serde_json::from_value(v["dtype"].clone())
            .map_err(|_| invalid(format!("unsupported dtype: {key}")))?;
        let shape: Vec<usize> =
            serde_json::from_value(v["shape"].clone()).map_err(|_| invalid("invalid shape"))?;
        let offsets: [u64; 2] = serde_json::from_value(v["data_offsets"].clone())
            .map_err(|_| invalid("invalid offsets"))?;
        let count = shape
            .iter()
            .try_fold(dtype.bytes(), |n, &d| n.checked_mul(d as u64))
            .ok_or_else(|| invalid("tensor size overflow"))?;
        if offsets[1].checked_sub(offsets[0]) != Some(count)
            || base.checked_add(offsets[1]).is_none_or(|end| end > size)
        {
            return Err(invalid(format!("invalid tensor range {key}")));
        }
        ranges.push(offsets);
        out.insert(
            key.clone(),
            Weight {
                shard: shard.into(),
                dtype,
                shape,
                offset: base + offsets[0],
                bytes: count,
            },
        );
    }
    ranges.sort();
    let mut cursor = 0;
    for [start, end] in ranges {
        if start != cursor {
            return Err(invalid("overlapping or non-contiguous tensor ranges"));
        }
        cursor = end;
    }
    if base.checked_add(cursor) != Some(size) {
        return Err(invalid("unindexed trailing tensor data"));
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_ranges_and_overflow() {
        for h in [
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,2]}}"#,
            r#"{"a":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}}"#,
            r#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
        ] {
            assert!(parse_header(h.as_bytes(), "a.safetensors", 8, 16).is_err());
        }
    }
    #[test]
    fn absolute_offsets() {
        let h = br#"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let weights = parse_header(h, "s", 100, 104).unwrap();
        assert_eq!(weights["a"].offset, 100);
    }
}

#[cfg(test)]
mod checkpoint_fixture_tests {
    use super::*;
    #[test]
    fn real_checkpoint_header_and_config_match_contract() {
        let config = json(include_bytes!("../tests/fixtures/qwen35-config.json")).unwrap();
        validate_config(&config).unwrap();
        let mut wrong = config.clone();
        wrong["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(1.0);
        assert!(validate_config(&wrong).is_err());
        let mut wrong = config;
        wrong["text_config"]["linear_num_value_heads"] = serde_json::json!(32);
        assert!(validate_config(&wrong).is_err());
        let bytes = include_bytes!("../tests/fixtures/qwen35-header.json");
        let tensors = parse_header(bytes, "model.safetensors", 76656, 4548144832 + 76656).unwrap();
        for (key, (dtype, shape)) in schema() {
            let w = &tensors[&key];
            assert_eq!(w.dtype, dtype, "{key}");
            assert_eq!(w.shape, shape, "{key}");
        }
        assert!(tensors.keys().all(|k| k.starts_with(PREFIX)
            || k.starts_with("model.visual.")
            || k.starts_with("mtp.")));
        assert!(!tensors.contains_key("lm_head.weight"));
    }
}
