use inferfabric_compiler::openpangu::{self, Manifest};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
const CONFIG: &[u8] = include_bytes!("../../../examples/openpangu/92b.reference-config.json");
fn config() -> Value {
    serde_json::from_slice(CONFIG).unwrap()
}
fn audit(c: &Value) -> inferfabric_model::Result<Manifest> {
    openpangu::from_config(&serde_json::to_vec(c).unwrap())
}
struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "inferfabric-openpangu-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        fs::write(p.join("config.json"), CONFIG).unwrap();
        Self(p)
    }
    fn shard(&self, name: &str, h: Value, bytes: u64) {
        let h = serde_json::to_vec(&h).unwrap();
        let mut f = fs::File::create(self.0.join(name)).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.set_len(8 + h.len() as u64 + bytes).unwrap();
    }
    fn inspect(&self) -> inferfabric_model::Result<Manifest> {
        openpangu::inspect_checkpoint(&self.0)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap()
    }
}
#[test]
fn reference_config_has_explicit_main_and_mtp_boundary() {
    let m = openpangu::from_config(CONFIG).unwrap();
    assert_eq!(m.layers.len(), 46);
    assert_eq!(
        m.layers
            .iter()
            .filter(|l| l.attention == "sparse_mla")
            .count(),
        16
    );
    assert_eq!(
        m.layers
            .iter()
            .filter(|l| l.attention == "sliding_window_mla")
            .count(),
        30
    );
    assert_eq!(
        m.layers.iter().filter(|l| l.feed_forward == "moe").count(),
        44
    );
    assert!(m.features["mhc"] && m.features["mome"] && m.features["mtp_declared"]);
    assert!(!m.executable && !m.hardware_qualified);
    assert!(m.inventory.is_none());
    assert_eq!(
        m.config["swa_layers"].as_array().unwrap().last().unwrap(),
        46
    );
}
#[test]
fn different_dimensions_are_not_qwen_or_92b_special_cased() {
    let mut c = config();
    c["hidden_size"] = json!(8192);
    c["num_attention_heads"] = json!(128);
    c["n_routed_experts"] = json!(512);
    c["custom_unresolved_field"] = json!({"layout":"future"});
    let m = audit(&c).unwrap();
    assert_eq!(m.config, c);
    assert_eq!(m.config["n_routed_experts"], 512);
}
#[test]
fn invalid_configs_fail_early() {
    for (key, v) in [
        ("num_experts_per_tok", json!(257)),
        ("num_hidden_layers", json!(0)),
        ("first_k_dense_replace", json!(47)),
        ("use_mome", json!("false")),
        ("swa_layers", json!([1, 1])),
        ("sliding_window_list", json!([])),
        ("model_type", json!("pangu_v3_moe")),
        ("mhc_use_gamma", json!(false)),
        ("q_lora_rank", json!(0)),
        ("dsa_layers", json!([99])),
    ] {
        let mut c = config();
        c[key] = v;
        assert!(audit(&c).is_err(), "{key}");
    }
}
#[test]
fn sharing_contracts_are_recorded_and_invalid_dependencies_rejected() {
    let mut c = config();
    c["cla_explicit_mapping"] = json!([[3, 0], [6, 3]]);
    let mut modes = vec!["unique"; 46];
    modes[6] = "shared";
    c["indexer_types"] = json!(modes);
    let m = audit(&c).unwrap();
    assert_eq!(m.layers[6].cla_source_layer, Some(3));
    assert!(m.features["index_sharing_declared"]);
    for mapping in [json!([[1, 2]]), json!([[2, 0], [2, 1]]), json!([[46, 0]])] {
        c["cla_explicit_mapping"] = mapping;
        assert!(audit(&c).is_err());
    }
}
#[test]
fn duplicate_json_is_rejected() {
    assert!(
        openpangu::from_config(br#"{"model_type":"pangu_v2_moe","model_type":"pangu_v3_moe"}"#)
            .is_err()
    );
    let mut c = String::from_utf8(CONFIG.to_vec()).unwrap();
    c = c.replacen('{', "{\"nested\":{\"a\":1,\"a\":2},", 1);
    assert!(openpangu::from_config(c.as_bytes()).is_err());
}
#[test]
fn sharded_quantized_inventory_and_identity() {
    let d = Dir::new();
    d.shard(
        "a.safetensors",
        json!({"model.experts.weight":{"dtype":"I8","shape":[2,3],"data_offsets":[0,6]}}),
        6,
    );
    d.shard(
        "b.safetensors",
        json!({"model.experts.scale":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
        8,
    );
    let index = json!({"metadata":{"total_size":14},"weight_map":{"model.experts.weight":"a.safetensors","model.experts.scale":"b.safetensors"}});
    fs::write(d.0.join("model.safetensors.index.json"), index.to_string()).unwrap();
    let a = d.inspect().unwrap().inventory.unwrap();
    assert_eq!(a.stored_tensor_bytes, 14);
    assert_eq!(a.bytes_by_dtype["I8"], 6);
    assert_eq!(a.tensor_count, 2);
    assert!(!a.payload_verified);
    let b = d.inspect().unwrap().inventory.unwrap();
    assert_eq!(a.metadata_sha256, b.metadata_sha256);
    let mut bad = index;
    bad["metadata"]["total_size"] = json!(13);
    fs::write(d.0.join("model.safetensors.index.json"), bad.to_string()).unwrap();
    assert!(d.inspect().is_err());
}
#[test]
fn sparse_large_file_needs_no_weight_payload() {
    let d = Dir::new();
    let size = 1u64 << 30;
    d.shard(
        "model.safetensors",
        json!({"synthetic.weight":{"dtype":"BF16","shape":[size/2],"data_offsets":[0,size]}}),
        size,
    );
    let m = d.inspect().unwrap();
    assert_eq!(m.inventory.unwrap().stored_tensor_bytes, size);
}
#[test]
fn bad_headers_sizes_and_paths_rejected() {
    let d = Dir::new();
    for h in [
        json!({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,3]}}),
        json!({"a":{"dtype":"I4","shape":[4],"data_offsets":[0,4]}}),
        json!({"a":{"dtype":"F32","shape":[u64::MAX,2],"data_offsets":[0,4]}}),
        json!({"a":{"dtype":"U8","shape":[4],"data_offsets":[1,5]}}),
        json!({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
    ] {
        d.shard("model.safetensors", h, 5);
        assert!(d.inspect().is_err());
    }
    for path in [
        "../outside.safetensors",
        "C:\\outside.safetensors",
        "/tmp/outside.safetensors",
    ] {
        fs::write(
            d.0.join("model.safetensors.index.json"),
            json!({"weight_map":{"a":path}}).to_string(),
        )
        .unwrap();
        assert!(d.inspect().is_err());
    }
}
#[test]
fn index_must_cover_exact_header_names() {
    let d = Dir::new();
    d.shard(
        "a.safetensors",
        json!({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
        4,
    );
    fs::write(
        d.0.join("model.safetensors.index.json"),
        json!({"weight_map":{"b":"a.safetensors"}}).to_string(),
    )
    .unwrap();
    assert!(d.inspect().is_err());
}
#[cfg(unix)]
#[test]
fn escaping_symlinks_are_rejected() {
    let a = Dir::new();
    let b = Dir::new();
    b.shard(
        "model.safetensors",
        json!({"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}),
        4,
    );
    std::os::unix::fs::symlink(b.0.join("model.safetensors"), a.0.join("model.safetensors"))
        .unwrap();
    assert!(a.inspect().is_err());
}

#[test]
fn released_openpangu_v2_config_preserves_identity() {
    let mut c = config();
    c["model_type"] = json!("openpangu_v2");
    c["architectures"] = json!(["OpenPanguV2ForCausalLM"]);
    let m = audit(&c).unwrap();
    assert_eq!(m.config, c);
    assert_eq!(m.layers.len(), 46);
    assert!(!m.executable);
}
#[test]
fn stale_index_opt_in_does_not_relax_tensor_ownership() {
    let d = Dir::new();
    d.shard(
        "one.safetensors",
        json!({"a":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}}),
        4,
    );
    fs::write(
        d.0.join("model.safetensors.index.json"),
        json!({"metadata":{"total_size":8},"weight_map":{"a":"one.safetensors"}}).to_string(),
    )
    .unwrap();
    assert!(d.inspect().is_err());
    let m = openpangu::inspect_checkpoint_with_options(&d.0, true).unwrap();
    let i = m.inventory.unwrap();
    assert!(!i.index_size_matches);
    assert_eq!(i.index_declared_bytes, Some(8));
    assert_eq!(i.stored_tensor_bytes, 4);
    fs::write(
        d.0.join("model.safetensors.index.json"),
        json!({"metadata":{"total_size":8},"weight_map":{"b":"one.safetensors"}}).to_string(),
    )
    .unwrap();
    assert!(openpangu::inspect_checkpoint_with_options(&d.0, true).is_err());
}
#[test]
fn tokenizer_has_separate_bounded_hash_budget() {
    let d = Dir::new();
    d.shard(
        "model.safetensors",
        json!({"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}),
        1,
    );
    let path = d.0.join("tokenizer.json");
    fs::File::create(&path)
        .unwrap()
        .set_len(17 * 1024 * 1024)
        .unwrap();
    assert!(
        d.inspect()
            .unwrap()
            .inventory
            .unwrap()
            .auxiliary_sha256
            .contains_key("tokenizer.json")
    );
    fs::File::create(&path)
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    assert!(d.inspect().is_err());
}
