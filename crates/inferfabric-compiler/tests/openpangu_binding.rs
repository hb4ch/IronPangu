use inferfabric_compiler::openpangu::{self, Inventory, Manifest, TensorDescriptor, binding};
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn fixture() -> Manifest {
    let f: Value = serde_json::from_str(include_str!("fixtures/openpangu92-binding.json")).unwrap();
    let mut m = openpangu::from_config(&serde_json::to_vec(&f["config"]).unwrap()).unwrap();
    let tensors: BTreeMap<_, _> = f["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, t)| {
            (
                name.clone(),
                TensorDescriptor {
                    shard: "synthetic.safetensors".into(),
                    dtype: t["dtype"].as_str().unwrap().into(),
                    shape: serde_json::from_value(t["shape"].clone()).unwrap(),
                    offset: 0,
                    bytes: t["bytes"].as_u64().unwrap(),
                },
            )
        })
        .collect();
    m.inventory = Some(Inventory {
        scope: "test",
        payload_verified: false,
        stored_tensor_bytes: tensors.values().map(|t| t.bytes).sum(),
        index_declared_bytes: None,
        index_size_matches: true,
        bytes_by_dtype: BTreeMap::new(),
        tensor_count: tensors.len(),
        tensors,
        shards: BTreeMap::new(),
        index_sha256: None,
        auxiliary_sha256: BTreeMap::new(),
        metadata_sha256: "fixture".into(),
        capacity_note: "test",
    });
    m
}
#[test]
fn released_contract_binds_dense_experts_mhc_mome_and_shared_indices() {
    let m = fixture();
    let b = binding::bind(&m).unwrap();
    assert_eq!(b.bindings.len(), m.inventory.as_ref().unwrap().tensor_count);
    assert_eq!(b.shared_index_sources.get(&3), Some(&0));
    assert_eq!(b.mtp_bytes, 0);
    assert!(!b.executable);
    assert_eq!(
        b.bindings["model.layers.0.self_attn.q_b_proj.weight"].transform,
        "dynamic_w8a8_weight_out_in"
    );
    assert_eq!(
        b.bindings["model.layers.0.attn_mhc_module.phi.weight"].transform,
        "cast_fp32"
    );
}
#[test]
fn missing_expert_scale_wrong_shape_dtype_and_extra_weights_fail() {
    for mutation in 0..4 {
        let mut m = fixture();
        let t = &mut m.inventory.as_mut().unwrap().tensors;
        let name = "model.layers.2.mlp.experts.1.up_proj.weight_scale";
        match mutation {
            0 => {
                t.remove(name);
            }
            1 => t.get_mut(name).unwrap().shape = vec![1, 1024],
            2 => t.get_mut(name).unwrap().dtype = "I8".into(),
            _ => {
                t.insert("model.unexpected.weight".into(), t[name].clone());
            }
        }
        assert!(binding::bind(&m).is_err());
    }
}
#[test]
fn incompatible_quantization_and_missing_index_producer_fail() {
    for (path, value) in [
        ("quant", json!(false)),
        ("index", json!("shared")),
        ("dimension", json!(u64::MAX)),
    ] {
        let mut m = fixture();
        match path {
            "quant" => {
                m.config["quantization_config"]["config_groups"]["group_0"]["input_activations"]["dynamic"] =
                    value
            }
            "index" => m.layers[0].indexer_mode = Some(value.as_str().unwrap().into()),
            _ => m.config["n_shared_experts"] = value,
        }
        assert!(binding::bind(&m).is_err());
    }
}
#[test]
fn mtp_is_excluded_only_within_declared_layer_range() {
    let mut m = fixture();
    m.config["num_nextn_predict_layers"] = json!(1);
    let i = m.inventory.as_mut().unwrap();
    let t = i.tensors["model.norm.weight"].clone();
    i.tensors
        .insert("model.layers.4.fixture.weight".into(), t.clone());
    let b = binding::bind(&m).unwrap();
    assert_eq!(b.mtp_bytes, t.bytes);
    m.config["num_nextn_predict_layers"] = json!(0);
    assert!(binding::bind(&m).is_err());
}
