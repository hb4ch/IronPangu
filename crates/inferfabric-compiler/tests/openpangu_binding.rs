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
        metadata_sha256: "a".repeat(64),
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
#[test]
fn tp_ep_slices_cover_weights_once_and_keep_experts_whole() {
    let mut m = fixture();
    m.config["n_routed_experts"] = json!(8);
    let tensors = &mut m.inventory.as_mut().unwrap().tensors;
    for layer in [2, 3] {
        let template: Vec<_> = tensors
            .iter()
            .filter(|(name, _)| name.starts_with(&format!("model.layers.{layer}.mlp.experts.0.")))
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect();
        for e in 2..8 {
            for (name, t) in &template {
                tensors.insert(
                    name.replace(".experts.0.", &format!(".experts.{e}.")),
                    t.clone(),
                );
            }
        }
        tensors
            .get_mut(&format!("model.layers.{layer}.mlp.gate.weight"))
            .unwrap()
            .shape[0] = 8;
        tensors
            .get_mut(&format!("model.layers.{layer}.mlp.e_score_correction_bias"))
            .unwrap()
            .shape[0] = 8;
    }
    // Keep synthetic storage lengths consistent with the expanded router rows.
    for layer in [2, 3] {
        tensors
            .get_mut(&format!("model.layers.{layer}.mlp.gate.weight"))
            .unwrap()
            .bytes = 8 * 2560 * 2;
        tensors
            .get_mut(&format!("model.layers.{layer}.mlp.e_score_correction_bias"))
            .unwrap()
            .bytes = 8 * 4;
    }
    let spec = inferfabric_model::OpenPanguParallel {
        tp: 8,
        ep: 8,
        devices: (0..8).collect(),
        max_num_seqs: 4,
        max_num_batched_tokens: 16,
        max_model_len: 2048,
    };
    let plan = openpangu::parallel::compile(&m, &spec).unwrap();
    assert_eq!(plan.experts_per_rank, 1);
    let name = "model.layers.0.self_attn.q_b_proj.weight";
    let original = &m.inventory.as_ref().unwrap().tensors[name];
    for (rank, r) in plan.ranks.iter().enumerate() {
        let w = &r.weights[name];
        assert_eq!(w.bytes * 8, original.bytes);
        assert_eq!(w.offset, original.offset + w.bytes * rank as u64);
        assert!(r.weights.contains_key(&format!(
            "model.layers.2.mlp.experts.{rank}.gate_proj.weight"
        )));
        assert!(!r.weights.contains_key(&format!(
            "model.layers.2.mlp.experts.{}.gate_proj.weight",
            (rank + 1) % 8
        )));
        assert_eq!(
            r.weights["model.embed_tokens.weight"].placement,
            "replicated"
        );
        assert_eq!(
            r.weights["model.layers.0.self_attn.q_b_proj.weight_scale"].local_shape,
            [1152, 1]
        );
    }
    assert!(
        plan.schedule
            .iter()
            .any(|s| s.collective == "all_reduce_sum_fp32")
    );
    for rank in 0..8 {
        let init = openpangu::parallel::initialization_plan(&plan, rank).unwrap();
        init.validate().unwrap();
        assert_eq!(init.parameters.len(), plan.ranks[rank].weights.len());
        assert_eq!(init.instructions.len(), init.parameters.len());
        let binary = inferfabric_compiler::initialization::encode(&init).unwrap();
        assert_eq!(&binary[..8], b"IFINIT01");
        for parameter in &init.parameters {
            assert_eq!(
                parameter.shape,
                plan.ranks[rank].weights[&parameter.name].local_shape
            );
        }
    }
    assert!(!plan.executable);
    let mut bad = spec.clone();
    bad.ep = 4;
    assert!(openpangu::parallel::compile(&m, &bad).is_err());
    bad = spec;
    bad.max_model_len = 524289;
    assert!(openpangu::parallel::compile(&m, &bad).is_err());
}
