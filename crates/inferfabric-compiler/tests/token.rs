use inferfabric_compiler::{
    openpangu::{Inventory, TensorDescriptor},
    token::initialization_plans,
};
use inferfabric_dsl::{authoring::ModelDefinition, definition};
use inferfabric_ir::initialization::{DType, Instruction};
use std::collections::BTreeMap;
fn fixture() -> (
    inferfabric_authoring::execution::Graph,
    inferfabric_model::OpenPanguParallel,
    Inventory,
) {
    let source = r#"language inferfabric 1;
model Renamed {context=512;eos=148902;
let e=embedding(weight="lookup",width=2560,vocab=151552);
let l=linear(e,weight="projection",width=18944,placement="tp_rows");
let logits=gather(l);yield logits;}
plan P for Renamed {target=ascend_token_v1;tp=8;ep=8;devices=[0,1,2,3,4,5,6,7];max_num_seqs=1;max_num_batched_tokens=1;max_model_len=512;}
"#;
    let ModelDefinition::Execution { graph, parallel } = definition(source).unwrap() else {
        panic!()
    };
    let mut tensors = BTreeMap::new();
    for (name, dtype, shape, elem) in [
        ("lookup.weight", "BF16", vec![151552, 2560], 2),
        ("projection.weight", "I8", vec![151552, 2560], 1),
        ("projection.weight_scale", "BF16", vec![151552, 1], 2),
    ] {
        let bytes = shape.iter().product::<u64>() * elem;
        tensors.insert(
            name.into(),
            TensorDescriptor {
                shard: "weights.safetensors".into(),
                dtype: dtype.into(),
                shape,
                offset: 4096,
                bytes,
            },
        );
    }
    let inventory = Inventory {
        scope: "test",
        payload_verified: false,
        stored_tensor_bytes: 0,
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
    };
    (graph, parallel, inventory)
}
#[test]
fn paired_projection_binds_both_replicated_bf16_matrices() {
    let (mut graph, parallel, mut inventory) = fixture();
    graph.nodes[1].attrs.extend([
        ("placement".into(), "replicated".into()),
        ("width".into(), "9472".into()),
        ("paired_weight".into(), "other".into()),
        ("paired_width".into(), "9472".into()),
    ]);
    let weight = inventory.tensors.get_mut("projection.weight").unwrap();
    weight.dtype = "BF16".into();
    weight.shape[0] = 9472;
    weight.bytes = 9472 * 2560 * 2;
    let other = weight.clone();
    inventory.tensors.insert("other.weight".into(), other);
    let plans = initialization_plans(&graph, &parallel, &inventory).unwrap();
    for plan in plans {
        assert_eq!(plan.parameters.len(), 3);
        for name in ["projection.weight", "other.weight"] {
            let p = plan.parameters.iter().find(|p| p.name == name).unwrap();
            assert_eq!(p.shape, vec![9472, 2560]);
            assert_eq!(p.dtype, DType::Bf16);
        }
    }
    let mut bad = graph.clone();
    bad.nodes[1].attrs.remove("paired_width");
    assert!(bad.validate().is_err());
    bad = graph.clone();
    bad.nodes[1].attrs.remove("placement");
    assert!(bad.validate().is_err());
    bad = graph.clone();
    bad.nodes[1].attrs.insert("placement".into(), "tp_rows".into());
    assert!(bad.validate().is_err());
    inventory.tensors.get_mut("other.weight").unwrap().dtype = "I8".into();
    assert!(initialization_plans(&graph, &parallel, &inventory).is_err());
}

#[test]
fn graph_drives_binding_sharding_and_scale_cast_without_model_names() {
    let (graph, parallel, mut inventory) = fixture();
    inventory.tensors.insert(
        "unused.invalid".into(),
        TensorDescriptor {
            shard: "ignored".into(),
            dtype: "unknown".into(),
            shape: vec![],
            offset: 0,
            bytes: 0,
        },
    );
    let plans = initialization_plans(&graph, &parallel, &inventory).unwrap();
    assert_eq!(plans.len(), 8);
    for (r, p) in plans.iter().enumerate() {
        assert_eq!(p.parameters.len(), 3);
        assert_eq!(p.parameters[1].shape, vec![18944, 2560]);
        assert_eq!(p.parameters[2].dtype, DType::F32);
        let Instruction::LoadSlice { offset, bytes, .. } = p.instructions[1];
        assert_eq!(bytes, 18944 * 2560);
        assert_eq!(offset, 4096 + bytes * r as u64);
        let Instruction::LoadSlice { offset, .. } = p.instructions[0];
        assert_eq!(offset, 4096);
    }
}
#[test]
fn column_placement_packs_source_rows_and_replicates_scales() {
    let (mut graph, parallel, mut inventory) = fixture();
    graph.nodes[1]
        .attrs
        .insert("placement".into(), "tp_columns".into());
    let w = inventory.tensors.get_mut("projection.weight").unwrap();
    w.shape = vec![18944, 20480];
    w.bytes = 18944 * 20480;
    let scale = inventory
        .tensors
        .get_mut("projection.weight_scale")
        .unwrap();
    scale.shape = vec![18944, 1];
    scale.bytes = 18944 * 2;
    let plans = initialization_plans(&graph, &parallel, &inventory).unwrap();
    for (rank, plan) in plans.iter().enumerate() {
        assert_eq!(plan.parameters[1].shape, vec![18944, 2560]);
        let Instruction::LoadSlice {
            offset,
            bytes,
            source_rows,
            ..
        } = &plan.instructions[1];
        assert_eq!(*offset, 4096 + rank as u64 * 2560);
        assert_eq!(*bytes, 18944 * 2560);
        assert_eq!(source_rows.as_ref().unwrap().row_bytes, 2560);
        assert_eq!(source_rows.as_ref().unwrap().stride_bytes, 20480);
        let Instruction::LoadSlice {
            offset,
            source_rows,
            ..
        } = &plan.instructions[2];
        assert_eq!(*offset, 4096);
        assert!(source_rows.is_none());
        assert_eq!(plan.parameters[2].shape, vec![18944, 1]);
    }
}
#[test]
fn invalid_placement_missing_scale_and_shape_fail_before_execution() {
    for mutation in 0..4 {
        let (mut graph, parallel, mut inventory) = fixture();
        match mutation {
            0 => {
                inventory.tensors.remove("projection.weight_scale");
            }
            1 => {
                inventory
                    .tensors
                    .get_mut("projection.weight")
                    .unwrap()
                    .shape[0] = 1
            }
            2 => {
                graph.nodes[1]
                    .attrs
                    .insert("placement".into(), "replicated".into());
            }
            _ => {
                inventory
                    .tensors
                    .get_mut("projection.weight_scale")
                    .unwrap()
                    .dtype = "I8".into()
            }
        }
        assert!(initialization_plans(&graph, &parallel, &inventory).is_err());
    }
}
#[test]
fn bundle_publication_is_complete_and_refuses_overwrite() {
    let (graph, parallel, inventory) = fixture();
    let root = std::env::temp_dir().join(format!(
        "inferfabric-token-bundle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let output = root.join("model");
    inferfabric_compiler::token::write_bundle(&graph, &parallel, &inventory, &output).unwrap();
    assert_eq!(
        std::fs::read(output.join("execution.bin")).unwrap(),
        graph.encode().unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(output.join("execution.html")).unwrap(),
        inferfabric_compiler::token_view::render(&graph).unwrap()
    );
    for rank in 0..8 {
        assert!(output.join(format!("rank-{rank}.bin")).is_file());
    }
    let physical: inferfabric_plan::physical::Plan =
        serde_json::from_slice(&std::fs::read(output.join("physical.json")).unwrap()).unwrap();
    inferfabric_plan::physical::verify(&physical).unwrap();
    assert!(
        inferfabric_compiler::token::write_bundle(&graph, &parallel, &inventory, &output).is_err()
    );
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    std::fs::remove_dir_all(root).unwrap();
}
