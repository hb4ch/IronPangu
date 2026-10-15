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
    for rank in 0..8 {
        assert!(output.join(format!("rank-{rank}.bin")).is_file());
    }
    assert!(
        inferfabric_compiler::token::write_bundle(&graph, &parallel, &inventory, &output).is_err()
    );
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    std::fs::remove_dir_all(root).unwrap();
}
