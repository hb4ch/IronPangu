use inferfabric_compiler::{
    openpangu::{Inventory, TensorDescriptor},
    token::initialization_plans,
};
use inferfabric_dsl::{authoring::ModelDefinition, definition};
use inferfabric_ir::initialization::{DType, Instruction};
use std::collections::BTreeMap;
#[test]
fn window_attention_batches_queries_with_one_metadata_buffer() {
    let ModelDefinition::Execution {graph,..}=definition(include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric")).unwrap() else {panic!()};
    for count in [2,4,16] {
        let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap().with_prefill_tokens(count).unwrap();
        let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
        let mut windows=0;
        for (i,r) in p.instructions.iter().zip(&recipes) {
            if matches!(i.op.as_str(),"attention_metadata"|"latent_attention") {
                assert_eq!(r["packed_prefill"]["api_repetitions"],1);
                assert_eq!(r["packed_prefill"]["assembly_copies"],0);
            }
            if matches!(i.op.as_str(),"attention_metadata"|"attention_metadata_zero") {
                assert_eq!(r["value_fields"][0]["prefill_shape"],serde_json::json!([1024]));
            }
            if i.op=="latent_attention" {windows+=1;assert_eq!(r["value_fields"][0]["prefill_shape"],serde_json::json!([count,i.width]));}
        }
        assert_eq!(windows,30);
    }
}
#[test]
fn packed_attention_is_one_causal_call_without_assembly() {
    for source in [include_str!("../../../examples/openpangu/92b-tp8-ep8-c8.inferfabric"),include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric")] {
        let ModelDefinition::Execution {graph,..}=definition(source).unwrap() else {panic!()};
        for count in [2,4,16] {
            let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap().with_prefill_tokens(count).unwrap();
            let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
            let mut attentions=0;
            for (i,r) in p.instructions.iter().zip(&recipes) {
                if matches!(i.op.as_str(),"indexed_packed_attention"|"packed_latent_attention") {
                    attentions+=1;
                    assert_eq!(r["packed_prefill"]["api_repetitions"],1);
                    assert_eq!(r["packed_prefill"]["assembly_copies"],0);
                    assert_eq!(r["value_fields"][0]["prefill_shape"],serde_json::json!([count,i.width]));
                    assert!(r["packed_prefill"]["causal_alignment"].is_string());
                }
                if i.op=="packed_attention_metadata" {
                    let field=r["value_fields"].as_array().unwrap().iter().find(|f|f["name"]=="packed_query_lengths").unwrap();
                    assert_eq!(field["shape"],serde_json::json!([count]));
                    assert_eq!(field["prefill_shape"],serde_json::json!([count]));
                }
            }
            assert_eq!(attentions,16);
        }
    }
}
#[test]
fn packed_quantized_index_selection_has_no_row_assembly() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric");
    for (source, quantized) in [(source.to_string(),true),(source.replace("cache_dtype = \"int8\"","cache_dtype = \"bf16\""),false)] {
        let ModelDefinition::Execution {graph,..}=definition(&source).unwrap() else {panic!()};
        for count in [2,4,16] {
            let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap().with_prefill_tokens(count).unwrap();
            let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
            for (i,r) in p.instructions.iter().zip(&recipes).filter(|(i,_)|i.op=="sparse_index") {
                assert_eq!(r["packed_prefill"]["api_repetitions"],if quantized {1}else{count});
                assert_eq!(r["packed_prefill"]["assembly_copies"],if quantized {0}else{count});
                assert_eq!(r["value_fields"][0]["prefill_shape"],serde_json::json!([count,1,i.width]));
                assert_eq!(r["packed_prefill"]["causal_alignment"].is_string(),quantized);
            }
        }
    }
}
#[test]
fn long_context_plans_require_implemented_cache_paths() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric");
    let ModelDefinition::Execution {mut graph,..}=definition(source).unwrap() else {panic!()};
    for context in [513,2048,2049,65536] {
        graph.max_context=context;
        let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap().with_prefill_tokens(4).unwrap();
        let (_,states)=inferfabric_compiler::token_serving::program_envelope(&graph,8,&p).unwrap();
        for s in states.iter().filter(|s|!s.retention.is_full()) {
            assert_eq!(s.capacity_tokens,if context==513 {528}else{544});
        }
        p.validate().unwrap();
    }
    graph.max_context=65537;assert!(graph.validate().is_err());
    let ModelDefinition::Execution{mut graph,..}=definition(include_str!("../../../examples/openpangu/92b-tp8-ep8-c8.inferfabric")).unwrap() else {panic!()};
    graph.max_context=513;assert!(graph.validate().is_err());
}
#[test]
fn dense_attention_metadata_is_shared_only_with_matching_windows() {
    let ModelDefinition::Execution {graph,..}=definition(include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric")).unwrap() else {panic!()};
    let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap();
    let mut windows=BTreeMap::new();
    for (id,n) in p.instructions.iter().enumerate().filter(|(_,n)|n.op=="attention_metadata") {
        assert!(windows.insert(n.attrs["window"].clone(),id).is_none());
        let zero=&p.instructions[n.inputs[0]];
        assert_eq!(zero.op,"attention_metadata_zero");
        assert_eq!(zero.attrs,n.attrs);
    }
    assert_eq!(windows.len(),2);
    for n in p.instructions.iter().filter(|n|n.op=="latent_attention") {
        assert_eq!(windows[&n.attrs["window"]],n.inputs[2]);
    }
}
#[test]
fn compiled_serving_ring_matches_prefill_and_physical_storage() {
    use inferfabric_ir::serving::{Retention,Storage};
    let ModelDefinition::Execution {mut graph,..}=definition(include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric")).unwrap() else {panic!()};
    graph.max_context=129;
    let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap().with_prefill_tokens(4).unwrap();
    let (_,states)=inferfabric_compiler::token_serving::program_envelope(&graph,8,&p).unwrap();
    let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
    let mut rings=0;
    for state in &states {
        if matches!(&state.storage,Storage::Pages{encoding,..} if encoding=="bf16_latent512_rotary64") {
            rings+=1;assert_eq!(state.capacity_tokens,144);
            assert_eq!(state.retention,Retention::SlidingWindow{max_query_tokens:4});
            let id=p.instructions.iter().position(|i|i.logical_node==state.logical_node && i.op=="cache_norm_rope_append").unwrap();
            let fields=recipes[id]["value_fields"].as_array().unwrap();
            let cache=fields.iter().find(|f|f["name"]=="bf16_cache").unwrap();
            assert_eq!(cache["shape"],serde_json::json!([9,16,1,576]));
            assert_eq!(cache["max_query_tokens"],4);
        } else {assert!(state.retention.is_full());}
    }
    assert_eq!(rings,30);
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric").replace("mla_value(attended,",r#"mla_value(choose(c8, attended, add(attended, latent_attention(query, cache, scale_dim = 192, window = 2048, selection = "all_causal_within_window"))),"#);
    assert!(definition(&source).unwrap_err().to_string().contains("exactly one attention"));
}
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
fn learned_index_graph_has_explicit_shared_dependencies_and_paged_state() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric");
    let ModelDefinition::Execution{graph,..}=definition(source).unwrap() else {panic!()};
    let types=graph.validate().unwrap();
    let selectors:Vec<_>=graph.nodes.iter().enumerate().filter(|(_,n)|n.op=="sparse_index").collect();
    assert_eq!(selectors.len(),8);
    for (id,_) in selectors {
        assert_eq!(types[id],inferfabric_authoring::execution::Type::Indices(2048));
        let users:Vec<_>=graph.nodes.iter().filter(|n|n.inputs.contains(&id)).collect();
        assert_eq!(users.len(),2);
        assert!(users.iter().all(|n|n.op=="latent_attention" && n.attrs["selection"]=="learned_indices"));
    }
    let (_,states)=inferfabric_compiler::token_serving::envelope(&graph,8).unwrap();
    assert_eq!(states.iter().filter(|s|matches!(&s.storage,inferfabric_ir::serving::Storage::Pages{encoding,row_bytes:130,..} if encoding=="int8_index_keys_f16_scales")).count(),8);
    let p=inferfabric_compiler::token_program::lower(&graph,None).unwrap();
    assert_eq!(p.instructions.iter().filter(|i|i.op=="indexed_packed_attention").count(),16);
    assert_eq!(p.instructions.iter().filter(|i|i.op=="sparse_index").count(),8);
    assert_eq!(p.instructions.iter().filter(|i|i.op=="index_quantize").count(),16);
    assert_eq!(p.instructions.iter().filter(|i|i.op=="quant_scales_f16").count(),16);
    for selector in p.instructions.iter().filter(|i|i.op=="sparse_index") {
        assert_eq!(p.instructions[selector.inputs[0]].op,"quant_scales_f16");
        assert_eq!(p.instructions[selector.inputs[1]].dtype,"f16");
    }
    let recipes=serde_json::to_string(&inferfabric_compiler::token_program::recipes(&p,&graph)).unwrap();
    assert!(recipes.contains("aclnnAiInfraQuantLightningIndexer"));
    assert!(recipes.contains("index_cache_scales"));
    let mut short_context=graph.clone();short_context.max_context=129;
    let short_program=inferfabric_compiler::token_program::lower(&short_context,None).unwrap();
    let short_recipes=inferfabric_compiler::token_program::recipes(&short_program,&short_context);
    for (instruction,recipe) in short_program.instructions.iter().zip(&short_recipes) {
        let expected=match instruction.op.as_str() {
            "cache_c8_norm_rope_append"=>5,
            "cache_norm_rope_append"|"cache_append"=>9,
            "index_cache_append"=>3,
            _=>continue,
        };
        let table=recipe["value_fields"].as_array().unwrap().iter().find(|field|field["name"]=="device_block_table").unwrap();
        assert_eq!(table["shape"],serde_json::json!([1,expected]));
    }
    for (id,_) in graph.nodes.iter().enumerate().filter(|(_,n)|n.op=="mla_query") {
        let ops:Vec<_>=p.instructions.iter().filter(|i|i.logical_node==id).collect();
        assert_eq!(ops.iter().map(|i|i.op.as_str()).collect::<Vec<_>>(),["query_project","column_slice","rope_bf16","concat_columns"]);
        assert!(ops.iter().all(|i|i.dtype=="bf16"));
        assert_eq!(ops[1].attrs["start"],"128");
    }
    let mut invalid_rotary=graph.clone();
    invalid_rotary.nodes.iter_mut().find(|n|n.op=="mla_query").unwrap().attrs.insert("rotary_dtype".into(),"f16".into());
    assert!(invalid_rotary.validate().is_err());
    let mut default_rotary=graph.clone();
    for node in default_rotary.nodes.iter_mut().filter(|n|n.op=="mla_query") {node.attrs.remove("rotary_dtype");}
    let default_rotary=inferfabric_compiler::token_program::lower(&default_rotary,None).unwrap();
    assert_eq!(default_rotary.instructions.iter().filter(|i|i.op=="query_rotary_extract").count(),46);
    let ModelDefinition::Execution{graph:bf16,..}=definition(&source.replace("cache_dtype = \"int8\"","cache_dtype = \"bf16\"")).unwrap() else {panic!()};
    let bf16=inferfabric_compiler::token_program::lower(&bf16,None).unwrap();
    assert!(!bf16.instructions.iter().any(|i|i.op=="index_quantize" || i.op=="quant_scales_f16"));
    assert!(definition(&source.replace("cache_dtype = \"int8\"","cache_dtype = \"int4\"")).is_err());
    assert_eq!(p.instructions.iter().filter(|i|i.op=="packed_attention_metadata").count(),1);
    let mut bad=graph.clone();
    let attention=bad.nodes.iter_mut().find(|n|n.attrs.get("selection").map(String::as_str)==Some("learned_indices")).unwrap();
    attention.inputs.pop();
    assert!(bad.validate().is_err());
}

#[test]
fn rotary_prefix_expands_into_explicit_bf16_column_operations() {
    let (mut graph, _, inventory) = fixture();
    graph.nodes.insert(1, inferfabric_authoring::execution::Node {
        source: None, name: "leading_rotary".into(), op: "rotary_prefix".into(), inputs: vec![0],
        attrs: BTreeMap::from([("heads".into(), "20".into()), ("head_dim".into(), "128".into()),
            ("rotary_dim".into(), "64".into()), ("theta".into(), "6400000".into())]),
    });
    graph.nodes[2].inputs=vec![1];graph.nodes[3].inputs=vec![2];graph.output=3;
    let p=inferfabric_compiler::token_program::lower(&graph,Some(&inventory)).unwrap();
    let ops:Vec<_>=p.instructions.iter().filter(|i|i.logical_node==1).collect();
    assert_eq!(ops.iter().map(|i|i.op.as_str()).collect::<Vec<_>>(),
        ["column_slice","column_slice","rope_bf16","concat_columns"]);
    assert_eq!(ops[0].attrs["start"],"0");assert_eq!(ops[1].attrs["start"],"64");
    assert_eq!(ops[0].width,1280);assert_eq!(ops[3].width,2560);
    assert!(ops.iter().all(|i|i.dtype=="bf16"));
    p.encode().unwrap();
    let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
    let text=serde_json::to_string(&recipes).unwrap();
    assert!(text.contains("materialize declared strided BF16 columns"));
    for (key,value) in [("heads","0"),("heads","65"),("head_dim","64"),("rotary_dim","32"),("theta","10000")] {
        let mut bad=graph.clone();bad.nodes[1].attrs.insert(key.into(),value.into());
        assert!(bad.validate().is_err());
    }
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
        inferfabric_compiler::token_program::lower(&graph, Some(&inventory)).unwrap().encode().unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(output.join("execution.html")).unwrap(),
        inferfabric_compiler::token_view::render_with_program(&graph,
            &inferfabric_compiler::token_physical::lower(&graph, &parallel.devices).unwrap(),
            &inferfabric_compiler::token_program::lower(&graph, Some(&inventory)).unwrap()).unwrap()
    );
    for rank in 0..8 {
        assert!(output.join(format!("rank-{rank}.bin")).is_file());
    }
    let physical: inferfabric_plan::physical::Plan =
        serde_json::from_slice(&std::fs::read(output.join("physical.json")).unwrap()).unwrap();
    inferfabric_plan::physical::verify(&physical).unwrap();
    let serving: inferfabric_ir::serving::Manifest = serde_json::from_slice(
        &std::fs::read(output.join("serving.json")).unwrap()).unwrap();
    serving.validate().unwrap();
    assert_eq!(serving.identity.checkpoint_metadata_sha256, inventory.metadata_sha256);
    assert_eq!(serving.variants[0].max_tokens, 1);
    assert!(serving.qualification.is_none());
    assert!(
        inferfabric_compiler::token::write_bundle(&graph, &parallel, &inventory, &output).is_err()
    );
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn serving_state_is_derived_from_expanded_operators() {
    let ModelDefinition::Execution { graph, parallel } = definition(include_str!(
        "../../../examples/openpangu/92b-tp8-ep8-c8.inferfabric"
    )).unwrap() else { panic!() };
    let (variants, states) = inferfabric_compiler::token_serving::envelope(&graph, parallel.devices.len()).unwrap();
    assert_eq!(variants[0].max_tokens, 1);
    assert!(!variants[0].packed);
    assert_eq!(states.len(), 46 * 4);
    use inferfabric_ir::serving::Storage;
    assert_eq!(states.iter().filter(|s| matches!(s.storage, Storage::Slot { .. })).count(), 46 * 3);
    assert_eq!(states.iter().filter(|s| matches!(s.storage, Storage::Pages { row_bytes: 656, page_tokens: 32, .. })).count(), 16);
    assert_eq!(states.iter().filter(|s| matches!(s.storage, Storage::Pages { row_bytes: 1152, page_tokens: 16, .. })).count(), 30);
    assert!(states.iter().all(|s| !s.checkpoint_supported && s.ranks.len() == 8));
    let program = inferfabric_compiler::token_program::lower(&graph, None).unwrap();
    let recipes = inferfabric_compiler::token_program::recipes(&program, &graph);
    let mut control_offsets = Vec::new();
    for state in states.iter().filter(|s| matches!(s.storage, Storage::Pages { .. })) {
        let lowered: Vec<_> = program.instructions.iter().enumerate().filter(|(_, i)|
            i.logical_node == state.logical_node && matches!(i.op.as_str(), "cache_append" | "cache_norm_rope_append" | "cache_c8_norm_rope_append")).collect();
        assert_eq!(lowered.len(), 1, "cache state must have one executable binding");
        let fields = recipes[lowered[0].0]["value_fields"].as_array().unwrap();
        assert!(fields.iter().any(|f| f["name"] == "device_block_table"));
        let append = fields.iter().find(|f| f["name"] == "physical_append_position").unwrap();
        let Storage::Pages { page_tokens, .. } = state.storage else { unreachable!() };
        assert_eq!(append["page_tokens"], page_tokens);
        assert_eq!(append["control_bank_bytes"], 46 * 16);
        control_offsets.push(append["byte_offset"].as_u64().unwrap());
    }
    control_offsets.sort_unstable();
    assert_eq!(control_offsets, (0..46).map(|i| i*16).collect::<Vec<_>>());
}

#[test]
fn resident_capacity_is_executable_and_inspectable_without_claiming_batching() {
    let (graph, _, inventory) = fixture();
    let serial = inferfabric_compiler::token_program::lower(&graph, Some(&inventory)).unwrap();
    assert!(serial.clone().with_resident_sequences(0).is_err());
    assert!(serial.clone().with_resident_sequences(65).is_err());
    let resident = serial.clone().with_resident_sequences(2).unwrap();
    let binary = resident.encode().unwrap();
    assert_eq!(&binary[..8], b"IFTOKN03");
    assert_eq!(u32::from_le_bytes(binary[24..28].try_into().unwrap()), 2);
    assert_eq!(&serial.encode().unwrap()[..8], b"IFTOKN02");
    assert_eq!(resident.instructions, serial.instructions);
    let recipes = inferfabric_compiler::token_program::recipes(&resident, &graph);
    for (instruction, recipe) in resident.instructions.iter().zip(recipes) {
        let fields = recipe["value_fields"].as_array().unwrap();
        if instruction.op == "mome_inplace" || instruction.op == "mome" {
            assert!(fields.iter().any(|f| f["name"] == "recurrent_history" && f["shape"][0] == 2));
        }
        if matches!(instruction.op.as_str(), "cache_append" | "cache_norm_rope_append" | "cache_c8_norm_rope_append") {
            assert!(fields.iter().any(|f| f["name"] == "resident_partition" && f["resident_sequences"] == 2));
        }
    }
    let (variants, _) = inferfabric_compiler::token_serving::envelope(&graph, 8).unwrap();
    assert_eq!(variants[0].max_sequences, 1);
}

#[test]
fn packed_prefill_has_versioned_bounds_and_explicit_materialization() {
    let (graph, _, inventory) = fixture();
    let serial = inferfabric_compiler::token_program::lower(&graph, Some(&inventory)).unwrap();
    for count in [0, 3, 5, 32] {
        assert!(serial.clone().with_prefill_tokens(count).is_err());
    }
    let packed=serial.with_resident_sequences(2).unwrap().with_prefill_tokens(4).unwrap();
    let binary=packed.encode().unwrap();
    assert_eq!(&binary[..8], b"IFTOKN04");
    assert_eq!(u32::from_le_bytes(binary[24..28].try_into().unwrap()),2);
    assert_eq!(u32::from_le_bytes(binary[28..32].try_into().unwrap()),4);
    let recipes=inferfabric_compiler::token_program::recipes(&packed,&graph);
    for (instruction,recipe) in packed.instructions.iter().zip(recipes) {
        let variant=&recipe["packed_prefill"];
        assert_eq!(variant["tokens"],4);
        if instruction.op=="gather" {
            assert_eq!(variant["assembly_copies"],32);
        }
        if instruction.op=="latent_attention" || instruction.op=="packed_latent_attention" {
            assert_eq!(variant["api_repetitions"],4);
            assert_eq!(variant["assembly_copies"],4);
        }
    }
}

#[test]
fn shared_page_policy_is_executable_and_preserves_variant_widths() {
    let (graph,_,inventory)=fixture();
    let p=inferfabric_compiler::token_program::lower(&graph,Some(&inventory)).unwrap()
        .with_resident_sequences(2).unwrap().with_prefill_tokens(4).unwrap().with_shared_pages(true).unwrap();
    let binary=p.encode().unwrap();
    assert_eq!(&binary[..8],b"IFTOKN05");
    assert_eq!(u32::from_le_bytes(binary[24..28].try_into().unwrap()),2);
    assert_eq!(u32::from_le_bytes(binary[28..32].try_into().unwrap()),4);
    assert_eq!(&p.with_shared_pages(false).unwrap().encode().unwrap()[..8],b"IFTOKN04");
}
