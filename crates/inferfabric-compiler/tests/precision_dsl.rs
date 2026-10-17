use inferfabric_compiler::token_program::lower;
use inferfabric_dsl::{authoring::ModelDefinition, definition};
const SOURCE: &str = r#"language inferfabric 1;
model Typed {context=512;eos=148902;
let e=embedding(weight="lookup",width=2560,vocab=151552);
let f=cast(e,dtype=f32);let b=cast(f,dtype=bf16);
let q=quantize(b,dtype=int8,scheme=per_token_symmetric,scale_dtype=f32);
let l=linear(q,weight="projection",width=18944,placement=tp_rows);
let out=gather(l);yield out;}
plan P for Typed {target=ascend_token_v1;tp=8;ep=8;devices=[0,1,2,3,4,5,6,7];max_num_seqs=1;max_num_batched_tokens=1;max_model_len=512;}"#;
#[test]
fn dtype_and_runtime_quantization_are_typed_dsl_values() {
    let ModelDefinition::Execution { graph, .. } = definition(SOURCE).unwrap() else {
        panic!()
    };
    let p = lower(&graph, None).unwrap();
    assert_eq!(p.instructions.iter().filter(|n| n.op == "cast").count(), 2);
    assert_eq!(
        p.instructions.iter().filter(|n| n.op == "quantize").count(),
        1
    );
    let linear = p.instructions.iter().find(|n| n.op == "linear").unwrap();
    assert_eq!(p.instructions[linear.inputs[0]].kind, 5);
    for bad in [
        SOURCE.replace("dtype=f32", "dtype=int8"),
        SOURCE.replace("scheme=per_token_symmetric", "scheme=unknown"),
        SOURCE.replace("scale_dtype=f32", "scale_dtype=bf16"),
        SOURCE.replace("let out=gather(l)", "let out=gather(q)"),
    ] {
        assert!(definition(&bad).is_err());
    }
}
#[test]
fn quantized_cache_has_explicit_storage_and_decode_instructions() {
    let source = include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric");
    for dtype in ["int8", "int4"] {
        let s=source.replacen("eps = 1e-5, theta = 6400000", &format!("eps = 1e-5, theta = 6400000, cache_dtype={dtype}, cache_scale_dtype=f32, cache_quantization=per_token_head_symmetric"),1);
        let ModelDefinition::Execution { graph, .. } = definition(&s).unwrap() else {
            panic!()
        };
        let p=lower(&graph,None).unwrap();
        p.validate().unwrap();
        for op in ["cache_normalize","cache_quantize_append","cache_dequantize"] {
            assert_eq!(p.instructions.iter().filter(|i|i.op==op).count(),46);
        }
        assert!(!p.instructions.iter().any(|i|i.op=="cache_norm_rope_append"));
        for i in p.instructions.iter().filter(|i|i.op=="latent_attention") {
            assert_eq!(i.inputs.len(),4);
            let decoded=&p.instructions[i.inputs[3]];
            assert_eq!(decoded.op,"cache_dequantize");
            assert_eq!(decoded.attrs["cache_dtype"],dtype);
            assert_eq!(decoded.inputs,vec![i.inputs[1]]);
            assert_eq!(decoded.width,512*576);
        }
        assert!(definition(&s.replace("cache_scale_dtype=f32", "cache_scale_dtype=bf16")).is_err());
    }
}

#[test]
fn query_composites_expose_every_conversion_and_dependency() {
    let ModelDefinition::Execution { graph, .. } = definition(include_str!(
        "../../../examples/openpangu/92b-tp8-ep8.inferfabric"
    ))
    .unwrap() else {
        panic!()
    };
    let p = lower(&graph, None).unwrap();
    p.validate().unwrap();
    assert!(
        !p.instructions
            .iter()
            .any(|i| matches!(i.op.as_str(), "mla_query" | "rope"))
    );
    for op in [
        "query_project",
        "query_rotary_extract",
        "rope_f32",
        "concat_columns",
    ] {
        assert_eq!(p.instructions.iter().filter(|i| i.op == op).count(), 46);
    }
    for i in p.instructions.iter().filter(|i| i.op == "concat_columns") {
        assert_eq!(p.instructions[i.inputs[0]].op, "query_project");
        let round = &p.instructions[i.inputs[1]];
        assert_eq!(round.op, "cast");
        assert_eq!(round.dtype, "bf16");
        let rotate = &p.instructions[round.inputs[0]];
        assert_eq!(rotate.op, "rope_f32");
        assert_eq!(rotate.dtype, "f32");
        let extract = &p.instructions[rotate.inputs[0]];
        assert_eq!(extract.op, "query_rotary_extract");
        assert_eq!(extract.dtype, "f32");
        assert_eq!(extract.inputs, p.instructions[i.inputs[0]].inputs);
    }
}

#[test]
fn decoder_composites_have_explicit_calls_and_storage_effects() {
    let ModelDefinition::Execution { graph, .. } = definition(include_str!(
        "../../../examples/openpangu/92b-tp8-ep8.inferfabric"
    ))
    .unwrap() else {
        panic!()
    };
    let p = lower(&graph, None).unwrap();
    p.validate().unwrap();
    assert!(!p.instructions.iter().any(|i| matches!(
        i.op.as_str(),
        "router" | "expert_linear" | "expert_swiglu_down" | "mhc_pre" | "mla_query" | "rope"
    )));
    let metadata: Vec<_> = p
        .instructions
        .iter()
        .enumerate()
        .filter(|(_, i)| i.op == "attention_metadata")
        .collect();
    let windows: std::collections::BTreeSet<_> = p.instructions.iter()
        .filter(|i| i.op == "latent_attention").map(|i| &i.attrs["window"]).collect();
    assert_eq!(metadata.len(), windows.len());
    for i in &p.instructions {
        match i.op.as_str() {
            "latent_attention" => {
                let producer = &p.instructions[i.inputs[2]];
                assert_eq!(producer.op, "attention_metadata");
                assert_eq!(producer.attrs["window"], i.attrs["window"]);
            }
            "expert_gmm" => {
                let zero = &p.instructions[i.inputs[1]];
                assert_eq!(zero.op, "expert_zero");
                assert_eq!(zero.kernel, "aclnnInplaceZero");
                assert_eq!(zero.inputs, vec![i.inputs[0]]);
            }
            "expert_reduce" => {
                assert_eq!(i.inputs.len(), 3);
                assert_eq!(p.instructions[i.inputs[0]].op, "cast");
                assert_eq!(p.instructions[i.inputs[0]].dtype, "f32");
            }
            "cache_norm_rope_append" => {
                assert_eq!(i.inputs.len(), 2);
                assert_eq!(p.instructions[i.inputs[0]].op, "concat");
            }
            "router_select" => assert_eq!(p.instructions[i.inputs[0]].op, "router_projection"),
            "mhc_sinkhorn" => assert_eq!(p.instructions[i.inputs[0]].op, "mhc_pre_project"),
            _ => (),
        }
    }
    for (index, r) in inferfabric_compiler::token_program::recipes(&p, &graph).into_iter().enumerate() {
        if p.instructions[index].op == "expert_zero" {
            assert_eq!(r["apis"], serde_json::json!(["aclnnInplaceZero"]));
            assert!(r["materialization"].as_str().unwrap().contains("INT8 byte alias"));
        }
        assert!(
            r["apis"].as_array().unwrap().len() <= 1 || r["granularity"] == "initialization recipe"
        );
        assert_ne!(r["internal"], "unresolved backend recipe");
    }
}

#[test]
fn separate_expert_projections_share_explicit_route_quantization() {
    let source = include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric");
    let source=source.replace("let gate_up = expert_linear(x, routes, weight = expert.gate_proj,\n                               paired_weight = expert.up_proj, width = 1024);\n    let down = expert_swiglu_down(gate_up, routes, weight = expert.down_proj, width = 2560);",
        "let gate = expert_linear(x, routes, weight=expert.gate_proj, width=1024); let up=expert_linear(x,routes,weight=expert.up_proj,width=1024); let down=expert_swiglu_down(gate,up,routes,weight=expert.down_proj,width=2560);");
    let ModelDefinition::Execution { graph, .. } = definition(&source).unwrap() else {
        panic!()
    };
    let p = lower(&graph, None).unwrap();
    p.validate().unwrap();
    let packs: Vec<_> = p
        .instructions
        .iter()
        .filter(|i| i.op == "expert_pack")
        .collect();
    assert!(!packs.is_empty());
    for pack in packs {
        let gate = &p.instructions[pack.inputs[0]];
        let up = &p.instructions[pack.inputs[1]];
        assert_eq!(gate.op, "expert_gmm");
        assert_eq!(gate.inputs[0], up.inputs[0]);
    }
}
#[test]
fn integer_cache_scratch_is_page_padded_for_short_contexts() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric")
        .replace("context = 512", "context = 17")
        .replace("max_model_len = 512", "max_model_len = 17")
        .replacen("eps = 1e-5, theta = 6400000", "eps = 1e-5, theta = 6400000, cache_dtype=int4, cache_scale_dtype=f32, cache_quantization=per_token_head_symmetric", 1);
    let ModelDefinition::Execution {graph,..}=definition(&source).unwrap() else {panic!()};
    let p=lower(&graph,None).unwrap();
    p.validate().unwrap();
    for i in p.instructions.iter().filter(|i|i.op=="cache_dequantize") {
        assert_eq!(i.width,32*576);
    }
    let recipes=inferfabric_compiler::token_program::recipes(&p,&graph);
    for (i,n) in p.instructions.iter().enumerate().filter(|(_,n)|n.op=="cache_quantize_append") {
        assert_eq!(n.attrs["cache_dtype"],"int4");
        assert_eq!(recipes[i]["value_fields"][0]["shape"],serde_json::json!([32,1,576]));
        assert_eq!(recipes[i]["value_fields"][1]["shape"],serde_json::json!([32,1]));
    }
}

#[test]
fn packed_c8_uses_checkpoint_dsa_layers_without_cache_reconstruction() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8-c8.inferfabric");
    let ModelDefinition::Execution {graph,..}=definition(source).unwrap() else {panic!()};
    let p=lower(&graph,None).unwrap();
    assert_eq!(p,lower(&graph,None).unwrap());
    assert_eq!(p.instructions.iter().filter(|n|n.op=="cache_c8_norm_rope_append").count(),16);
    assert_eq!(p.instructions.iter().filter(|n|n.op=="cache_norm_rope_append").count(),30);
    assert_eq!(p.instructions.iter().filter(|n|n.op=="packed_latent_attention").count(),16);
    assert_eq!(p.instructions.iter().filter(|n|n.op=="packed_attention_metadata").count(),1);
    assert!(!p.instructions.iter().any(|n|matches!(n.op.as_str(),"cache_normalize"|"cache_dequantize"|"cache_quantize_append")));
    for n in p.instructions.iter().filter(|n|n.op=="cache_c8_norm_rope_append") {
        let layer:usize=n.attrs["weight"].split('.').nth(2).unwrap().parse().unwrap();
        assert_eq!(layer%3,0);
        assert_eq!(n.attrs["cache_quantization"],"pertile128");
        assert_eq!(n.attrs["cache_rope_dtype"],"bf16");
    }
    for n in p.instructions.iter().filter(|n|n.op=="packed_latent_attention") {
        assert_eq!(p.instructions[n.inputs[1]].op,"cache_c8_norm_rope_append");
        assert_eq!(p.instructions[n.inputs[2]].op,"packed_attention_metadata");
    }
    for bad in [source.replace("pertile128","per_token_head_symmetric"),source.replace("cache_rope_dtype = bf16","cache_rope_dtype = int8"),source.replace("cache_scale_dtype = f32","cache_scale_dtype = bf16")] {
        assert!(definition(&bad).is_err());
    }
}
