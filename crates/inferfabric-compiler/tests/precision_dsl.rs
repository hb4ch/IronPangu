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
    assert_eq!(metadata.len(), 1);
    for i in &p.instructions {
        match i.op.as_str() {
            "latent_attention" => assert_eq!(i.inputs[2], metadata[0].0),
            "expert_gmm" => {
                let zero = &p.instructions[i.inputs[1]];
                assert_eq!(zero.op, "expert_zero");
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
    for r in inferfabric_compiler::token_program::recipes(&p, &graph) {
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
