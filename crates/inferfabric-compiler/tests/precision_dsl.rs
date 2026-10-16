use inferfabric_dsl::{definition,authoring::ModelDefinition};
use inferfabric_compiler::token_program::lower;
const SOURCE:&str=r#"language inferfabric 1;
model Typed {context=512;eos=148902;
let e=embedding(weight="lookup",width=2560,vocab=151552);
let f=cast(e,dtype=f32);let b=cast(f,dtype=bf16);
let q=quantize(b,dtype=int8,scheme=per_token_symmetric,scale_dtype=f32);
let l=linear(q,weight="projection",width=18944,placement=tp_rows);
let out=gather(l);yield out;}
plan P for Typed {target=ascend_token_v1;tp=8;ep=8;devices=[0,1,2,3,4,5,6,7];max_num_seqs=1;max_num_batched_tokens=1;max_model_len=512;}"#;
#[test]
fn dtype_and_runtime_quantization_are_typed_dsl_values() {
    let ModelDefinition::Execution{graph,..}=definition(SOURCE).unwrap() else {panic!()};
    let p=lower(&graph,None).unwrap();
    assert_eq!(p.instructions.iter().filter(|n|n.op=="cast").count(),2);
    assert_eq!(p.instructions.iter().filter(|n|n.op=="quantize").count(),1);
    let linear=p.instructions.iter().find(|n|n.op=="linear").unwrap();
    assert_eq!(p.instructions[linear.inputs[0]].kind,5);
    for bad in [SOURCE.replace("dtype=f32","dtype=int8"),SOURCE.replace("scheme=per_token_symmetric","scheme=unknown"),SOURCE.replace("scale_dtype=f32","scale_dtype=bf16"),SOURCE.replace("let out=gather(l)","let out=gather(q)")] {assert!(definition(&bad).is_err());}
}
#[test]
fn quantized_cache_specs_are_preserved_but_not_silently_substituted() {
    let source=include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric");
    for dtype in ["int8","int4"] {
        let s=source.replacen("eps = 1e-5, theta = 6400000", &format!("eps = 1e-5, theta = 6400000, cache_dtype={dtype}, cache_scale_dtype=f32, cache_quantization=per_token_head_symmetric"),1);
        let ModelDefinition::Execution{graph,..}=definition(&s).unwrap() else {panic!()};
        let err=lower(&graph,None).unwrap_err().to_string();assert!(err.contains("refusing BF16 substitution"));
        assert!(definition(&s.replace("cache_scale_dtype=f32","cache_scale_dtype=bf16")).is_err());
    }
}
