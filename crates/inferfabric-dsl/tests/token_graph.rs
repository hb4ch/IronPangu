use inferfabric_dsl::{
    authoring::{self, ModelDefinition},
    definition,
};
const SOURCE: &str = include_str!("../../../examples/openpangu/92b-tp8-ep8.inferfabric");
#[test]
fn full_decoder_is_explicit_and_round_trips_through_compiled_authoring() {
    let model = definition(SOURCE).unwrap();
    let ModelDefinition::Execution { graph, parallel } = &model else {
        panic!("expected executable graph")
    };
    assert_eq!(graph.nodes.len(), 2109);
    assert_eq!(graph.nodes.iter().filter(|n| n.op == "router").count(), 44);
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.op == "latent_attention")
            .count(),
        46
    );
    assert_eq!(graph.nodes.iter().filter(|n| n.op == "mome").count(), 138);
    assert_eq!(graph.max_context, parallel.max_model_len);
    let encoded = authoring::encode(&model).unwrap();
    let decoded = authoring::decode(&encoded).unwrap();
    assert_eq!(model, decoded);
    let ModelDefinition::Execution { graph: other, .. } = decoded else {
        unreachable!()
    };
    assert_eq!(graph.encode().unwrap(), other.encode().unwrap());
    assert!(graph.dot().unwrap().contains("l45_out"));
}
#[test]
fn unsafe_dataflow_shapes_effects_and_budgets_are_rejected() {
    for source in [
        SOURCE.replacen("mhc_input(l0_hc)", "mhc_input(embedding)", 1),
        SOURCE.replacen("linear(l0_an,", "linear(not_defined,", 1),
        SOURCE.replacen("theta = 6400000", "theta = 10000", 1),
        SOURCE.replacen("start = 512, width = 64", "start = 540, width = 64", 1),
        SOURCE.replacen("placement = \"tp_rows\"", "placement = \"magic\"", 1),
        SOURCE.replacen("l2_eg, l2_eu, l2_route", "l2_eg, l2_eu, l3_route", 1),
        SOURCE.replacen("yield logits;", "yield normalized;", 1),
        SOURCE.replace("context = 512", "context = 513"),
        SOURCE.replace("max_num_seqs = 1", "max_num_seqs = 4"),
        SOURCE.replacen("mhc_post(l0_hc, l0_apn)", "mhc_post(l0_fhc, l0_apn)", 1),
    ] {
        assert!(definition(&source).is_err());
    }
}
#[test]
fn changing_dsl_changes_program_not_a_model_selector() {
    let bypass = SOURCE
        .lines()
        .filter(|s| {
            !s.trim_start().starts_with("let l0_")
                && !(1..46).any(|l| s.trim_start().starts_with(&format!("let l{l}_")))
        })
        .collect::<Vec<_>>()
        .join("\n")
        .replace("mhc_merge(l45_out,", "mhc_merge(streams,");
    let ModelDefinition::Execution { graph, .. } = definition(&bypass).unwrap() else {
        panic!()
    };
    assert_eq!(graph.nodes.len(), 7);
    let ModelDefinition::Execution {
        graph: original, ..
    } = definition(SOURCE).unwrap()
    else {
        panic!()
    };
    assert_ne!(graph.encode().unwrap(), original.encode().unwrap());
}
