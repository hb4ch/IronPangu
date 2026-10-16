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
    assert!(graph.dot().unwrap().contains("hidden.i45.block.output"));
}
#[test]
fn unsafe_dataflow_shapes_effects_and_budgets_are_rejected() {
    for source in [
        SOURCE.replacen("mhc_input(attn_hc)", "mhc_input(x)", 1),
        SOURCE.replacen(
            "linear(x, weight = w.q_a_proj",
            "linear(not_defined, weight = w.q_a_proj",
            1,
        ),
        SOURCE.replacen("theta = 6400000", "theta = 10000", 1),
        SOURCE.replacen("start = 512, width = 64", "start = 540, width = 64", 1),
        SOURCE.replacen("placement = \"tp_rows\"", "placement = \"magic\"", 1),
        SOURCE.replacen("gate, up, routes", "gate, up, x", 1),
        SOURCE.replacen("yield logits;", "yield normalized;", 1),
        SOURCE.replace("context = 512", "context = 513"),
        SOURCE.replace("max_num_seqs = 1", "max_num_seqs = 4"),
        SOURCE.replacen(
            "mhc_post(attn_hc, attn_post)",
            "mhc_post(ffn_hc, attn_post)",
            1,
        ),
    ] {
        assert!(definition(&source).is_err());
    }
}
#[test]
fn changing_dsl_changes_program_not_a_model_selector() {
    let bypass = SOURCE.replace("range(0, 46)", "range(0, 0)");
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

#[test]
fn reusable_model_expands_to_the_qualified_primitive_graph() {
    let ModelDefinition::Execution { graph, .. } = definition(SOURCE).unwrap() else {
        panic!()
    };
    let baseline: authoring::execution::Graph = serde_json::from_str(include_str!(
        "../../../docs/validation/2026-10-15-dsl-execution/execution.json"
    ))
    .unwrap();
    assert_eq!(graph.output, baseline.output);
    assert_eq!(graph.max_context, baseline.max_context);
    assert_eq!(graph.eos, baseline.eos);
    assert_eq!(graph.nodes.len(), baseline.nodes.len());
    for (actual, expected) in graph.nodes.iter().zip(&baseline.nodes) {
        assert_eq!(actual.op, expected.op, "{}", actual.name);
        assert_eq!(actual.inputs, expected.inputs, "{}", actual.name);
        assert_eq!(actual.attrs, expected.attrs, "{}", actual.name);
        let source = actual.source.as_ref().unwrap();
        assert!(source.start < source.end && source.end <= SOURCE.len());
    }
    assert!(SOURCE.lines().count() < 150);
}
