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
    assert_eq!(graph.nodes.len(), 1837);
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
    let mut wrong_route = graph.clone();
    let other_route = wrong_route
        .nodes
        .iter()
        .position(|n| n.op == "router")
        .unwrap();
    let down = wrong_route
        .nodes
        .iter_mut()
        .rev()
        .find(|n| n.op == "expert_swiglu_down")
        .unwrap();
    *down.inputs.last_mut().unwrap() = other_route;
    assert!(
        wrong_route
            .validate()
            .unwrap_err()
            .to_string()
            .contains("provenance")
    );
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
        SOURCE.replacen("gate_up, routes", "gate_up, x", 1),
        SOURCE.replacen(
            "paired_weight = expert.up_proj",
            "paired_weight = w.shared_experts.up_proj",
            1,
        ),
        SOURCE.replacen("yield logits;", "yield normalized;", 1),
        SOURCE.replace("context = 512", "context = 513"),
        SOURCE.replace("max_num_seqs = 1", "max_num_seqs = 4"),
        SOURCE.replacen(
            "mhc_post(attn_hc, attn_post)",
            "mhc_post(ffn_hc, attn_post)",
            1,
        ),
    ] {
        assert_ne!(source, SOURCE, "negative fixture must mutate the source");
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
fn reusable_model_preserves_layers_with_explicit_communication_layouts() {
    let ModelDefinition::Execution { graph, .. } = definition(SOURCE).unwrap() else {
        panic!()
    };
    let types = graph.validate().unwrap();
    let baseline: authoring::execution::Graph = serde_json::from_str(include_str!(
        "../../../docs/validation/2026-10-15-dsl-execution/execution.json"
    ))
    .unwrap();
    assert_eq!(graph.max_context, baseline.max_context);
    assert_eq!(graph.eos, baseline.eos);
    assert_eq!(baseline.nodes.len() - graph.nodes.len(), 272);
    for op in [
        "linear",
        "mhc_pre",
        "mhc_post",
        "mome",
        "router",
        "expert_swiglu_down",
        "latent_attention",
    ] {
        assert_eq!(
            graph.nodes.iter().filter(|n| n.op == op).count(),
            baseline.nodes.iter().filter(|n| n.op == op).count()
        );
    }
    assert_eq!(graph.nodes.iter().filter(|n| n.op == "gather").count(), 1);
    assert_eq!(
        graph.nodes.iter().filter(|n| n.op == "allreduce").count(),
        92
    );
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.attrs.get("placement").is_some_and(|p| p == "tp_columns"))
            .count(),
        48
    );
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.attrs.get("placement").is_some_and(|p| p == "tp_channels"))
            .count(),
        46
    );
    let mut shared = 0;
    for node in &graph.nodes {
        let source = node.source.as_ref().unwrap();
        assert!(source.start < source.end && source.end <= SOURCE.len());
        if node.op == "mla_query" {
            assert_eq!(node.attrs["input_layout"], "rank_local");
            assert_eq!(
                types[node.inputs[0]],
                authoring::execution::Type::Tensor(1152)
            );
            assert_eq!(graph.nodes[node.inputs[0]].attrs["placement"], "tp_rows");
        }
        if node.op == "linear" && node.attrs["weight"].contains("shared_experts") {
            shared += 1;
            assert_eq!(node.attrs["placement"], "replicated");
        }
    }
    assert_eq!(shared, 132);
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|n| n.op == "expert_linear")
            .count(),
        44
    );
    for n in graph.nodes.iter().filter(|n| n.op == "expert_linear") {
        assert!(n.attrs["paired_weight"].contains("{expert}"));
    }
    for source in [
        SOURCE.replace(
            "input_layout = \"rank_local\"",
            "input_layout = \"replicated\"",
        ),
        SOURCE.replace(
            "input_layout = \"rank_local\"",
            "input_layout = \"unknown\"",
        ),
    ] {
        assert!(definition(&source).is_err());
    }
    assert!(SOURCE.lines().count() < 150);
}
