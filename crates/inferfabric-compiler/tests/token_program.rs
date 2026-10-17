use inferfabric_authoring::execution::Graph;
use inferfabric_compiler::token_program::lower;

#[test]
fn output_rows_proves_real_92b_suffix_without_changing_the_program() {
    use inferfabric_dsl::{authoring::ModelDefinition, definition};
    let ModelDefinition::Execution { mut graph, .. } = definition(include_str!(
        "../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric"
    )).unwrap() else { panic!() };
    graph.max_context = 65536;
    let p = lower(&graph, None).unwrap().with_resident_sequences(3).unwrap()
        .with_uncaptured_prefill().unwrap();
    let before = serde_json::to_vec(&p).unwrap();
    let rows = inferfabric_compiler::output_rows::plan(&p.instructions, p.output).unwrap().unwrap_or_else(|| panic!("output={} tail={:?}", p.output, &p.instructions[p.instructions.len()-6..]));
    let suffix = &p.instructions[rows.first..=rows.last];
    assert!(suffix.iter().any(|n| n.op == "linear"));
    assert_eq!(suffix.last().unwrap().op, "head_to_token");
    assert!(suffix.iter().any(|n| n.op == "gather_heads"));
    assert_eq!(rows.source_width, p.instructions[rows.source].width);
    assert!(p.instructions[..rows.first].iter().any(|n| n.op == "cache_c8_norm_rope_append"));
    assert_eq!(serde_json::to_vec(&p).unwrap(), before);
}

#[test]
fn fanout_reuses_a_cast_without_erasing_rounding() {
    use inferfabric_authoring::execution::Node;
    use std::collections::BTreeMap;
    let mut graph: Graph =
        serde_json::from_str(include_str!("fixtures/viewer-scopes.json")).unwrap();
    // Insert two FP32 reductions of the same embedding, then rounded addition.
    let mut tail = graph.nodes.split_off(1);
    for (k, op, inputs) in [
        (1, "allreduce", vec![0]),
        (2, "allreduce", vec![0]),
        (3, "add", vec![1, 2]),
    ] {
        graph.nodes.push(Node {
            source: None,
            name: format!("fanout{k}"),
            op: op.into(),
            inputs,
            attrs: BTreeMap::new(),
        });
    }
    for n in &mut tail {
        for i in &mut n.inputs {
            *i = if *i == 0 { 3 } else { *i + 3 };
        }
    }
    graph.nodes.extend(tail);
    graph.output += 3;
    let p = lower(&graph, None).unwrap();
    assert_eq!(p.conversions_shared, 1);
    let reductions: Vec<_> = p
        .instructions
        .iter()
        .filter(|n| n.op == "allreduce")
        .collect();
    assert_eq!(reductions[0].inputs, reductions[1].inputs);
}

#[test]
fn conversions_are_explicit_shared_and_rounding_is_preserved() {
    let graph: Graph = serde_json::from_str(include_str!(
        "../../../docs/validation/2026-10-16-rank-layouts/execution.json"
    ))
    .unwrap();
    let p = lower(&graph, None).unwrap();
    assert_eq!(p, lower(&graph, None).unwrap());
    assert!(p.encode().is_err(), "unbound projection must not execute");
    assert!(p.instructions.iter().any(|i| i.op == "cast"));
    let mut conversions = std::collections::BTreeSet::new();
    for (id, i) in p.instructions.iter().enumerate() {
        assert!(i.inputs.iter().all(|&j| j < id));
        if i.op == "cast" {
            assert_ne!(p.instructions[i.inputs[0]].dtype, i.dtype);
            assert!(conversions.insert((i.inputs[0], &i.dtype)));
        }
        if i.op == "add_f32" {
            assert!(i.inputs.iter().all(|&j| p.instructions[j].dtype == "f32"));
            let round = &p.instructions[id + 1];
            assert_eq!(round.op, "cast");
            assert_eq!(round.inputs, vec![id]);
            assert_eq!(round.dtype, "bf16");
        }
        if i.op == "allreduce" {
            assert_eq!(p.instructions[i.inputs[0]].dtype, i.dtype);
        }
    }
    assert_eq!(p.instructions[p.output].width, 151552);
}

#[test]
fn bulk_collective_domain_is_versioned_and_requires_uncaptured_execution() {
    use inferfabric_dsl::{authoring::ModelDefinition, definition};
    let ModelDefinition::Execution { mut graph, .. } = definition(include_str!(
        "../../../examples/openpangu/92b-tp8-ep8-c8-learned.inferfabric"
    )).unwrap() else { panic!() };
    graph.max_context = 65536;
    let p = lower(&graph, None).unwrap();
    assert!(p.clone().with_bulk_collective_domain().is_err());
    let p = p.with_uncaptured_prefill().unwrap().with_bulk_collective_domain().unwrap();
    assert_eq!(p.format, "inferfabric.token-program.v9");
    p.validate().unwrap();
    let recipes = inferfabric_compiler::token_program::recipes(&p, &graph);
    for (i, r) in p.instructions.iter().zip(&recipes) {
        if i.op == "allreduce" {
            assert_eq!(r["collective_domains"]["uncaptured_bulk"], "AI_CPU");
            assert_eq!(r["collective_domains"]["decode"], "HOST");
        }
        if i.op == "gather_heads" { assert_eq!(r["collective_domains"]["uncaptured_bulk"], "HOST"); }
    }
    let mut invalid = p.clone();
    invalid.uncaptured_prefill = false;
    assert!(invalid.validate().is_err());
    let p = p.with_output_rows().unwrap();
    assert_eq!(p.format, "inferfabric.token-program.v9");
    p.validate().unwrap();
}
