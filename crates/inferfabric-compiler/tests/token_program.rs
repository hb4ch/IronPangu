use inferfabric_authoring::execution::Graph;
use inferfabric_compiler::token_program::lower;

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
