use inferfabric_plan::{ir::*, *};
use std::collections::BTreeMap;
fn graph() -> LogicalGraph {
    serde_json::from_str(include_str!(
        "../../../examples/physical-plan/residual-state.logical.json"
    ))
    .unwrap()
}
fn inputs() -> BTreeMap<String, Vec<f32>> {
    BTreeMap::from([("x".into(), vec![1., 2., -1., 1.])])
}
#[test]
fn binary_execution_matches_independent_residual_state_reference() {
    let c = compile(graph()).unwrap();
    assert_eq!(c.physical.folded_nodes, ["scale"]);
    assert_eq!(c.physical.removed_nodes, ["unused"]);
    assert!(
        c.physical
            .steps
            .iter()
            .any(|s| !s.reuse_dependencies.is_empty())
    );
    let bytes = encode(&c.physical).unwrap();
    assert_eq!(bytes, encode(&compile(graph()).unwrap().physical).unwrap());
    drop(c); // No logical or typed graph is supplied to the executor.
    let mut run = Executor::new(decode(&bytes).unwrap()).unwrap();
    let first = run.run(&inputs()).unwrap();
    // xW = [7,10,2,2]; relu(xW+bias)+2*x = [10,13,2,2].
    assert_eq!(first.outputs["result"], [10., 13., 2., 2.]);
    let second = run.run(&inputs()).unwrap();
    assert_eq!(second.outputs["result"], [20., 26., 4., 4.]);
    assert_eq!(second.state["history"], second.outputs["result"]);
    let mut fresh = Executor::new(decode(&bytes).unwrap()).unwrap();
    assert_eq!(fresh.run(&inputs()).unwrap().outputs, first.outputs);
}
#[test]
fn source_order_does_not_change_topological_plan() {
    let a = compile(graph()).unwrap();
    let mut g = graph();
    g.nodes.reverse();
    assert_eq!(a.physical, compile(g).unwrap().physical);
}
#[test]
fn planner_rejects_invalid_graphs() {
    let mut g = graph();
    g.nodes[0].inputs[0] = "missing".into();
    assert!(compile(g).is_err());
    let mut g = graph();
    g.nodes[0].inputs[0] = "result".into();
    assert!(compile(g).is_err());
    let mut g = graph();
    g.inputs[0].shape = vec![2, 3];
    assert!(compile(g).is_err());
    let mut g = graph();
    g.updates.insert("weights".into(), "result".into());
    assert!(compile(g).is_err());
    let mut g = graph();
    g.nodes[0].name = "weights".into();
    assert!(compile(g).is_err());
    let mut g = graph();
    g.constants[0].data[0] = f32::NAN;
    assert!(compile(g).is_err());
    let mut g = graph();
    g.inputs[0].shape = vec![usize::MAX, 2];
    assert!(compile(g).is_err());
    let mut g = graph();
    g.version = 99;
    assert!(compile(g).is_err());
}
#[test]
fn corruption_and_resigned_semantic_mutations_fail() {
    let p = compile(graph()).unwrap().physical;
    let original = encode(&p).unwrap();
    for cut in [0, 8, 47, original.len() - 1] {
        assert!(decode(&original[..cut]).is_err());
    }
    let mut b = original.clone();
    *b.last_mut().unwrap() ^= 1;
    assert!(decode(&b).is_err());
    let mut b = original.clone();
    b.push(0);
    assert!(decode(&b).is_err());
    let mut mutations = vec![];
    let mut q = p.clone();
    q.steps[1].dependencies.clear();
    mutations.push(q);
    let mut q = p.clone();
    q.steps[0].inputs[0] = usize::MAX;
    mutations.push(q);
    let mut q = p.clone();
    q.steps[0].stream = 1;
    mutations.push(q);
    let mut q = p.clone();
    q.target = "ascend".into();
    mutations.push(q);
    let mut q = p.clone();
    q.buffers[q.steps[0].output].offset = Some(usize::MAX);
    mutations.push(q);
    let mut q = p.clone();
    q.buffers[q.steps[0].output].lifetime = Some([0, 0]);
    mutations.push(q);
    let mut q = p.clone();
    let id = q
        .steps
        .iter()
        .position(|s| !s.reuse_dependencies.is_empty())
        .unwrap();
    q.steps[id].reuse_dependencies.clear();
    mutations.push(q);
    let mut q = p.clone();
    q.buffers[q.steps[1].output].offset = q.buffers[q.steps[0].output].offset;
    mutations.push(q);
    for q in mutations {
        use sha2::{Digest, Sha256};
        assert!(verify(&q).is_err());
        assert!(encode(&q).is_err());
        let payload = serde_json::to_vec(&q).unwrap();
        let mut b = b"IFPLAN01".to_vec();
        b.extend((payload.len() as u64).to_le_bytes());
        b.extend(Sha256::digest(&payload));
        b.extend(payload);
        assert!(decode(&b).is_err()); // Valid checksum does not bypass semantic verification.
    }
}
#[test]
fn failed_invocation_does_not_commit_state() {
    let mut run = Executor::new(compile(graph()).unwrap().physical).unwrap();
    assert!(run.run(&BTreeMap::new()).is_err());
    assert!(
        run.run(&BTreeMap::from([("x".into(), vec![f32::MAX; 4])]))
            .is_err()
    );
    assert_eq!(
        run.run(&inputs()).unwrap().outputs["result"],
        [10., 13., 2., 2.]
    );
}
#[test]
fn simultaneous_state_swap_and_constant_only_output() {
    let g = LogicalGraph {
        version: 1,
        name: "swap".into(),
        inputs: vec![],
        constants: vec![],
        states: vec![
            Constant {
                name: "a".into(),
                shape: vec![1],
                data: vec![1.],
            },
            Constant {
                name: "b".into(),
                shape: vec![1],
                data: vec![2.],
            },
        ],
        nodes: vec![],
        outputs: vec!["a".into()],
        updates: BTreeMap::from([("a".into(), "b".into()), ("b".into(), "a".into())]),
    };
    let mut run =
        Executor::new(decode(&encode(&compile(g).unwrap().physical).unwrap()).unwrap()).unwrap();
    let r = run.run(&BTreeMap::new()).unwrap();
    assert_eq!(r.outputs["a"], [1.]);
    assert_eq!(r.state["a"], [2.]);
    assert_eq!(r.state["b"], [1.]);
    assert_eq!(r.arena_bytes, 0);
    let mut g = graph();
    g.outputs = vec!["scale".into()];
    g.updates.clear();
    let c = compile(g).unwrap();
    assert!(c.physical.steps.is_empty());
    assert_eq!(
        Executor::new(c.physical)
            .unwrap()
            .run(&inputs())
            .unwrap()
            .outputs["scale"],
        [2.; 4]
    );
}
#[test]
fn blocked_matmul_handles_rectangular_tails() {
    let (m, k, n) = (17, 13, 19);
    let g = LogicalGraph {
        version: 1,
        name: "rectangular".into(),
        inputs: vec![
            Tensor {
                name: "a".into(),
                shape: vec![m, k],
            },
            Tensor {
                name: "b".into(),
                shape: vec![k, n],
            },
        ],
        constants: vec![],
        states: vec![],
        nodes: vec![LogicalNode {
            name: "out".into(),
            op: Op::Matmul,
            inputs: vec!["a".into(), "b".into()],
        }],
        outputs: vec!["out".into()],
        updates: BTreeMap::new(),
    };
    let p = compile(g).unwrap().physical;
    assert_eq!(p.steps[0].kernel, Kernel::MatmulBlockedF32V1);
    let a: Vec<_> = (0..m * k).map(|i| (i % 7) as f32 - 3.).collect();
    let b: Vec<_> = (0..k * n).map(|i| (i % 5) as f32 - 2.).collect();
    let mut expected = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k)
                .map(|z| a[i * k + z] as f64 * b[z * n + j] as f64)
                .sum::<f64>() as f32;
        }
    }
    let report = Executor::new(p)
        .unwrap()
        .run(&BTreeMap::from([("a".into(), a), ("b".into(), b)]))
        .unwrap();
    assert_eq!(report.outputs["out"], expected);
}
#[test]
fn html_embeds_untrusted_names_without_script_termination() {
    let mut g = graph();
    g.name = "</script><img src=x onerror=alert(1)>".into();
    let html = visualize::html(&compile(g).unwrap().physical).unwrap();
    assert!(!html.contains("</script><img"));
    assert!(html.contains("\\u003c/script\\u003e"));
}
