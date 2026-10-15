use inferfabric_dsl::{
    authoring::{self, ModelAuthor, ModelDefinition},
    definition, syntax,
};
use inferfabric_model::Result;
use inferfabric_plan::{
    self as plan,
    ir::{Constant, Kernel, LogicalGraph, LogicalNode, Op, Tensor},
};
use std::collections::BTreeMap;
const SOURCE: &str = include_str!("../../../examples/authoring/residual.inferfabric");
struct Compiled;
impl ModelAuthor for Compiled {
    fn define(&self) -> Result<ModelDefinition> {
        Ok(ModelDefinition::Graph {
            graph: LogicalGraph {
                version: 1,
                name: "Residual".into(),
                inputs: vec![Tensor {
                    name: "x".into(),
                    shape: vec![2, 2],
                }],
                constants: vec![Constant {
                    name: "weight".into(),
                    shape: vec![2, 2],
                    data: vec![1., 0., 0., 1.],
                }],
                states: vec![],
                nodes: vec![
                    LogicalNode {
                        name: "projected".into(),
                        op: Op::Matmul,
                        inputs: vec!["x".into(), "weight".into()],
                    },
                    LogicalNode {
                        name: "summed".into(),
                        op: Op::Add,
                        inputs: vec!["projected".into(), "x".into()],
                    },
                    LogicalNode {
                        name: "output".into(),
                        op: Op::Relu,
                        inputs: vec!["summed".into()],
                    },
                ],
                outputs: vec!["output".into()],
                updates: BTreeMap::new(),
            },
        })
    }
}
#[test]
fn dual_authoring_identical_plan_and_execution() {
    let dsl = definition(SOURCE).unwrap();
    let compiled = authoring::elaborate(&Compiled).unwrap();
    assert_eq!(dsl, compiled);
    assert_eq!(
        authoring::encode(&dsl).unwrap(),
        authoring::encode(&compiled).unwrap()
    );
    let ModelDefinition::Graph { graph } = dsl else {
        panic!()
    };
    let normal = plan::compile(graph.clone()).unwrap();
    let debug = plan::compile_with_options(
        graph,
        &plan::CompileOptions {
            debug: true,
            kernel_overrides: BTreeMap::from([("projected".into(), Kernel::MatmulBlockedF32V1)]),
        },
    )
    .unwrap();
    let input = BTreeMap::from([("x".into(), vec![-1., 2., 3., -4.])]);
    let mut a = plan::Executor::new(normal.physical.clone()).unwrap();
    let mut b = plan::Executor::new(debug.physical).unwrap();
    assert_eq!(
        a.run(&input).unwrap().outputs,
        b.run(&input).unwrap().outputs
    );
    let changed = plan::override_kernels(
        &normal.physical,
        &BTreeMap::from([("projected".into(), Kernel::MatmulBlockedF32V1)]),
    )
    .unwrap();
    assert_eq!(changed.steps[0].kernel, Kernel::MatmulBlockedF32V1);
    for overrides in [
        BTreeMap::from([("projected".into(), Kernel::AddF32V1)]),
        BTreeMap::from([("missing".into(), Kernel::MatmulF32V1)]),
    ] {
        assert!(plan::override_kernels(&normal.physical, &overrides).is_err());
    }
    assert_eq!(normal.physical.steps[0].kernel, Kernel::MatmulF32V1);
}
#[test]
fn graph_state_and_debug_optimization() {
    let source = "language inferfabric 1; model Counter { input x: tensor<f32>[1]; state s: tensor<f32>[1] = [0]; const a: tensor<f32>[1] = [2]; const b: tensor<f32>[1] = [3]; let folded = add(a,b); let unused = relu(x); let y = add(s,x); update s = y; yield y; } plan CPU for Counter { target = cpu_f32_v1; }";
    let ModelDefinition::Graph { graph } = definition(source).unwrap() else {
        panic!()
    };
    let normal = plan::compile(graph.clone()).unwrap();
    let debug = plan::compile_with_options(
        graph,
        &plan::CompileOptions {
            debug: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(normal.physical.steps.len(), 1);
    assert_eq!(debug.physical.steps.len(), 3);
    let mut runtime = plan::Executor::new(debug.physical).unwrap();
    let input = BTreeMap::from([("x".into(), vec![2.])]);
    assert_eq!(runtime.run(&input).unwrap().outputs["y"], vec![2.]);
    assert_eq!(runtime.run(&input).unwrap().outputs["y"], vec![4.]);
}
#[test]
fn syntax_and_semantic_rejections() {
    let ast = syntax::parse(SOURCE).unwrap();
    for m in ast.model {
        assert!(!SOURCE[m.start..m.end].is_empty());
    }
    for s in [
        SOURCE.replace("matmul(x, weight)", "matmul(x)"),
        SOURCE.replace("relu(summed)", "relu(missing)"),
        SOURCE.replace("[2, 2]", "[0, 2]"),
        SOURCE.replace("cpu_f32_v1", "ascend_unqualified"),
        SOURCE.replace("yield output;", "yield output; input x: tensor<f32>[2,2];"),
        SOURCE.replace("relu(summed)", "custom(summed)"),
        SOURCE.replace("1, 0, 0, 1", "1e99, 0, 0, 1"),
        SOURCE.replace("for Residual", "for Missing"),
        format!("{SOURCE} module X {{}}"),
    ] {
        assert!(definition(&s).is_err(), "{s}");
    }
    assert!(
        syntax::parse(&format!(
            "language inferfabric 1; model X {{ x = {}0{}; }} plan P for X {{}}",
            "[".repeat(70),
            "]".repeat(70)
        ))
        .is_err()
    );
    let qwen = include_str!("../../../examples/qwen35-2b.inferfabric")
        .replace("Qwen3.5-2B-text", "label # with // comments");
    assert!(definition(&qwen).is_ok());
    let mut artifact: serde_json::Value =
        serde_json::from_slice(&authoring::encode(&definition(SOURCE).unwrap()).unwrap()).unwrap();
    artifact["version"] = 99.into();
    assert!(authoring::decode(&serde_json::to_vec(&artifact).unwrap()).is_err());
}

#[test]
fn compiled_implementation_preserves_state_failure_boundary() {
    struct Add {
        bad: bool,
    }
    impl plan::CompiledKernel for Add {
        fn operation(&self) -> Op {
            Op::Add
        }
        fn execute(&self, _: &[Vec<usize>], inputs: &[Vec<f32>], _: &[usize]) -> Result<Vec<f32>> {
            if self.bad {
                return Ok(vec![f32::NAN]);
            }
            Ok(inputs[0]
                .iter()
                .zip(&inputs[1])
                .map(|(a, b)| a + b)
                .collect())
        }
    }
    let source = "language inferfabric 1; model Counter { input x: tensor<f32>[1]; state s: tensor<f32>[1] = [0]; let y = add(x,s); yield y; update s = y; } plan P for Counter { target = cpu_f32_v1; }";
    let ModelDefinition::Graph { graph } = definition(source).unwrap() else {
        panic!()
    };
    let physical = plan::compile(graph).unwrap().physical;
    let mut runtime = plan::Executor::new(physical.clone()).unwrap();
    runtime
        .bind_implementation("y", Box::new(Add { bad: false }))
        .unwrap();
    let input = BTreeMap::from([("x".into(), vec![2.])]);
    assert_eq!(runtime.run(&input).unwrap().outputs["y"], vec![2.]);
    assert_eq!(runtime.run(&input).unwrap().outputs["y"], vec![4.]);
    assert!(
        runtime
            .bind_implementation("missing", Box::new(Add { bad: false }))
            .is_err()
    );
    assert!(
        runtime
            .bind_implementation("y", Box::new(Add { bad: false }))
            .is_err()
    );
    let mut failed = plan::Executor::new(physical).unwrap();
    failed
        .bind_implementation("y", Box::new(Add { bad: true }))
        .unwrap();
    assert!(failed.run(&input).is_err());
}
