use inferfabric_dsl::{authoring::ModelDefinition, definition};
const SOURCE: &str = r#"language inferfabric 1;
module Temporal(x, w) {
    let y = mome(x, weight = w, taps = 3);
    yield y;
}
module Project(x, w, width) {
    let local = linear(x, weight = w, width = div(width, 8), placement = "tp_rows");
    let output = gather(local);
    yield output;
}
model Tiny {
    context = 512;
    eos = 148902;
    let w = weights("checkpoint");
    let embedded = embedding(weight = w.embedding, width = 2560, vocab = 151552);
    let hidden = fold i in range(0, 2) carry(x = embedded) {
        let y = Temporal(x, w.shared_conv);
        yield y;
    };
    let logits = Project(hidden, w.head, 151552);
    yield logits;
}
plan P for Tiny {target=ascend_token_v1;tp=8;ep=8;devices=[0,1,2,3,4,5,6,7];max_num_seqs=1;max_num_batched_tokens=1;max_model_len=512;}
"#;
#[test]
fn loop_instances_share_weights_but_not_state_and_keep_source_stack() {
    let ModelDefinition::Execution { graph, .. } = definition(SOURCE).unwrap() else {
        panic!()
    };
    assert_eq!(graph.nodes.len(), 5);
    assert_eq!(graph.nodes[1].attrs["weight"], "checkpoint.shared_conv");
    assert_eq!(graph.nodes[1].attrs, graph.nodes[2].attrs);
    assert_ne!(graph.nodes[1].name, graph.nodes[2].name);
    assert!(graph.nodes[1].name.contains(".i0."));
    assert!(graph.nodes[2].name.contains(".i1."));
    let a = graph.nodes[1].source.as_ref().unwrap();
    let b = graph.nodes[2].source.as_ref().unwrap();
    assert_eq!(a.start, b.start);
    assert_eq!(a.frames.len(), 2);
    assert_eq!(a.frames[1].module, "Temporal");
    assert_ne!(a.frames, b.frames);
    assert!(SOURCE[a.start..a.end].contains("mome"));
    assert_eq!(graph.nodes[3].attrs["width"], "18944");
    let zero = SOURCE.replace("range(0, 2)", "range(0, 0)");
    let ModelDefinition::Execution { graph, .. } = definition(&zero).unwrap() else {
        panic!()
    };
    assert_eq!(graph.nodes.len(), 3);
}
#[test]
fn scoped_bindings_compile_time_branches_and_named_arguments() {
    let s=SOURCE.replace("Temporal(x, w.shared_conv)","choose(eq(mod(i, 2), 0), Temporal(w = child(w.layers, i, \"conv\"), x = x), Temporal(x, w.shared_conv))");
    let ModelDefinition::Execution { graph, .. } = definition(&s).unwrap() else {
        panic!()
    };
    assert_eq!(graph.nodes[1].attrs["weight"], "checkpoint.layers.0.conv");
    assert_eq!(graph.nodes[2].attrs["weight"], "checkpoint.shared_conv");
}
#[test]
fn unsafe_or_ambiguous_expansions_fail() {
    for (old, new) in [
        ("module Temporal(x, w)", "module Temporal(x, x)"),
        ("Temporal(x, w.shared_conv)", "Temporal(x)"),
        (
            "Temporal(x, w.shared_conv)",
            "Temporal(x, w = w.shared_conv, w = w.shared_conv)",
        ),
        (
            "Temporal(x, w.shared_conv)",
            "Temporal(x, missing = w.shared_conv)",
        ),
        (
            "Temporal(x, w.shared_conv)",
            "Temporal(w = w.shared_conv, x)",
        ),
        (
            "Temporal(x, w.shared_conv)",
            "Temporal(x, child(w, \"../escape\"))",
        ),
        (
            "Temporal(x, w.shared_conv)",
            "Temporal(x, child(\"not_scope\", i))",
        ),
        ("weight = w, taps = 3", "weight = x, taps = 3"),
        ("range(0, 2)", "range(0, 4097)"),
        ("range(0, 2)", "range(3, 2)"),
        ("range(0, 2)", "range(0, embedded)"),
        (
            "Temporal(x, w.shared_conv)",
            "choose(x, Temporal(x, w.shared_conv), Temporal(x, w.shared_conv))",
        ),
        ("div(width, 8)", "div(width, 0)"),
        ("div(width, 8)", "div(width, 7)"),
        ("mome(x, weight = w, taps = 3)", "Temporal(x, w)"),
        ("let y = mome", "let x = mome"),
        (
            "yield y;",
            "yield y; let after = mome(x, weight = w, taps = 3);",
        ),
        (
            "mome(x, weight = w, taps = 3)",
            "mome(embedded, weight = w, taps = 3)",
        ),
    ] {
        assert!(
            definition(&SOURCE.replace(old, new)).is_err(),
            "mutation {old} -> {new}"
        );
    }
}
#[test]
fn scalar_only_nested_expansion_is_bounded() {
    let s = SOURCE
        .replace(
            "let y = Temporal(x, w.shared_conv);",
            r#"let y = fold j in range(0, 4096) carry(z = x) { let alias = z; yield alias; };"#,
        )
        .replace("range(0, 2)", "range(0, 4096)");
    let e = definition(&s).unwrap_err().to_string();
    assert!(e.contains("work limit"), "{e}");
}

#[test]
fn indirect_recursion_and_nested_parser_depth_are_rejected() {
    let s = SOURCE
        .replace(
            "module Temporal",
            "module Other(x,w) {let y=Temporal(x,w);yield y;} module Temporal",
        )
        .replace("mome(x, weight = w, taps = 3)", "Other(x,w)");
    assert!(
        definition(&s)
            .unwrap_err()
            .to_string()
            .contains("recursive")
    );
    let mut e = String::from("x");
    for _ in 0..80 {
        e = format!("fold j in range(0,1) carry(z=x) {{let y={e};yield y;}}");
    }
    let s = SOURCE.replace("Temporal(x, w.shared_conv)", &e);
    assert!(
        definition(&s)
            .unwrap_err()
            .to_string()
            .contains("nesting limit")
    );
}
