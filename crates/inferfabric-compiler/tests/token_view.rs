use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_compiler::token_view::render;

fn graph() -> Graph {
    serde_json::from_str(include_str!(
        "../../../docs/validation/2026-10-16-reusable-dsl/execution.json"
    ))
    .unwrap()
}

#[test]
fn viewer_embeds_validated_types_and_preserves_provenance() {
    let graph = graph();
    let html = render(&graph).unwrap();
    assert_eq!(html, render(&graph).unwrap());
    let json = html
        .split("id=\"plan-data\">")
        .nth(1)
        .unwrap()
        .split("</script>")
        .next()
        .unwrap();
    let data: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(data["graph"], serde_json::to_value(&graph).unwrap());
    assert_eq!(data["types"].as_array().unwrap().len(), 2109);
    assert_eq!(data["node_metadata"].as_array().unwrap().len(), 2109);
    let sink = graph
        .nodes
        .iter()
        .position(|n| n.op == "sink_normalize")
        .unwrap();
    assert_eq!(
        data["node_metadata"][sink]["bindings"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    for (i, ty) in graph.validate().unwrap().iter().enumerate() {
        match ty {
            Type::Tensor(w) | Type::Experts(w) => {
                assert_eq!(data["types"][i]["width"], *w);
            }
            _ => assert!(data["types"][i].get("width").is_none()),
        }
    }
}

#[test]
fn arbitrary_scope_names_and_missing_provenance_are_supported() {
    let mut graph: Graph =
        serde_json::from_str(include_str!("fixtures/viewer-scopes.json")).unwrap();
    let html = render(&graph).unwrap();
    assert_eq!(html, render(&graph).unwrap());
    assert!(html.contains("RenamedScopeFixture"));
    for node in &mut graph.nodes {
        node.source = None;
    }
    assert!(render(&graph).is_ok());
}

#[test]
fn viewer_rejects_invalid_graph_and_escapes_embedded_html() {
    let mut graph = graph();
    graph.name = "</script><script>alert('x')</script>&".into();
    let html = render(&graph).unwrap();
    assert!(!html.contains(&graph.name));
    assert!(html.contains("\\u003c/script\\u003e"));
    graph.nodes[0].inputs.push(1);
    assert!(render(&graph).is_err());
}
