use inferfabric_compiler::{bound::Plan, model_inspect};
fn fixture() -> Plan {
    serde_json::from_str(include_str!(
        "../../../docs/validation/2026-09-08-qwen-plan/model.typed-plan.json"
    ))
    .unwrap()
}
#[test]
fn full_qwen_layer_partition_and_dependencies() {
    let plan = fixture();
    let view = model_inspect::inspect(&plan).unwrap();
    assert_eq!(view.groups.len(), 26);
    assert_eq!(view.groups.iter().filter(|g| g.kind == "Delta").count(), 18);
    assert_eq!(view.groups.iter().filter(|g| g.kind == "Full").count(), 6);
    assert_eq!(
        view.groups
            .iter()
            .flat_map(|g| g.nodes.iter().copied())
            .collect::<Vec<_>>(),
        (0..640).collect::<Vec<_>>()
    );
    assert_eq!(plan.checkpoint.weights.len(), 320);
    assert_eq!(plan.states.len(), 48);
    assert_eq!(view.groups[0].nodes.len(), 1);
    assert_eq!(view.groups[25].nodes.len(), 3);
    for (layer, group) in view.groups[1..25].iter().enumerate() {
        assert_eq!(group.name, format!("Layer {layer}"));
        let op = &plan.nodes[group.nodes[0]].operation;
        assert!(
            matches!(op,inferfabric_compiler::bound::Op::ZeroCenteredRms {weight,..} if weight.ends_with(&format!("layers.{layer}.input_layernorm.weight")))
        );
    }
}
#[test]
fn inspection_rejects_modified_graph_and_memory_summary() {
    let mut p = fixture();
    p.nodes[1].inputs[0] = "missing".into();
    assert!(model_inspect::inspect(&p).is_err());
    let mut p = fixture();
    p.recurrent_and_conv_bytes_per_request += 1;
    assert!(model_inspect::inspect(&p).is_err());
    let mut p = fixture();
    p.nodes.swap(10, 11);
    assert!(model_inspect::inspect(&p).is_err());
}
#[test]
fn html_embeds_complete_ir_and_escapes_script_boundaries() {
    let mut p = fixture();
    p.spec.model.name = "Qwen </script><script>bad()</script> & \u{2028}".into();
    let html = model_inspect::html(&p).unwrap();
    assert!(!html.contains("<script>bad()"));
    let json = html
        .split("type=\"application/json\">")
        .nth(1)
        .unwrap()
        .split("</script>")
        .next()
        .unwrap();
    let data: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(data["plan"]["nodes"].as_array().unwrap().len(), 640);
    assert_eq!(data["plan"]["spec"]["model"]["name"], p.spec.model.name);
}
