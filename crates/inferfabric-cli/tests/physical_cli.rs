use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
#[test]
fn plan_dump_visualize_and_execute_without_source() {
    let dir = std::env::temp_dir().join(format!(
        "inferfabric-plan-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("source.json"),
        include_bytes!("../../../examples/physical-plan/residual-state.logical.json"),
    )
    .unwrap();
    fs::write(
        dir.join("inputs.json"),
        include_bytes!("../../../examples/physical-plan/inputs.json"),
    )
    .unwrap();
    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_inferfabric"))
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run(&["plan", "source.json", "model.ifplan", "--dump-ir", "ir"]);
    for name in [
        "00-logical.json",
        "01-typed.json",
        "02-optimized.json",
        "03-physical.json",
        "04-bundle-manifest.json",
    ] {
        let _: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("ir").join(name)).unwrap()).unwrap();
    }
    fs::remove_file(dir.join("source.json")).unwrap();
    run(&["explain", "model.ifplan", "--html", "plan.html"]);
    let html = fs::read_to_string(dir.join("plan.html")).unwrap();
    assert!(html.contains("Memory reuse edges"));
    assert!(!html.contains("__PLAN__"));
    run(&["execute", "model.ifplan", "inputs.json", "report.json"]);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(
        report[0]["outputs"]["result"],
        serde_json::json!([10.0, 13.0, 2.0, 2.0])
    );
    assert_eq!(
        report[1]["outputs"]["result"],
        serde_json::json!([20.0, 26.0, 4.0, 4.0])
    );
    fs::remove_dir_all(dir).unwrap();
}
