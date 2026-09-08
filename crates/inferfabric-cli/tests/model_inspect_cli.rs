use std::{fs, process::Command};
#[test]
fn explain_full_model_offline() {
    let dir = std::env::temp_dir().join(format!("inferfabric-model-ui-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("model.json"),
        include_bytes!("../../../docs/validation/2026-09-08-qwen-plan/model.typed-plan.json"),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_inferfabric"))
        .current_dir(&dir)
        .args(["explain-model", "model.json", "--html", "model.html"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let html = fs::read_to_string(dir.join("model.html")).unwrap();
    assert!(html.contains("checkpoint-bound typed mathematical DAG"));
    assert!(!html.contains("__MODEL_DATA__"));
    fs::write(dir.join("bad.json"), "{}").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_inferfabric"))
        .current_dir(&dir)
        .args(["explain-model", "bad.json", "--html", "bad.html"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(!dir.join("bad.html").exists());
    fs::remove_dir_all(dir).unwrap();
}
