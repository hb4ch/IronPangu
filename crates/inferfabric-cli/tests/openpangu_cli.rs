use std::{fs, io::Write, process::Command};
#[test]
fn openpangu_audit_commands_and_failure_do_not_publish_output() {
    let dir =
        std::env::temp_dir().join(format!("inferfabric-openpangu-cli-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    fs::write(
        dir.join("config.json"),
        include_bytes!("../../../examples/openpangu/92b.reference-config.json"),
    )
    .unwrap();
    let run = |command: &str, input: &str, output: &str| {
        Command::new(env!("CARGO_BIN_EXE_inferfabric"))
            .current_dir(&dir)
            .args([command, input, output])
            .output()
            .unwrap()
    };
    let result = run("inspect-openpangu", "config.json", "config-audit.json");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("config-audit.json")).unwrap()).unwrap();
    assert_eq!(manifest["layers"].as_array().unwrap().len(), 46);
    assert_eq!(manifest["executable"], false);
    let h = br#"{"test.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut f = fs::File::create(dir.join("model.safetensors")).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(h).unwrap();
    f.write_all(&[0, 0, 0, 0]).unwrap();
    drop(f);
    let result = run("inspect-openpangu-checkpoint", ".", "checkpoint-audit.json");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("checkpoint-audit.json")).unwrap()).unwrap();
    assert_eq!(manifest["inventory"]["stored_tensor_bytes"], 4);
    assert_eq!(manifest["inventory"]["payload_verified"], false);
    fs::write(dir.join("bad.json"), "{}").unwrap();
    assert!(
        !run("inspect-openpangu", "bad.json", "bad-output.json")
            .status
            .success()
    );
    assert!(!dir.join("bad-output.json").exists());
    fs::remove_dir_all(dir).unwrap();
}
