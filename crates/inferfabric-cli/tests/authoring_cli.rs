use std::{fs, process::Command};
#[test]
fn dsl_artifact_debug_and_tuned_execution() {
    let dir = std::env::temp_dir().join(format!("inferfabric-authoring-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let exe = env!("CARGO_BIN_EXE_inferfabric");
    let source = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/authoring/residual.inferfabric"
    );
    let file = |n: &str| dir.join(n).to_str().unwrap().to_string();
    let run = |args: &[&str]| {
        let r = Command::new(exe).args(args).output().unwrap();
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    };
    run(&["model-source", source, &file("model.json")]);
    run(&["plan", source, &file("dsl.bin"), "--dump-ir", &file("ir")]);
    run(&["plan", &file("model.json"), &file("compiled.bin")]);
    assert_eq!(
        fs::read(file("dsl.bin")).unwrap(),
        fs::read(file("compiled.bin")).unwrap()
    );
    run(&["plan-debug", source, &file("debug.bin")]);
    fs::write(
        file("overrides.json"),
        r#"{"projected":"matmul_blocked_f32_v1"}"#,
    )
    .unwrap();
    run(&[
        "tune-plan",
        &file("debug.bin"),
        &file("overrides.json"),
        &file("tuned.bin"),
    ]);
    fs::write(file("inputs.json"), r#"[{"x":[-1,2,3,-4]}]"#).unwrap();
    run(&[
        "execute",
        &file("tuned.bin"),
        &file("inputs.json"),
        &file("results.json"),
    ]);
    run(&["explain", &file("tuned.bin"), "--html", &file("plan.html")]);
    fs::write(file("legacy.inferfabric"), "model = old\nhidden = 2048\n").unwrap();
    assert!(
        !Command::new(exe)
            .args([
                "model-source",
                &file("legacy.inferfabric"),
                &file("must-not-exist")
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
    assert!(!dir.join("must-not-exist").exists());
    fs::remove_dir_all(dir).unwrap();
}
