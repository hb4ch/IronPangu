use super::*;
type ConvPrepare = unsafe extern "C" fn(SessionPtr, u64, u64, u64, u64, i64, i64, *mut u64) -> i32;
pub fn run_conv(checkpoint_dir: &Path, library: &Path, device: i32) -> Result<serde_json::Value> {
    let checkpoint = inferfabric_compiler::checkpoint::inspect(checkpoint_dir)?;
    let key = format!("{PREFIX}layers.0.linear_attn.conv1d.weight");
    let (weight, digest, c, _) = load_weight(checkpoint_dir, &checkpoint, &key)?;
    let api = Api::load(library)?;
    let prepare: ConvPrepare = unsafe {
        *api._library
            .get(b"inferfabric_acl_conv_prepare\0")
            .map_err(|e| invalid(e.to_string()))?
    };
    let mut reports = vec![];
    for batch in [1usize, 2] {
        let session = Session::open(&api, device)?;
        let x = session.allocate(batch * c * 2)?;
        let w = session.allocate(c * 4 * 2)?;
        let history = session.allocate(batch * c * 3 * 2)?;
        let out = session.allocate(batch * c * 2)?;
        session.write(w, &weight)?;
        let initial: Vec<_> = (0..batch * c * 3)
            .map(|i| bf16(((i * 7 % 41) as f32 - 20.) / 32.))
            .collect();
        session.write(history, &initial)?;
        session.write(x, &vec![0; batch * c])?;
        let mut operation = 0;
        api.check(unsafe {
            prepare(
                session.ptr,
                x,
                w,
                history,
                out,
                batch as i64,
                c as i64,
                &mut operation,
            )
        })?;
        let mut graph = 0;
        api.check(unsafe { (api.capture)(session.ptr, operation, &mut graph) })?;
        let mut carried = initial;
        let mut worst = 0f32;
        let mut token = 0;
        for chunk in [1usize, 3, 2] {
            for _ in 0..chunk {
                let input: Vec<_> = (0..batch * c)
                    .map(|i| bf16(((i * 13 + token * 17) % 53) as f32 / 16. - 1.5))
                    .collect();
                session.write(x, &input)?;
                session.write(history, &carried)?;
                api.check(unsafe { (api.execute)(session.ptr, operation) })?;
                let eager = session.read(out, batch * c)?;
                let eager_history = session.read(history, batch * c * 3)?;
                session.write(history, &carried)?;
                session.write(out, &vec![bf16(-123.); batch * c])?;
                api.check(unsafe { (api.replay)(session.ptr, graph) })?;
                let replay = session.read(out, batch * c)?;
                let next = session.read(history, batch * c * 3)?;
                if eager != replay || eager_history != next {
                    return Err(invalid("convolution graph/eager mismatch"));
                }
                for i in 0..batch * c {
                    let mut sum = 0f32;
                    for j in 0..3 {
                        sum += float(carried[i * 3 + j]) * float(weight[(i % c) * 4 + j]);
                    }
                    sum += float(input[i]) * float(weight[(i % c) * 4 + 3]);
                    let rounded = float(bf16(sum));
                    let expected = float(bf16(rounded / (1. + (-rounded).exp())));
                    let actual = float(replay[i]);
                    let err = (actual - expected).abs();
                    if !actual.is_finite() || err > 0.001 + 0.01 * expected.abs() {
                        return Err(invalid(format!(
                            "convolution mismatch {actual} vs {expected}"
                        )));
                    }
                    worst = worst.max(err);
                    if next[i * 3..i * 3 + 3] != [carried[i * 3 + 1], carried[i * 3 + 2], input[i]]
                    {
                        return Err(invalid("convolution raw history mismatch"));
                    }
                }
                carried = next;
                token += 1;
            }
        }
        session.close()?;
        reports.push(serde_json::json!({"batch":batch,"channels":c,"tokens":token,"chunks":[1,3,2],"max_error":worst,"graph_eager":"bitwise_equal","history":"exact_raw_last_three"}));
        eprintln!("convolution batch={batch} channels={c} passed");
    }
    Ok(
        serde_json::json!({"device":device,"weight":key,"weight_sha256":digest,"native_sha256":sha(&std::fs::read(library)?),"cases":reports}),
    )
}
