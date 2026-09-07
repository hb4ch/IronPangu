//! Stateful recurrence qualification. Reference and native state use explicit key-major layout.
use super::*;
type DeltaPrepare = unsafe extern "C" fn(
    SessionPtr,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i64,
    i64,
    i64,
    *mut u64,
) -> i32;
pub(super) fn write_f32(s: &Session<'_>, id: u64, data: &[f32]) -> Result<()> {
    s.api.check(unsafe {
        (s.api.write)(
            s.ptr,
            id,
            0,
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
        )
    })
}
pub(super) fn read_f32(s: &Session<'_>, id: u64, count: usize) -> Result<Vec<f32>> {
    let mut data = vec![0f32; count];
    s.api.check(unsafe {
        (s.api.read)(s.ptr, id, 0, data.as_mut_ptr().cast(), (count * 4) as u64)
    })?;
    Ok(data)
}
// Keep the reference arguments explicit to match the mathematical recurrence.
#[allow(clippy::too_many_arguments)]
pub(super) fn reference_step(
    state: &mut [f32],
    q: &[u16],
    k: &[u16],
    v: &[u16],
    g: &[f32],
    beta: &[u16],
    dk: usize,
    dv: usize,
) -> Vec<u16> {
    let mut out = vec![0; g.len() * dv];
    for h in 0..g.len() {
        let matrix = &mut state[h * dk * dv..(h + 1) * dk * dv];
        for s in matrix.iter_mut() {
            *s *= g[h].exp();
        }
        for j in 0..dv {
            let mut prediction = 0f32;
            for i in 0..dk {
                prediction += float(k[h * dk + i]) * matrix[i * dv + j];
            }
            let residual = (float(v[h * dv + j]) - prediction) * float(beta[h]);
            for i in 0..dk {
                matrix[i * dv + j] += float(k[h * dk + i]) * residual;
            }
            let mut output = 0f32;
            for i in 0..dk {
                output += (float(q[h * dk + i]) / (dk as f32).sqrt()) * matrix[i * dv + j];
            }
            out[h * dv + j] = bf16(output);
        }
    }
    out
}
pub fn run_delta(library: &Path, device: i32) -> Result<serde_json::Value> {
    let api = Api::load(library)?;
    let prepare: DeltaPrepare = unsafe {
        *api._library
            .get(b"pangu_acl_delta_prepare\0")
            .map_err(|e| invalid(e.to_string()))?
    };
    let mut reports = vec![];
    for (bh, dk, dv) in [(2usize, 128usize, 64usize), (16, 128, 128)] {
        let session = Session::open(&api, device)?;
        let q = session.allocate(bh * dk * 2)?;
        let k = session.allocate(bh * dk * 2)?;
        let v = session.allocate(bh * dv * 2)?;
        let g = session.allocate(bh * 4)?;
        let beta = session.allocate(bh * 2)?;
        let state = session.allocate(bh * dk * dv * 4)?;
        let out = session.allocate(bh * dv * 2)?;
        let initial: Vec<_> = (0..bh * dk * dv)
            .map(|i| ((i * 7 % 43) as f32 - 21.) * 0.0007 + 0.0000003)
            .collect();
        let mut operation = 0;
        api.check(unsafe {
            prepare(
                session.ptr,
                q,
                k,
                v,
                g,
                beta,
                state,
                out,
                bh as i64,
                dk as i64,
                dv as i64,
                &mut operation,
            )
        })?;
        let mut graph = 0;
        // Initialize every captured input and disposable state before warmup.
        session.write(q, &vec![0; bh * dk])?;
        session.write(k, &vec![0; bh * dk])?;
        session.write(v, &vec![0; bh * dv])?;
        session.write(beta, &vec![0; bh])?;
        write_f32(&session, g, &vec![0.; bh])?;
        write_f32(&session, state, &initial)?;
        api.check(unsafe { (api.capture)(session.ptr, operation, &mut graph) })?;
        let mut ref_state = initial.clone();
        let mut native_state = initial.clone();
        let mut worst_state = 0f32;
        let mut worst_output = 0f32;
        let mut chunk_ends = vec![];
        let mut token = 0;
        // Irregular host chunk boundaries deliberately do not reset persistent state.
        for chunk in [2usize, 1, 3] {
            for _ in 0..chunk {
                let qh: Vec<_> = (0..bh * dk)
                    .map(|i| bf16(((i * 17 + token * 3) % 31) as f32 / 180. - 0.083))
                    .collect();
                let kh: Vec<_> = (0..bh * dk)
                    .map(|i| bf16(((i * 13 + token * 5) % 29) as f32 / 170. - 0.08))
                    .collect();
                let vh: Vec<_> = (0..bh * dv)
                    .map(|i| bf16(((i * 7 + token * 11) % 37) as f32 / 31. - 0.5))
                    .collect();
                let gh: Vec<_> = (0..bh)
                    .map(|h| [0., -0.1, -2., -20., -80., -0.9][(token + h) % 6])
                    .collect();
                let bhv: Vec<_> = (0..bh)
                    .map(|h| bf16([0., 1., 0.4][(token + h) % 3]))
                    .collect();
                session.write(q, &qh)?;
                session.write(k, &kh)?;
                session.write(v, &vh)?;
                session.write(beta, &bhv)?;
                write_f32(&session, g, &gh)?;
                // Restore exactly the same input state for eager and graph execution.
                write_f32(&session, state, &native_state)?;
                api.check(unsafe { (api.execute)(session.ptr, operation) })?;
                let eager = session.read(out, bh * dv)?;
                let eager_state = read_f32(&session, state, ref_state.len())?;
                write_f32(&session, state, &native_state)?;
                session.write(out, &vec![bf16(-123.); bh * dv])?;
                api.check(unsafe { (api.replay)(session.ptr, graph) })?;
                let replay = session.read(out, bh * dv)?;
                let replay_state = read_f32(&session, state, ref_state.len())?;
                if eager != replay
                    || eager_state
                        .iter()
                        .zip(&replay_state)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(invalid("delta graph/eager output or state mismatch"));
                }
                let expected = reference_step(&mut ref_state, &qh, &kh, &vh, &gh, &bhv, dk, dv);
                for (&a, &e) in replay_state.iter().zip(&ref_state) {
                    let error = (a - e).abs();
                    if !a.is_finite() || error > 2e-6 + 2e-5 * e.abs() {
                        return Err(invalid(format!("delta FP32 state mismatch {a} vs {e}")));
                    }
                    worst_state = worst_state.max(error);
                }
                for (&a, &e) in replay.iter().zip(&expected) {
                    let error = (float(a) - float(e)).abs();
                    if !float(a).is_finite() || error > 0.0001 + 0.01 * float(e).abs() {
                        return Err(invalid("delta output mismatch"));
                    }
                    worst_output = worst_output.max(error);
                }
                native_state = replay_state;
                token += 1;
            }
            chunk_ends.push(token);
        }
        session.close()?;
        reports.push(serde_json::json!({"bh":bh,"dk":dk,"dv":dv,"tokens":token,"chunk_ends":chunk_ends,"state_dtype":"FP32","state_layout":"head,key,value","minimum_decay":-80,"max_state_error":worst_state,"max_output_error":worst_output,"graph_eager":"bitwise_equal"}));
        eprintln!("delta bh={bh} dk={dk} dv={dv} passed");
    }
    Ok(
        serde_json::json!({"device":device,"cases":reports,"native_sha256":sha(&std::fs::read(library)?),"rust_sha256":sha(&std::fs::read(std::env::current_exe()?)?),"scope":"prepared FP32 recurrence primitive; no complete model inference"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_recurrence_uses_updated_state() {
        let mut state = vec![1., 2., 3., 4., 5., 6.];
        let out = reference_step(
            &mut state,
            &[bf16(1.), bf16(0.)],
            &[bf16(1.), bf16(0.)],
            &[bf16(7.), bf16(8.), bf16(9.)],
            &[0.],
            &[bf16(1.)],
            2,
            3,
        );
        assert_eq!(state, vec![7., 8., 9., 4., 5., 6.]);
        assert_eq!(
            out,
            vec![
                bf16(7. / 2f32.sqrt()),
                bf16(8. / 2f32.sqrt()),
                bf16(9. / 2f32.sqrt())
            ]
        );
    }
}
