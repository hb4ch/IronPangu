use super::recurrent::write_f32;
use super::*;
type Qk =
    unsafe extern "C" fn(SessionPtr, u64, u64, u64, u64, u64, u64, u64, u64, i64, *mut u64) -> i32;
type Attention = unsafe extern "C" fn(
    SessionPtr,
    u64,
    u64,
    u64,
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
type Sequence = unsafe extern "C" fn(SessionPtr, *const u64, u64, *mut u64) -> i32;
fn write_i64(s: &Session<'_>, id: u64, v: &[i64]) -> Result<()> {
    s.api.check(unsafe {
        (s.api.write)(
            s.ptr,
            id,
            0,
            v.as_ptr().cast(),
            std::mem::size_of_val(v) as u64,
        )
    })
}
pub(super) fn qk_ref(
    input: &[u16],
    weights: &[u16],
    heads: usize,
    batch: usize,
    packed: bool,
    position: usize,
) -> Vec<u16> {
    let mut out = vec![0; batch * heads * 256];
    let stride = if packed { 512 } else { 256 };
    for r in 0..batch {
        for h in 0..heads {
            let x = &input[(r * heads + h) * stride..(r * heads + h) * stride + 256];
            let mean = x.iter().map(|&v| float(v).powi(2)).sum::<f32>() / 256.;
            let inv = (mean + 1e-6).sqrt().recip();
            let target = &mut out[(r * heads + h) * 256..(r * heads + h + 1) * 256];
            for j in 0..256 {
                target[j] = bf16((float(x[j]) * inv) * (1. + float(weights[j])));
            }
            for j in 0..32 {
                let angle = position as f32 / (10000000f32.powf(j as f32 / 32.));
                let c = float(bf16(angle.cos()));
                let s = float(bf16(angle.sin()));
                let re = float(target[j]);
                let im = float(target[j + 32]);
                target[j] = bf16(float(bf16(re * c)) - float(bf16(im * s)));
                target[j + 32] = bf16(float(bf16(im * c)) + float(bf16(re * s)));
            }
        }
    }
    out
}
pub fn run_attention(
    checkpoint_dir: &Path,
    library: &Path,
    device: i32,
) -> Result<serde_json::Value> {
    let checkpoint = inferfabric_compiler::checkpoint::inspect(checkpoint_dir)?;
    let (qweight, qhash, _, _) = load_weight(
        checkpoint_dir,
        &checkpoint,
        &format!("{PREFIX}layers.3.self_attn.q_norm.weight"),
    )?;
    let (kweight, khash, _, _) = load_weight(
        checkpoint_dir,
        &checkpoint,
        &format!("{PREFIX}layers.3.self_attn.k_norm.weight"),
    )?;
    let api = Api::load(library)?;
    unsafe {
        let qk: Qk = *api
            ._library
            .get(b"inferfabric_acl_attention_qk_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let attention: Attention = *api
            ._library
            .get(b"inferfabric_acl_paged_attention_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = *api
            ._library
            .get(b"inferfabric_acl_sequence_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let batch = 2usize;
        let page = 4usize;
        let slots = 24usize;
        let context = 12usize;
        let s = Session::open(&api, device)?;
        let qp = s.allocate(batch * 4096 * 2)?;
        let ki = s.allocate(batch * 512 * 2)?;
        let vi = s.allocate(batch * 512 * 2)?;
        let q = s.allocate(batch * 2048 * 2)?;
        let k = s.allocate(batch * 512 * 2)?;
        let gate = s.allocate(batch * 2048 * 2)?;
        let kg = s.allocate(256 * 4)?;
        let qg = s.allocate(256 * 4)?;
        write_f32(
            &s,
            kg,
            &kweight.iter().map(|&v| 1. + float(v)).collect::<Vec<_>>(),
        )?;
        write_f32(
            &s,
            qg,
            &qweight.iter().map(|&v| 1. + float(v)).collect::<Vec<_>>(),
        )?;
        let positions = s.allocate(batch * 8)?;
        let append = s.allocate(batch * 8)?;
        let table = s.allocate(batch * context * 8)?;
        let mask = s.allocate(batch * context * 4)?;
        let keys = s.allocate(slots * 512 * 2)?;
        let values = s.allocate(slots * 512 * 2)?;
        let out = s.allocate(batch * 2048 * 2)?;
        let mut pre = 0;
        api.check(qk(
            s.ptr,
            qp,
            ki,
            qg,
            kg,
            positions,
            q,
            k,
            gate,
            batch as i64,
            &mut pre,
        ))?;
        let mut att = 0;
        api.check(attention(
            s.ptr,
            q,
            k,
            vi,
            gate,
            keys,
            values,
            append,
            table,
            mask,
            out,
            batch as i64,
            slots as i64,
            context as i64,
            &mut att,
        ))?;
        let mut op = 0;
        api.check(sequence(s.ptr, [pre, att].as_ptr(), 2, &mut op))?;
        s.write(qp, &vec![0; batch * 4096])?;
        s.write(ki, &vec![0; batch * 512])?;
        s.write(vi, &vec![0; batch * 512])?;
        write_i64(&s, positions, &[0, 0])?;
        write_i64(&s, append, &[0, 1])?;
        write_i64(&s, table, &vec![0; batch * context])?;
        write_f32(&s, mask, &vec![0.; batch * context])?;
        let poison = vec![bf16(0.75); slots * 512];
        s.write(keys, &poison)?;
        s.write(values, &poison)?;
        let mut graph = 0;
        api.check((api.capture)(s.ptr, op, &mut graph))?;
        s.write(keys, &poison)?;
        s.write(values, &poison)?;
        let mut cpu_keys = poison.clone();
        let mut cpu_values = poison;
        let maps = [[4, 1, 5], [0, 3, 2]];
        let mut reports = vec![];
        for token in 0..6usize {
            let qin: Vec<_> = (0..batch * 4096)
                .map(|i| bf16(((i * 13 + token * 7) % 79) as f32 / 48. - 0.7))
                .collect();
            let kin: Vec<_> = (0..batch * 512)
                .map(|i| bf16(((i * 11 + token * 17) % 67) as f32 / 43. - 0.8))
                .collect();
            let vin: Vec<_> = (0..batch * 512)
                .map(|i| bf16(((i * 17 + token * 5) % 71) as f32 / 41. - 0.9))
                .collect();
            let mut lookup = vec![0i64; batch * context];
            let mut mask_values = vec![f32::NEG_INFINITY; batch * context];
            let mut indices = vec![0; batch];
            for r in 0..batch {
                for j in 0..=token {
                    lookup[r * context + j] = (maps[r][j / page] * page + j % page) as i64;
                    mask_values[r * context + j] = 0.;
                }
                indices[r] = lookup[r * context + token];
            }
            s.write(qp, &qin)?;
            s.write(ki, &kin)?;
            s.write(vi, &vin)?;
            write_i64(&s, positions, &[token as i64, token as i64])?;
            write_i64(&s, append, &indices)?;
            write_i64(&s, table, &lookup)?;
            write_f32(&s, mask, &mask_values)?;
            api.check((api.execute)(s.ptr, op))?;
            let eager = s.read(out, batch * 2048)?;
            let eager_keys = s.read(keys, slots * 512)?;
            let eager_values = s.read(values, slots * 512)?;
            s.write(out, &vec![bf16(-123.); batch * 2048])?;
            api.check((api.replay)(s.ptr, graph))?;
            let actual = s.read(out, batch * 2048)?;
            if actual != eager
                || s.read(keys, slots * 512)? != eager_keys
                || s.read(values, slots * 512)? != eager_values
            {
                return Err(invalid("paged attention graph/eager mismatch"));
            }
            let qr = qk_ref(&qin, &qweight, 8, batch, true, token);
            let kr = qk_ref(&kin, &kweight, 2, batch, false, token);
            let mut expected = vec![0; batch * 2048];
            for r in 0..batch {
                let start = indices[r] as usize * 512;
                cpu_keys[start..start + 512].copy_from_slice(&kr[r * 512..(r + 1) * 512]);
                cpu_values[start..start + 512].copy_from_slice(&vin[r * 512..(r + 1) * 512]);
                for h in 0..8 {
                    let kh = h / 4;
                    let mut scores = vec![0.; token + 1];
                    for j in 0..=token {
                        let slot = lookup[r * context + j] as usize;
                        let mut sum = 0f32;
                        for d in 0..256 {
                            sum += float(qr[(r * 8 + h) * 256 + d])
                                * float(cpu_keys[slot * 512 + kh * 256 + d]);
                        }
                        scores[j] = float(bf16(float(bf16(sum)) * 0.0625));
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum = scores.iter().map(|x| (x - max).exp()).sum::<f32>();
                    let probs: Vec<_> = scores
                        .iter()
                        .map(|x| float(bf16((x - max).exp() / sum)))
                        .collect();
                    for d in 0..256 {
                        let mut sum = 0f32;
                        for j in 0..=token {
                            let slot = lookup[r * context + j] as usize;
                            sum += probs[j] * float(cpu_values[slot * 512 + kh * 256 + d]);
                        }
                        let gf = float(qin[(r * 8 + h) * 512 + 256 + d]);
                        let gate = float(bf16(1. / (1. + (-gf).exp())));
                        expected[(r * 8 + h) * 256 + d] = bf16(float(bf16(sum)) * gate);
                    }
                }
            }
            let mut error = 0f64;
            let mut scale = 0f64;
            let mut max = 0f32;
            for (&a, &e) in actual.iter().zip(&expected) {
                let a = float(a);
                let e = float(e);
                if !a.is_finite() {
                    return Err(invalid("nonfinite attention output"));
                }
                max = max.max((a - e).abs());
                error += ((a - e) as f64).powi(2);
                scale += (e as f64).powi(2);
            }
            let relative_l2 = (error / scale.max(1e-20)).sqrt();
            if relative_l2 > 0.02 {
                return Err(invalid(format!(
                    "attention relative L2 mismatch {relative_l2} max {max}"
                )));
            }
            reports.push(serde_json::json!({"position":token,"relative_l2":relative_l2,"max_error":max,"graph_eager":"bitwise_equal"}));
            eprintln!("paged attention position={token} relative_l2={relative_l2}");
        }
        s.close()?;
        Ok(
            serde_json::json!({"device":device,"batch":batch,"page_tokens":page,"context_bucket":context,"physical_pages":6,"page_maps":maps,"q_norm_sha256":qhash,"k_norm_sha256":khash,"results":reports,"native_sha256":sha(&std::fs::read(library)?),"scope":"QK normalization, partial RoPE, paged causal GQA and output gating"}),
        )
    }
}
