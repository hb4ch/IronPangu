//! Complete first delta decoder layer, qualified independently before model-wide lowering.
use super::recurrent::{read_f32, reference_step, write_f32};
use super::*;
use std::collections::BTreeMap;
type Delta = unsafe extern "C" fn(
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
type Conv = unsafe extern "C" fn(SessionPtr, u64, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Transform = unsafe extern "C" fn(
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
    *mut u64,
) -> i32;
type Gated = unsafe extern "C" fn(SessionPtr, u64, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Point = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i32, *mut u64) -> i32;
type Sequence = unsafe extern "C" fn(SessionPtr, *const u64, u64, *mut u64) -> i32;
pub(super) struct Weight {
    pub(super) handle: u64,
    pub(super) b: Vec<u16>,
    pub(super) f: Vec<f32>,
    pub(super) hash: String,
}
pub(super) fn norm(x: &[u16], weight: &[u16], width: usize) -> Vec<u16> {
    x.chunks(width)
        .flat_map(|row| {
            let mean = row.iter().map(|&v| float(v).powi(2)).sum::<f32>() / width as f32;
            let inv = (mean + 1e-6).sqrt().recip();
            row.iter()
                .zip(weight)
                .map(move |(&v, &w)| bf16((float(v) * inv) * (1. + float(w))))
        })
        .collect()
}
pub(super) fn point(a: &[u16], b: &[u16], multiply: bool) -> Vec<u16> {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            bf16(if multiply {
                float(a) * float(b)
            } else {
                float(a) + float(b)
            })
        })
        .collect()
}
pub(super) fn silu(v: u16) -> u16 {
    let x = float(v);
    bf16(x / (1. + (-x).exp()))
}
pub(super) fn cpu_layer(
    input: &[u16],
    w: &BTreeMap<String, Weight>,
    history: &mut [u16],
    state: &mut [f32],
    batch: usize,
) -> Vec<u16> {
    let get = |key: &str| &w[key].b;
    let x = norm(input, get("input_layernorm.weight"), 2048);
    let proj = |x: &[u16], key: &str, n, k| reference(x, get(key), batch, n, k);
    let qkv = proj(&x, "linear_attn.in_proj_qkv.weight", 6144, 2048);
    let a = proj(&x, "linear_attn.in_proj_a.weight", 16, 2048);
    let b = proj(&x, "linear_attn.in_proj_b.weight", 16, 2048);
    let z = proj(&x, "linear_attn.in_proj_z.weight", 2048, 2048);
    let cw = get("linear_attn.conv1d.weight");
    let mut conv = vec![0; batch * 6144];
    for i in 0..conv.len() {
        let mut sum = 0f32;
        for j in 0..3 {
            sum += float(history[i * 3 + j]) * float(cw[(i % 6144) * 4 + j]);
        }
        sum += float(qkv[i]) * float(cw[(i % 6144) * 4 + 3]);
        conv[i] = silu(bf16(sum));
        history.copy_within(i * 3 + 1..i * 3 + 3, i * 3);
        history[i * 3 + 2] = qkv[i];
    }
    let mut q = vec![0; batch * 2048];
    let mut k = q.clone();
    let mut v = q.clone();
    for r in 0..batch {
        for h in 0..16 {
            for section in 0..3 {
                let row = &conv[r * 6144 + section * 2048 + h * 128
                    ..r * 6144 + section * 2048 + (h + 1) * 128];
                let dest = if section == 0 {
                    &mut q
                } else if section == 1 {
                    &mut k
                } else {
                    &mut v
                };
                if section == 2 {
                    dest[r * 2048 + h * 128..r * 2048 + (h + 1) * 128].copy_from_slice(row);
                    continue;
                }
                let sum = float(bf16(
                    row.iter().map(|&v| float(bf16(float(v) * float(v)))).sum(),
                ));
                let inv = float(bf16(float(bf16(sum + 1e-6)).sqrt().recip()));
                for (j, &element) in row.iter().enumerate() {
                    dest[r * 2048 + h * 128 + j] = bf16(float(element) * inv);
                }
            }
        }
    }
    let mut g = vec![0.; batch * 16];
    let mut beta = vec![0; batch * 16];
    for i in 0..g.len() {
        let x = float(a[i]) + float(get("linear_attn.dt_bias")[i % 16]);
        let soft = if x > 20. { x } else { x.exp().ln_1p() };
        g[i] = -w["linear_attn.A_log"].f[i % 16].exp() * soft;
        beta[i] = bf16(1. / (1. + (-float(b[i])).exp()));
    }
    let core = reference_step(state, &q, &k, &v, &g, &beta, 128, 128);
    let mut gated = vec![0; core.len()];
    for h in 0..batch * 16 {
        let row = &core[h * 128..(h + 1) * 128];
        let mean = row.iter().map(|&v| float(v).powi(2)).sum::<f32>() / 128.;
        let inv = (mean + 1e-6).sqrt().recip();
        for (j, &element) in row.iter().enumerate() {
            let index = h * 128 + j;
            let zf = float(z[index]);
            gated[index] = bf16(
                (float(bf16(float(element) * inv)) * w["linear_attn.norm.weight"].f[j])
                    * (zf / (1. + (-zf).exp())),
            );
        }
    }
    let attn = proj(&gated, "linear_attn.out_proj.weight", 2048, 2048);
    let residual = point(input, &attn, false);
    let ff = norm(&residual, get("post_attention_layernorm.weight"), 2048);
    let gate = proj(&ff, "mlp.gate_proj.weight", 6144, 2048)
        .into_iter()
        .map(silu)
        .collect::<Vec<_>>();
    let up = proj(&ff, "mlp.up_proj.weight", 6144, 2048);
    let activated = point(&gate, &up, true);
    let down = proj(&activated, "mlp.down_proj.weight", 2048, 6144);
    point(&residual, &down, false)
}
pub fn run_layer(checkpoint_dir: &Path, library: &Path, device: i32) -> Result<serde_json::Value> {
    let checkpoint = inferfabric_compiler::checkpoint::inspect(checkpoint_dir)?;
    let api = Api::load(library)?;
    unsafe {
        let delta: Delta = *api
            ._library
            .get(b"inferfabric_acl_delta_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let conv_fn: Conv = *api
            ._library
            .get(b"inferfabric_acl_conv_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let transform: Transform = *api
            ._library
            .get(b"inferfabric_acl_delta_transforms_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let gated_fn: Gated = *api
            ._library
            .get(b"inferfabric_acl_gated_rms_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let pw: Point = *api
            ._library
            .get(b"inferfabric_acl_pointwise_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let seq: Sequence = *api
            ._library
            .get(b"inferfabric_acl_sequence_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let batch = 1usize;
        let session = Session::open(&api, device)?;
        let mut weights = BTreeMap::new();
        let prefix = format!("{PREFIX}layers.0.");
        for (key, meta) in checkpoint
            .weights
            .iter()
            .filter(|(key, _)| key.starts_with(&prefix))
        {
            let mut file = File::open(checkpoint_dir.join(&meta.shard))?;
            file.seek(SeekFrom::Start(meta.offset))?;
            let mut bytes = vec![0; meta.bytes as usize];
            file.read_exact(&mut bytes)?;
            let hash = sha(&bytes);
            let handle = session.allocate(bytes.len())?;
            api.check((api.write)(
                session.ptr,
                handle,
                0,
                bytes.as_ptr().cast(),
                bytes.len() as u64,
            ))?;
            let (b, f) = if meta.dtype == Dtype::BF16 {
                (
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| u16::from_le_bytes(*v))
                        .collect(),
                    vec![],
                )
            } else {
                (
                    vec![],
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect(),
                )
            };
            weights.insert(
                key[prefix.len()..].to_string(),
                Weight { handle, b, f, hash },
            );
        }
        let weight = |name: &str| weights[name].handle;
        let alloc = |count| session.allocate(count * 2);
        let input = alloc(batch * 2048)?;
        let normed = alloc(batch * 2048)?;
        let qkv = alloc(batch * 6144)?;
        let a = alloc(batch * 16)?;
        let b = alloc(batch * 16)?;
        let z = alloc(batch * 2048)?;
        let history = alloc(batch * 6144 * 3)?;
        let convolved = alloc(batch * 6144)?;
        let q = alloc(batch * 2048)?;
        let k = alloc(batch * 2048)?;
        let v = alloc(batch * 2048)?;
        let g = session.allocate(batch * 16 * 4)?;
        let beta = alloc(batch * 16)?;
        let state = session.allocate(batch * 16 * 128 * 128 * 4)?;
        let core = alloc(batch * 2048)?;
        let gated = alloc(batch * 2048)?;
        let attn = alloc(batch * 2048)?;
        let residual = alloc(batch * 2048)?;
        let ffn = alloc(batch * 2048)?;
        let gate = alloc(batch * 6144)?;
        let up = alloc(batch * 6144)?;
        let activated = alloc(batch * 6144)?;
        let product = alloc(batch * 6144)?;
        let down = alloc(batch * 2048)?;
        let output = alloc(batch * 2048)?;
        let gamma = |name: &str| -> Result<u64> {
            let values: Vec<_> = weights[name].b.iter().map(|&v| 1. + float(v)).collect();
            let id = session.allocate(values.len() * 4)?;
            write_f32(&session, id, &values)?;
            Ok(id)
        };
        let gamma_in = gamma("input_layernorm.weight")?;
        let gamma_post = gamma("post_attention_layernorm.weight")?;
        let mut ops = vec![];
        macro_rules! prepare {
            ($expr:expr) => {{
                let mut id = 0;
                api.check($expr(&mut id))?;
                ops.push(id);
            }};
        }
        prepare!(|id| (api.norm_prepare)(
            session.ptr,
            input,
            gamma_in,
            normed,
            batch as i64,
            2048,
            id
        ));
        for (out, name, n) in [
            (qkv, "linear_attn.in_proj_qkv.weight", 6144),
            (a, "linear_attn.in_proj_a.weight", 16),
            (b, "linear_attn.in_proj_b.weight", 16),
            (z, "linear_attn.in_proj_z.weight", 2048),
        ] {
            prepare!(|id| (api.prepare)(
                session.ptr,
                normed,
                weight(name),
                out,
                batch as i64,
                n,
                2048,
                id
            ));
        }
        prepare!(|id| conv_fn(
            session.ptr,
            qkv,
            weight("linear_attn.conv1d.weight"),
            history,
            convolved,
            batch as i64,
            6144,
            id
        ));
        prepare!(|id| transform(
            session.ptr,
            convolved,
            a,
            b,
            weight("linear_attn.A_log"),
            weight("linear_attn.dt_bias"),
            q,
            k,
            v,
            g,
            beta,
            batch as i64,
            id
        ));
        prepare!(|id| delta(
            session.ptr,
            q,
            k,
            v,
            g,
            beta,
            state,
            core,
            (batch * 16) as i64,
            128,
            128,
            id
        ));
        prepare!(|id| gated_fn(
            session.ptr,
            core,
            z,
            weight("linear_attn.norm.weight"),
            gated,
            (batch * 16) as i64,
            128,
            id
        ));
        prepare!(|id| (api.prepare)(
            session.ptr,
            gated,
            weight("linear_attn.out_proj.weight"),
            attn,
            batch as i64,
            2048,
            2048,
            id
        ));
        prepare!(|id| pw(
            session.ptr,
            input,
            attn,
            residual,
            (batch * 2048) as i64,
            0,
            id
        ));
        prepare!(|id| (api.norm_prepare)(
            session.ptr,
            residual,
            gamma_post,
            ffn,
            batch as i64,
            2048,
            id
        ));
        for (out, name) in [(gate, "mlp.gate_proj.weight"), (up, "mlp.up_proj.weight")] {
            prepare!(|id| (api.prepare)(
                session.ptr,
                ffn,
                weight(name),
                out,
                batch as i64,
                6144,
                2048,
                id
            ));
        }
        prepare!(|id| pw(
            session.ptr,
            gate,
            0,
            activated,
            (batch * 6144) as i64,
            2,
            id
        ));
        prepare!(|id| pw(
            session.ptr,
            activated,
            up,
            product,
            (batch * 6144) as i64,
            1,
            id
        ));
        prepare!(|id| (api.prepare)(
            session.ptr,
            product,
            weight("mlp.down_proj.weight"),
            down,
            batch as i64,
            2048,
            6144,
            id
        ));
        prepare!(|id| pw(
            session.ptr,
            residual,
            down,
            output,
            (batch * 2048) as i64,
            0,
            id
        ));
        let mut sequence = 0;
        api.check(seq(
            session.ptr,
            ops.as_ptr(),
            ops.len() as u64,
            &mut sequence,
        ))?;
        let mut native_history = vec![0; batch * 6144 * 3];
        let mut native_state = vec![0.; batch * 16 * 128 * 128];
        let mut cpu_history = native_history.clone();
        let mut cpu_state = native_state.clone();
        session.write(input, &vec![0; batch * 2048])?;
        session.write(history, &native_history)?;
        write_f32(&session, state, &native_state)?;
        let mut graph = 0;
        api.check((api.capture)(session.ptr, sequence, &mut graph))?;
        let mut results = vec![];
        for token in 0..3 {
            let x: Vec<_> = (0..batch * 2048)
                .map(|i| bf16((((i * 11 + token * 17) % 73) as f32 - 36.) / 64.))
                .collect();
            session.write(input, &x)?;
            session.write(history, &native_history)?;
            write_f32(&session, state, &native_state)?;
            api.check((api.execute)(session.ptr, sequence))?;
            let eager = session.read(output, batch * 2048)?;
            let eager_h = session.read(history, native_history.len())?;
            let eager_s = read_f32(&session, state, native_state.len())?;
            session.write(history, &native_history)?;
            write_f32(&session, state, &native_state)?;
            session.write(output, &vec![bf16(-123.); batch * 2048])?;
            api.check((api.replay)(session.ptr, graph))?;
            let actual = session.read(output, batch * 2048)?;
            native_history = session.read(history, native_history.len())?;
            native_state = read_f32(&session, state, native_state.len())?;
            if actual != eager || native_history != eager_h || native_state != eager_s {
                return Err(invalid("delta layer graph/eager mismatch"));
            }
            let expected = cpu_layer(&x, &weights, &mut cpu_history, &mut cpu_state, batch);
            let mut max_error = 0f32;
            let mut squared_error = 0f64;
            let mut squared_ref = 0f64;
            for (&a, &e) in actual.iter().zip(&expected) {
                let a = float(a);
                let e = float(e);
                if !a.is_finite() {
                    return Err(invalid("nonfinite layer output"));
                }
                max_error = max_error.max((a - e).abs());
                squared_error += ((a - e) as f64).powi(2);
                squared_ref += (e as f64).powi(2);
            }
            let relative_l2 = (squared_error / squared_ref.max(1e-20)).sqrt();
            if relative_l2 > 0.025 {
                return Err(invalid(format!(
                    "delta layer relative L2 mismatch {relative_l2}; max_error {max_error}"
                )));
            }
            results.push(serde_json::json!({"token":token,"max_error":max_error,"relative_l2":relative_l2,"graph_eager":"bitwise_equal"}));
            eprintln!("delta layer token={token} relative_l2={relative_l2} max_error={max_error}");
        }
        session.close()?;
        Ok(
            serde_json::json!({"device":device,"layer":0,"batch":batch,"operations":ops.len(),"results":results,"weights":weights.iter().map(|(k,v)|(k,&v.hash)).collect::<BTreeMap<_,_>>(),"native_sha256":sha(&std::fs::read(library)?),"scope":"complete first delta decoder layer; not full model inference"}),
        )
    }
}
