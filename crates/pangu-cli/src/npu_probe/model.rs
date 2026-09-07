//! Single-request full-model qualification driven by compiler-emitted native blocks.
use super::recurrent::write_f32;
use super::*;
use pangu_compiler::lower::{Binding, Transform};
use std::collections::BTreeMap;
type Embedding = unsafe extern "C" fn(SessionPtr, u64, u64, u64, i64, i64, *mut u64) -> i32;
type Block = unsafe extern "C" fn(
    SessionPtr,
    i32,
    *const u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    i64,
    i64,
    *mut u64,
    *mut u64,
    *mut u64,
) -> i32;
type Sequence = unsafe extern "C" fn(SessionPtr, *const u64, u64, *mut u64) -> i32;
struct Region {
    handle: u64,
    bytes: usize,
}
fn write_bytes(s: &Session<'_>, id: u64, data: &[u8]) -> Result<()> {
    s.api
        .check(unsafe { (s.api.write)(s.ptr, id, 0, data.as_ptr().cast(), data.len() as u64) })
}
fn read_bytes(s: &Session<'_>, id: u64, count: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; count];
    s.api
        .check(unsafe { (s.api.read)(s.ptr, id, 0, v.as_mut_ptr().cast(), count as u64) })?;
    Ok(v)
}
fn upload(
    s: &Session<'_>,
    dir: &Path,
    c: &Checkpoint,
    binding: &Binding,
    cache: &mut BTreeMap<(String, bool), u64>,
    hashes: &mut BTreeMap<String, String>,
) -> Result<u64> {
    let transformed = binding.transform == Transform::OnePlusFp32;
    let key = (binding.name.clone(), transformed);
    if let Some(&id) = cache.get(&key) {
        return Ok(id);
    }
    let w = c
        .weights
        .get(&binding.name)
        .ok_or_else(|| invalid("native binding absent"))?;
    let root = dir.canonicalize()?;
    let path = root.join(&w.shard).canonicalize()?;
    if path.parent() != Some(root.as_path()) {
        return Err(invalid("native shard escaped checkpoint"));
    }
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(w.offset))?;
    let mut bytes = vec![0u8; w.bytes as usize];
    file.read_exact(&mut bytes)?;
    hashes.insert(binding.name.clone(), sha(&bytes));
    if transformed {
        if w.dtype != Dtype::BF16 || w.shape.len() != 1 {
            return Err(invalid("invalid zero-centered gamma transform"));
        }
        bytes = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|v| (1. + float(u16::from_le_bytes(*v))).to_le_bytes())
            .collect();
    }
    let id = s.allocate(bytes.len())?;
    write_bytes(s, id, &bytes)?;
    cache.insert(key, id);
    Ok(id)
}
pub fn run_model(
    dsl: &Path,
    dir: &Path,
    library: &Path,
    device: i32,
    tokens: &[u32],
) -> Result<serde_json::Value> {
    let spec = pangu_dsl::parse_checkpoint(&std::fs::read_to_string(dsl)?)?;
    let checkpoint = pangu_compiler::checkpoint::inspect(dir)?;
    let plan = pangu_compiler::bound::compile(&spec, checkpoint)?;
    let context = spec.context_buckets[0];
    let program = pangu_compiler::lower::lower(&plan, context)?;
    if tokens.is_empty()
        || tokens.len() > context
        || tokens.iter().any(|&t| t as usize >= spec.model.vocab)
    {
        return Err(invalid("model probe token IDs exceed context/vocabulary"));
    }
    let mut cpu = super::cpu_model::CpuModel::new(dir, &plan.checkpoint)?;
    let api = Api::load(library)?;
    let s = Session::open(&api, device)?;
    unsafe {
        let embed: Embedding = *api
            ._library
            .get(b"pangu_acl_embedding_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let block: Block = *api
            ._library
            .get(b"pangu_acl_model_layer_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let sequence: Sequence = *api
            ._library
            .get(b"pangu_acl_sequence_prepare\0")
            .map_err(|e| invalid(e.to_string()))?;
        let mut cache = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        let c = &plan.checkpoint;
        let embedding = upload(
            &s,
            dir,
            c,
            &Binding {
                name: program.embedding.clone(),
                transform: Transform::Identity,
            },
            &mut cache,
            &mut hashes,
        )?;
        let ids = s.allocate(8)?;
        let positions = s.allocate(8)?;
        let append = s.allocate(8)?;
        let table = s.allocate(context * 8)?;
        let mask = s.allocate(context * 4)?;
        let mut hidden = s.allocate(2048 * 2)?;
        let mut ops = vec![];
        let mut operation = 0;
        api.check(embed(
            s.ptr,
            embedding,
            ids,
            hidden,
            248320,
            2048,
            &mut operation,
        ))?;
        ops.push(operation);
        let slots = context.div_ceil(program.page_tokens) * program.page_tokens;
        let mut states = vec![];
        for item in &program.blocks {
            let mut bindings = vec![];
            for b in &item.bindings {
                bindings.push(upload(&s, dir, c, b, &mut cache, &mut hashes)?);
            }
            let output = s.allocate(2048 * 2)?;
            let mut a = 0;
            let mut b = 0;
            let mut op = 0;
            let is_full = item.kind == pangu_model::Layer::Full;
            api.check(block(
                s.ptr,
                i32::from(is_full),
                bindings.as_ptr(),
                bindings.len() as u64,
                hidden,
                output,
                positions,
                append,
                table,
                mask,
                slots as i64,
                context as i64,
                &mut a,
                &mut b,
                &mut op,
            ))?;
            states.push(Region {
                handle: a,
                bytes: if is_full {
                    slots * 512 * 2
                } else {
                    6144 * 3 * 2
                },
            });
            states.push(Region {
                handle: b,
                bytes: if is_full {
                    slots * 512 * 2
                } else {
                    16 * 128 * 128 * 4
                },
            });
            ops.push(op);
            hidden = output;
            eprintln!("prepared model layer {} {:?}", item.layer, item.kind);
        }
        let gamma = upload(&s, dir, c, &program.final_norm, &mut cache, &mut hashes)?;
        let normalized = s.allocate(2048 * 2)?;
        let logits = s.allocate(248320 * 2)?;
        let mut norm = 0;
        api.check((api.norm_prepare)(
            s.ptr, hidden, gamma, normalized, 1, 2048, &mut norm,
        ))?;
        ops.push(norm);
        let mut head = 0;
        api.check((api.prepare)(
            s.ptr, normalized, embedding, logits, 1, 248320, 2048, &mut head,
        ))?;
        ops.push(head);
        let mut root = 0;
        api.check(sequence(s.ptr, ops.as_ptr(), ops.len() as u64, &mut root))?;
        if hashes.len() != plan.checkpoint.weights.len() {
            return Err(invalid("full-model weight binding coverage mismatch"));
        }
        for state in &states {
            write_bytes(&s, state.handle, &vec![0; state.bytes])?;
        }
        write_bytes(&s, ids, &(tokens[0] as i64).to_le_bytes())?;
        write_bytes(&s, positions, &0i64.to_le_bytes())?;
        write_bytes(&s, append, &0i64.to_le_bytes())?;
        let table_values: Vec<u8> = (0..context)
            .flat_map(|i| (i as i64).to_le_bytes())
            .collect();
        write_bytes(&s, table, &table_values)?;
        let mut mask_values = vec![f32::NEG_INFINITY; context];
        mask_values[0] = 0.;
        write_f32(&s, mask, &mask_values)?;
        eprintln!("warming and capturing complete 24-layer graph");
        let mut graph = 0;
        api.check((api.capture)(s.ptr, root, &mut graph))?;
        for state in &states {
            write_bytes(&s, state.handle, &vec![0; state.bytes])?;
        }
        let mut results = vec![];
        for (position, &token) in tokens.iter().enumerate() {
            write_bytes(&s, ids, &(token as i64).to_le_bytes())?;
            write_bytes(&s, positions, &(position as i64).to_le_bytes())?;
            write_bytes(&s, append, &(position as i64).to_le_bytes())?;
            mask_values[position] = 0.;
            write_f32(&s, mask, &mask_values)?;
            let before = states
                .iter()
                .map(|r| read_bytes(&s, r.handle, r.bytes))
                .collect::<Result<Vec<_>>>()?;
            api.check((api.execute)(s.ptr, root))?;
            let eager = s.read(logits, 248320)?;
            let after = states
                .iter()
                .map(|r| read_bytes(&s, r.handle, r.bytes))
                .collect::<Result<Vec<_>>>()?;
            for (state, data) in states.iter().zip(&before) {
                write_bytes(&s, state.handle, data)?;
            }
            s.write(logits, &vec![bf16(-123.); 248320])?;
            api.check((api.replay)(s.ptr, graph))?;
            let output = s.read(logits, 248320)?;
            if output != eager {
                return Err(invalid("full-model eager/graph logits mismatch"));
            }
            for (state, data) in states.iter().zip(&after) {
                if read_bytes(&s, state.handle, state.bytes)? != *data {
                    return Err(invalid("full-model eager/graph state mismatch"));
                }
            }
            if output.iter().any(|&x| !float(x).is_finite()) {
                return Err(invalid("full-model nonfinite logits"));
            }
            let mut sorted: Vec<_> = output
                .iter()
                .enumerate()
                .map(|(i, &x)| (i, float(x)))
                .collect();
            sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
            let expected = cpu.step(token, position)?;
            let mut difference = 0f64;
            let mut magnitude = 0f64;
            let mut max_error = 0f32;
            for (&a, &e) in output.iter().zip(&expected) {
                let a = float(a);
                let e = float(e);
                if !e.is_finite() {
                    return Err(invalid("CPU reference nonfinite logits"));
                }
                difference += ((a - e) as f64).powi(2);
                magnitude += (e as f64).powi(2);
                max_error = max_error.max((a - e).abs());
            }
            let relative_l2 = (difference / magnitude.max(1e-20)).sqrt();
            let reference_top = expected
                .iter()
                .enumerate()
                .max_by(|a, b| float(*a.1).total_cmp(&float(*b.1)))
                .unwrap()
                .0;
            if relative_l2 > 0.05 || reference_top != sorted[0].0 {
                return Err(invalid(format!(
                    "full-model CPU/NPU mismatch: relative_l2={relative_l2} reference_top={reference_top} native_top={}",
                    sorted[0].0
                )));
            }
            results.push(serde_json::json!({"position":position,"input_token":token,"cpu_reference_relative_l2":relative_l2,"cpu_reference_max_error":max_error,"cpu_reference_top1":reference_top,"top5":&sorted[..5],"graph_eager_logits":"bitwise_equal","graph_eager_states":"bitwise_equal"}));
            eprintln!(
                "full model position={position} top_token={} logit={}",
                sorted[0].0, sorted[0].1
            );
        }
        s.close()?;
        Ok(
            serde_json::json!({"device":device,"program":program,"bound_weights":hashes.len(),"weight_sha256":hashes,"results":results,"native_sha256":sha(&std::fs::read(library)?),"rust_sha256":sha(&std::fs::read(std::env::current_exe()?)?),"scope":"full-model scalar Rust CPU/eager/graph qualification on supplied token IDs; no serving readiness"}),
        )
    }
}
