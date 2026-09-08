//! Independent scalar full-model oracle for bounded numerical qualification.
use super::attention::qk_ref;
use super::layer::{Weight, cpu_layer, norm, point, silu};
use super::*;
use std::collections::BTreeMap;
struct State {
    history: Vec<u16>,
    recurrent: Vec<f32>,
    keys: Vec<u16>,
    values: Vec<u16>,
}
pub(super) struct CpuModel<'a> {
    dir: std::path::PathBuf,
    checkpoint: &'a Checkpoint,
    embedding: Vec<u16>,
    final_norm: Vec<u16>,
    states: Vec<State>,
}
impl<'a> CpuModel<'a> {
    pub(super) fn new(dir: &Path, c: &'a Checkpoint) -> Result<Self> {
        let (embedding, _, _, _) = load_weight(dir, c, &format!("{PREFIX}embed_tokens.weight"))?;
        let (final_norm, _, _, _) = load_weight(dir, c, &format!("{PREFIX}norm.weight"))?;
        let states = (0..24)
            .map(|i| State {
                history: if i % 4 == 3 {
                    vec![]
                } else {
                    vec![0; 6144 * 3]
                },
                recurrent: if i % 4 == 3 {
                    vec![]
                } else {
                    vec![0.; 16 * 128 * 128]
                },
                keys: vec![],
                values: vec![],
            })
            .collect();
        Ok(Self {
            dir: dir.to_path_buf(),
            checkpoint: c,
            embedding,
            final_norm,
            states,
        })
    }
    pub(super) fn step(&mut self, token: u32, position: usize) -> Result<Vec<u16>> {
        let start = token as usize * 2048;
        let mut x = self.embedding[start..start + 2048].to_vec();
        for layer in 0..24 {
            let prefix = format!("{PREFIX}layers.{layer}.");
            let mut w = BTreeMap::new();
            for (key, meta) in self
                .checkpoint
                .weights
                .iter()
                .filter(|(k, _)| k.starts_with(&prefix))
            {
                let mut file = File::open(self.dir.join(&meta.shard))?;
                file.seek(SeekFrom::Start(meta.offset))?;
                let mut bytes = vec![0; meta.bytes as usize];
                file.read_exact(&mut bytes)?;
                let (b, f) = if meta.dtype == Dtype::BF16 {
                    (
                        bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|b| u16::from_le_bytes(*b))
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
                            .map(|b| f32::from_le_bytes(*b))
                            .collect(),
                    )
                };
                w.insert(
                    key[prefix.len()..].to_string(),
                    Weight {
                        handle: 0,
                        b,
                        f,
                        hash: String::new(),
                    },
                );
            }
            let state = &mut self.states[layer];
            if layer % 4 != 3 {
                x = cpu_layer(&x, &w, &mut state.history, &mut state.recurrent, 1);
            } else {
                let input = norm(&x, &w["input_layernorm.weight"].b, 2048);
                let qp = reference(&input, &w["self_attn.q_proj.weight"].b, 1, 4096, 2048);
                let ki = reference(&input, &w["self_attn.k_proj.weight"].b, 1, 512, 2048);
                let vi = reference(&input, &w["self_attn.v_proj.weight"].b, 1, 512, 2048);
                let q = qk_ref(&qp, &w["self_attn.q_norm.weight"].b, 8, 1, true, position);
                let k = qk_ref(&ki, &w["self_attn.k_norm.weight"].b, 2, 1, false, position);
                state.keys.extend(k);
                state.values.extend(vi);
                let mut attended = vec![0; 2048];
                for h in 0..8 {
                    let kh = h / 4;
                    let mut scores = vec![0.; position + 1];
                    for (t, score) in scores.iter_mut().enumerate() {
                        let mut sum = 0f64;
                        for d in 0..256 {
                            sum += float(q[h * 256 + d]) as f64
                                * float(state.keys[t * 512 + kh * 256 + d]) as f64;
                        }
                        *score = float(bf16(float(bf16(sum as f32)) * 0.0625));
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum = scores.iter().map(|x| (x - max).exp()).sum::<f32>();
                    let probs: Vec<_> = scores
                        .iter()
                        .map(|x| float(bf16((x - max).exp() / sum)))
                        .collect();
                    for d in 0..256 {
                        let mut sum = 0f64;
                        for (t, &p) in probs.iter().enumerate() {
                            sum += p as f64 * float(state.values[t * 512 + kh * 256 + d]) as f64;
                        }
                        let gate = float(bf16(1. / (1. + (-float(qp[h * 512 + 256 + d])).exp())));
                        attended[h * 256 + d] = bf16(float(bf16(sum as f32)) * gate);
                    }
                }
                let attn = reference(&attended, &w["self_attn.o_proj.weight"].b, 1, 2048, 2048);
                let residual = point(&x, &attn, false);
                let ff = norm(&residual, &w["post_attention_layernorm.weight"].b, 2048);
                let gate = reference(&ff, &w["mlp.gate_proj.weight"].b, 1, 6144, 2048)
                    .into_iter()
                    .map(silu)
                    .collect::<Vec<_>>();
                let up = reference(&ff, &w["mlp.up_proj.weight"].b, 1, 6144, 2048);
                let product = point(&gate, &up, true);
                let down = reference(&product, &w["mlp.down_proj.weight"].b, 1, 2048, 6144);
                x = point(&residual, &down, false);
            }
            if layer % 4 == 3 {
                eprintln!("CPU reference position={position} completed layer {layer}");
            }
        }
        let normalized = norm(&x, &self.final_norm, 2048);
        Ok(reference(&normalized, &self.embedding, 1, 248320, 2048))
    }
}
