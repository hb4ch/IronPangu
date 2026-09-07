//! Startup compiler and checksummed binary container. No vendor toolchain invocation here.
use pangu_ir::*;
use pangu_model::{Layer, Result, Role, Spec, invalid};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 8] = b"PANGU\0\0\x01";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
pub fn key(spec: &Spec, target: &Target) -> Result<String> {
    let bytes = serde_json::to_vec(&(FORMAT_VERSION, "compiler-v1-templates-v1", spec, target))
        .map_err(|e| invalid(e.to_string()))?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    digest.update(include_bytes!("lib.rs"));
    digest.update(include_bytes!("../../pangu-ir/src/lib.rs"));
    digest.update(include_bytes!("../../pangu-model/src/lib.rs"));
    Ok(digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
pub fn compile(spec: &Spec, target: &Target) -> Result<Artifact> {
    spec.validate()?;
    let mut ranks = Vec::new();
    for (role, mesh) in [(Role::Prefill, &spec.prefill), (Role::Decode, &spec.decode)] {
        for cp in 0..mesh.cp {
            for tp in 0..mesh.tp {
                let rank = Rank { role, tp, cp };
                let mut states = Vec::new();
                let mut instructions = vec![Instruction::Kernel {
                    layer: None,
                    name: if mesh.sp {
                        "embedding_sequence_shard"
                    } else {
                        "embedding"
                    }
                    .into(),
                }];
                let mut seq = 0;
                for (i, layer) in spec.model.layers.iter().enumerate() {
                    let m = &spec.model;
                    instructions.push(Instruction::Kernel {
                        layer: Some(i),
                        name: "input_norm".into(),
                    });
                    if mesh.sp {
                        instructions.push(Instruction::Communication {
                            group: format!("{role:?}/tp/cp{cp}"),
                            sequence: seq,
                            op: CommOp::AllGather,
                            tensor: "normalized_token_shards".into(),
                        });
                        seq += 1;
                    }
                    match layer {
                        Layer::Full => {
                            states.push(StateLayout {
                                layer: i,
                                kind: StateKind::Kv,
                                shape: vec![2, spec.page_tokens, m.kv_heads / mesh.tp, m.head_dim],
                                dtype: "bf16".into(),
                            });
                            for name in ["qkv_projection", "qk_norm_rope", "paged_attention_local"]
                            {
                                instructions.push(Instruction::Kernel {
                                    layer: Some(i),
                                    name: name.into(),
                                });
                            }
                            if mesh.cp > 1 {
                                for step in 1..mesh.cp {
                                    let group = format!("{role:?}/cp/tp{tp}");
                                    instructions.push(Instruction::Communication {
                                        group: group.clone(),
                                        sequence: seq,
                                        op: CommOp::Send {
                                            peer: (cp + 1) % mesh.cp,
                                        },
                                        tensor: "kv_context_shard".into(),
                                    });
                                    instructions.push(Instruction::Communication {
                                        group,
                                        sequence: seq,
                                        op: CommOp::Receive {
                                            peer: (cp + mesh.cp - 1) % mesh.cp,
                                        },
                                        tensor: "kv_context_shard".into(),
                                    });
                                    instructions.push(Instruction::Kernel {
                                        layer: Some(i),
                                        name: format!(
                                            "causal_attention_online_softmax_merge_{step}"
                                        ),
                                    });
                                    seq += 1;
                                }
                            }
                            instructions.push(Instruction::Kernel {
                                layer: Some(i),
                                name: "attention_output_gate".into(),
                            });
                        }
                        Layer::Delta => {
                            states.push(StateLayout {
                                layer: i,
                                kind: StateKind::Recurrent,
                                shape: vec![m.delta_heads / mesh.tp, m.delta_dim, m.delta_dim],
                                dtype: "fp32".into(),
                            });
                            states.push(StateLayout {
                                layer: i,
                                kind: StateKind::Conv,
                                shape: vec![
                                    3 * m.delta_heads / mesh.tp * m.delta_dim,
                                    m.conv_width,
                                ],
                                dtype: "bf16".into(),
                            });
                            let group = format!("{role:?}/cp/tp{tp}");
                            if cp > 0 {
                                instructions.push(Instruction::Communication {
                                    group: group.clone(),
                                    sequence: seq + cp - 1,
                                    op: CommOp::Receive { peer: cp - 1 },
                                    tensor: format!("hybrid_boundary_layer{i}"),
                                });
                            }
                            for name in [
                                "delta_projection",
                                "causal_conv",
                                "qk_l2_norm_decay_beta",
                                "gated_delta",
                                "gated_norm",
                            ] {
                                instructions.push(Instruction::Kernel {
                                    layer: Some(i),
                                    name: name.into(),
                                });
                            }
                            if cp + 1 < mesh.cp {
                                instructions.push(Instruction::Communication {
                                    group,
                                    sequence: seq + cp,
                                    op: CommOp::Send { peer: cp + 1 },
                                    tensor: format!("hybrid_boundary_layer{i}"),
                                });
                            }
                            seq += mesh.cp - 1;
                        }
                    }
                    instructions.push(Instruction::Kernel {
                        layer: Some(i),
                        name: "output_projection".into(),
                    });
                    for branch in ["attention", "mlp"] {
                        if branch == "mlp" {
                            instructions.push(Instruction::Kernel {
                                layer: Some(i),
                                name: "post_norm".into(),
                            });
                            if mesh.sp {
                                instructions.push(Instruction::Communication {
                                    group: format!("{role:?}/tp/cp{cp}"),
                                    sequence: seq,
                                    op: CommOp::AllGather,
                                    tensor: "normalized_token_shards".into(),
                                });
                                seq += 1;
                            }
                            for name in ["mlp_up_gate_silu", "mlp_down"] {
                                instructions.push(Instruction::Kernel {
                                    layer: Some(i),
                                    name: name.into(),
                                });
                            }
                        }
                        if mesh.tp > 1 {
                            let group = format!("{role:?}/tp/cp{cp}");
                            instructions.push(Instruction::Communication {
                                group: group.clone(),
                                sequence: seq,
                                op: if mesh.sp {
                                    CommOp::ReduceScatter
                                } else {
                                    CommOp::AllReduce
                                },
                                tensor: format!("{branch}_partial_hidden"),
                            });
                            seq += 1;
                            instructions.push(Instruction::Kernel {
                                layer: Some(i),
                                name: "residual_add".into(),
                            });
                        } else {
                            instructions.push(Instruction::Kernel {
                                layer: Some(i),
                                name: "residual_add".into(),
                            });
                        }
                    }
                }
                instructions.push(Instruction::Kernel {
                    layer: None,
                    name: "final_norm".into(),
                });
                if mesh.sp {
                    instructions.push(Instruction::Communication {
                        group: format!("{role:?}/tp/cp{cp}"),
                        sequence: seq,
                        op: CommOp::AllGather,
                        tensor: "final_normalized_token_shards".into(),
                    });
                }
                // Replicated vocabulary head/sampling for the first structural plan.
                for name in ["lm_head", "sample_logits"] {
                    instructions.push(Instruction::Kernel {
                        layer: None,
                        name: name.into(),
                    });
                }
                instructions.push(Instruction::Fence);
                ranks.push(RankPlan {
                    rank,
                    states,
                    instructions,
                });
            }
        }
    }
    Ok(Artifact {
        version: FORMAT_VERSION,
        key: key(spec, target)?,
        target: target.clone(),
        spec: spec.clone(),
        ranks,
    })
}

pub fn encode(artifact: &Artifact) -> Result<Vec<u8>> {
    let payload = serde_json::to_vec(artifact).map_err(|e| invalid(e.to_string()))?;
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&Sha256::digest(&payload));
    out.extend(payload);
    Ok(out)
}
pub fn decode(bytes: &[u8]) -> Result<Artifact> {
    if bytes.len() < 48 || &bytes[..8] != MAGIC {
        return Err(invalid("invalid artifact header/version"));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| invalid("length"))?);
    if len != (bytes.len() - 48) as u64 || Sha256::digest(&bytes[48..])[..] != bytes[16..48] {
        return Err(invalid("artifact length/checksum mismatch"));
    }
    let a: Artifact = serde_json::from_slice(&bytes[48..]).map_err(|e| invalid(e.to_string()))?;
    // Reject a validly checksummed but semantically altered executable plan too.
    if a != compile(&a.spec, &a.target)? {
        return Err(invalid("artifact compatibility/plan mismatch"));
    }
    Ok(a)
}
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!(
        "{}.{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temp);
    }
    result
}
pub fn cached(spec: &Spec, target: &Target, dir: &Path) -> Result<(Artifact, bool)> {
    let path = dir.join(format!("{}.pangu-bin", key(spec, target)?));
    match fs::read(&path) {
        Ok(bytes) => {
            let a = decode(&bytes)?;
            if a.target != *target || a.spec != *spec {
                return Err(invalid("cache target mismatch"));
            }
            Ok((a, true))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let a = compile(spec, target)?;
            write_atomic(&path, &encode(&a)?)?;
            Ok((a, false))
        }
        Err(e) => Err(e.into()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> Spec {
        pangu_dsl::parse(include_str!("../../../examples/qwen35-2b.pangu")).unwrap()
    }
    #[test]
    fn binary_integrity_and_compatibility() {
        let a = compile(&spec(), &Target::mock()).unwrap();
        let mut bytes = encode(&a).unwrap();
        assert_eq!(decode(&bytes).unwrap(), a);
        bytes[50] ^= 1;
        assert!(decode(&bytes).is_err());
        let mut t = Target::mock();
        t.kernel_revision = "changed".into();
        assert_ne!(key(&spec(), &t).unwrap(), a.key);
    }
    #[test]
    fn distributed_layouts_and_communications() {
        let mut s = spec();
        s.prefill.tp = 2;
        s.decode.tp = 2;
        s.prefill.cp = 2;
        s.prefill.sp = true;
        let a = compile(&s, &Target::mock()).unwrap();
        assert_eq!(a.ranks.len(), 6);
        for rank in &a.ranks {
            assert_eq!(rank.states.len(), 42);
            assert_eq!(
                rank.states
                    .iter()
                    .find(|x| x.kind == StateKind::Kv)
                    .unwrap()
                    .shape,
                vec![2, 128, 1, 256]
            );
        }
        let p = &a.ranks[0];
        assert!(p.instructions.iter().any(|i| matches!(
            i,
            Instruction::Communication {
                op: CommOp::ReduceScatter,
                ..
            }
        )));
        assert!(p.instructions.iter().any(|i| matches!(
            i,
            Instruction::Communication {
                op: CommOp::Send { .. },
                ..
            }
        )));
        for cp in 0..2 {
            let left = a
                .ranks
                .iter()
                .find(|r| {
                    r.rank
                        == Rank {
                            role: Role::Prefill,
                            tp: 0,
                            cp,
                        }
                })
                .unwrap();
            let right = a
                .ranks
                .iter()
                .find(|r| {
                    r.rank
                        == Rank {
                            role: Role::Prefill,
                            tp: 1,
                            cp,
                        }
                })
                .unwrap();
            let collect = |r: &RankPlan| {
                r.instructions
                    .iter()
                    .filter(|i| {
                        matches!(
                            i,
                            Instruction::Communication {
                                op: CommOp::AllGather | CommOp::ReduceScatter | CommOp::AllReduce,
                                ..
                            }
                        )
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            };
            assert_eq!(collect(left), collect(right));
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    #[test]
    fn cache_roundtrip_miss_hit_and_invalidation() {
        let dir = std::env::temp_dir().join(format!(
            "pangu-cache-test-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let mut s = pangu_dsl::parse(include_str!("../../../examples/qwen35-2b.pangu")).unwrap();
        let (a, hit) = cached(&s, &Target::mock(), &dir).unwrap();
        assert!(!hit);
        assert!(cached(&s, &Target::mock(), &dir).unwrap().1);
        s.token_budget += 1;
        let (b, hit) = cached(&s, &Target::mock(), &dir).unwrap();
        assert!(!hit);
        assert_ne!(a.key, b.key);
        let path = dir.join(format!("{}.pangu-bin", b.key));
        fs::write(path, b"broken").unwrap();
        assert!(cached(&s, &Target::mock(), &dir).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}

pub mod bound;
pub mod checkpoint;

pub mod lower;
