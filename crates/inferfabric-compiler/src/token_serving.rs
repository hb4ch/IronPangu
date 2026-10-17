//! Serving capability/state lowering from typed operators, never model names.
use inferfabric_authoring::execution::{Graph, Type};
use inferfabric_ir::serving::*;
use inferfabric_model::{Result, invalid};

pub fn envelope(graph: &Graph, ranks: usize) -> Result<(Vec<Variant>, Vec<State>)> {
    let types = graph.validate()?;
    let context =
        u32::try_from(graph.max_context).map_err(|_| invalid("serving context overflow"))?;
    let owners = (0..ranks)
        .map(|r| u32::try_from(r).map_err(|_| invalid("rank overflow")))
        .collect::<Result<Vec<_>>>()?;
    let mut states = Vec::new();
    for (id, n) in graph.nodes.iter().enumerate() {
        let storage = match n.op.as_str() {
            "mome" => {
                let Type::Tensor(width) = types[n.inputs[0]] else {
                    return Err(invalid("convolution input"));
                };
                Storage::Slot {
                    bytes: (n.integer("taps")? as u64 - 1) * width as u64 * 2,
                    dtype: "bf16".into(),
                }
            }
            "cache_append" | "cache_norm_rope_append" => {
                let format = n
                    .attrs
                    .get("cache_dtype")
                    .map(String::as_str)
                    .unwrap_or("bf16");
                match format {
                    "bf16" => Storage::Pages {
                        page_tokens: 16,
                        row_bytes: 576 * 2,
                        encoding: "bf16_latent512_rotary64".into(),
                    },
                    "int8_ds_mla" => Storage::Pages {
                        page_tokens: 32,
                        row_bytes: 656,
                        encoding: "int8_latent512_tile128_f32_scales_bf16_rotary64".into(),
                    },
                    "int8" | "int4" => Storage::Slot {
                        bytes: u64::from(context.div_ceil(16) * 16)
                            * (576 * if format == "int8" { 8 } else { 4 } / 8 + 4),
                        dtype: format!("{format}_row576_f32_scale"),
                    },
                    _ => return Err(invalid("unimplemented serving cache layout")),
                }
            }
            _ => continue,
        };
        let window = graph
            .nodes
            .iter()
            .filter(|consumer| {
                consumer.op == "latent_attention" && consumer.inputs.get(1) == Some(&id)
            })
            .map(|consumer| consumer.integer("window"))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max();
        states.push(State {
            id: n.name.clone(),
            logical_node: id,
            ranks: owners.clone(),
            storage,
            capacity_tokens: match n
                .attrs
                .get("cache_dtype")
                .map(String::as_str)
                .unwrap_or("bf16")
            {
                _ if n.op == "mome" => context,
                "bf16" => 512,
                "int8_ds_mla" => context.div_ceil(32) * 32,
                _ => context.div_ceil(16) * 16,
            },
            window_tokens: window
                .map(|n| u32::try_from(n).map_err(|_| invalid("window overflow")))
                .transpose()?,
            checkpoint_supported: false,
        });
    }
    let variants = vec![Variant {
        id: "serial_token_v1".into(),
        phase: Phase::Decode,
        max_sequences: 1,
        max_tokens: 1,
        max_query_tokens: 1,
        max_context: context,
        numerical_class: "ascend_serial_token_v1".into(),
        graph_capable: true,
        packed: false,
    }];
    Ok((variants, states))
}

pub fn lower(
    graph: &Graph,
    physical: &inferfabric_plan::physical::Plan,
    program: &[u8],
    checkpoint: &str,
) -> Result<Manifest> {
    let (variants, states) = envelope(graph, physical.ranks.len())?;
    let vocab = graph
        .nodes
        .iter()
        .find(|n| n.op == "embedding")
        .ok_or_else(|| invalid("serving embedding contract"))?
        .integer("vocab")?;
    let manifest = Manifest { version: VERSION,
        identity: Identity { program_sha256: crate::token_physical::digest(program), checkpoint_metadata_sha256: checkpoint.into(),
            backend_revision: crate::token_physical::digest(&serde_json::to_vec(physical).map_err(|e| invalid(e.to_string()))?),
            devices: physical.ranks.iter().map(|r| r.device).collect() },
        vocab_size: u32::try_from(vocab).map_err(|_| invalid("vocab overflow"))?,
        eos: u32::try_from(graph.eos).map_err(|_| invalid("eos overflow"))?, variants, states, qualification: None,
        limitations: vec!["Single sequence and one input token per invocation; prompt processing is serial, not packed prefill".into(),
            "State is request-private; pooled pages, checkpoint restore and prefix reuse are not implemented".into(),
            "Compiler contract only: startup must verify native binary, frozen resources and hardware qualification before readiness".into()] };
    manifest.validate()?;
    Ok(manifest)
}
