//! Checkpoint-bound, typed mathematical IR. This is pre-link IR, not device code.
use crate::checkpoint::{Checkpoint, Dtype, PREFIX, sha};
use pangu_model::{Layer, Result, Spec, invalid};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dim {
    Tokens,
    Requests,
    RequestBoundaries,
    Pages,
    Fixed(usize),
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Scalar {
    Bf16,
    Fp32,
    I64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<Dim>,
    pub dtype: Scalar,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StateKind {
    PagedKeys,
    PagedValues,
    DeltaKeyValue,
    ConvPreviousInputs,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct State {
    pub tensor: Tensor,
    pub kind: StateKind,
    pub initially_zero: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "op")]
pub enum Op {
    Embedding {
        weight: String,
    },
    /// y = x * W^T, no bias. BF16 inputs/outputs, FP32 accumulation required.
    Linear {
        weight: String,
    },
    /// float(x) * rsqrt(mean(float(x)^2)+eps) * (1+float(w)), cast to BF16.
    ZeroCenteredRms {
        weight: String,
        epsilon: String,
    },
    Reshape,
    Split {
        axis: usize,
        lengths: Vec<usize>,
    },
    /// Rotate first rotary_dim channels via half-rotation; remaining channels unchanged.
    /// Text-only: all three mRoPE position axes equal the absolute text position.
    TextRope {
        rotary_dim: usize,
        theta: u64,
        interleaved_sections: [usize; 3],
    },
    /// Append K/V at absolute positions, causal attention with q_heads/kv_heads grouping.
    /// Packed request boundaries and page table are explicit operands.
    PagedAttention {
        keys: String,
        values: String,
        scale: String,
        causal: bool,
    },
    Sigmoid,
    Silu,
    Multiply,
    Add,
    /// Depthwise causal convolution over packed q/k/v channels, then SiLU.
    /// State contains the preceding kernel_width-1 raw projected inputs (oldest first).
    CausalConvSilu {
        weight: String,
        history: String,
        kernel_width: usize,
    },
    /// Reference fallback: x*x, sum, +epsilon, rsqrt, x*inv_norm each returns BF16.
    /// Reduction may accumulate FP32 but rounds to BF16 before adding epsilon.
    L2Norm {
        epsilon: String,
    },
    /// g = -exp(A_log.float()) * softplus(a.float()+dt_bias.float()), no clamp.
    DeltaDecay {
        a_log: String,
        dt_bias: String,
        clamp: bool,
    },
    /// Sequential recurrence: S*=exp(g); r=beta*(v-k^T*S); S+=k outer r;
    /// y=(q/sqrt(Dk))^T*S. FP32 arithmetic/state; BF16 output.
    /// Per-request state uses [request,head,key,value], never an implicit transpose.
    DeltaRecurrence {
        state: String,
        q_scale: String,
        state_layout: String,
    },
    /// RMS in FP32, cast normalized x to BF16, multiply direct FP32 weight,
    /// multiply SiLU(float(z)), then cast BF16. No (1+w).
    GatedRms {
        weight: String,
        epsilon: String,
        normalize_before_gate: bool,
    },
    /// Select the last consumed token of each nonempty request segment before LM projection.
    SelectFrontier,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Node {
    pub inputs: Vec<String>,
    pub outputs: Vec<Tensor>,
    pub operation: Op,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Plan {
    pub format: String,
    pub executable: bool,
    pub contract: String,
    pub reference_source_sha256: String,
    pub key: String,
    pub spec: Spec,
    pub checkpoint: Checkpoint,
    pub inputs: Vec<Tensor>,
    pub states: Vec<State>,
    pub nodes: Vec<Node>,
    pub output: String,
    pub text_weight_bytes: u64,
    pub recurrent_and_conv_bytes_per_request: u64,
    pub kv_bytes_per_page: u64,
    pub link_requirements: Vec<String>,
}
struct Builder {
    nodes: Vec<Node>,
    states: Vec<State>,
    serial: usize,
}
fn dims(shape: &[usize]) -> Vec<Dim> {
    std::iter::once(Dim::Tokens)
        .chain(shape.iter().copied().map(Dim::Fixed))
        .collect()
}
impl Builder {
    fn node(&mut self, op: Op, inputs: &[&str], shapes: Vec<(Vec<Dim>, Scalar)>) -> Vec<String> {
        let outputs: Vec<_> = shapes
            .into_iter()
            .map(|(shape, dtype)| {
                let name = format!("t{}", self.serial);
                self.serial += 1;
                Tensor { name, shape, dtype }
            })
            .collect();
        let names = outputs.iter().map(|t| t.name.clone()).collect();
        self.nodes.push(Node {
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs,
            operation: op,
        });
        names
    }
    fn one(&mut self, op: Op, inputs: &[&str], shape: &[usize], dtype: Scalar) -> String {
        self.node(op, inputs, vec![(dims(shape), dtype)]).remove(0)
    }
    fn linear(&mut self, x: &str, weight: String, width: usize) -> String {
        self.one(Op::Linear { weight }, &[x], &[width], Scalar::Bf16)
    }
    fn rms(&mut self, x: &str, weight: String, shape: &[usize]) -> String {
        self.one(
            Op::ZeroCenteredRms {
                weight,
                epsilon: "1e-6".into(),
            },
            &[x],
            shape,
            Scalar::Bf16,
        )
    }
    fn state(&mut self, name: String, shape: Vec<Dim>, dtype: Scalar, kind: StateKind) -> String {
        self.states.push(State {
            tensor: Tensor {
                name: name.clone(),
                shape,
                dtype,
            },
            initially_zero: matches!(
                kind,
                StateKind::DeltaKeyValue | StateKind::ConvPreviousInputs
            ),
            kind,
        });
        name
    }
}
pub fn compile(spec: &Spec, checkpoint: Checkpoint) -> Result<Plan> {
    spec.validate()?;
    let m = &spec.model;
    if [
        m.hidden,
        m.intermediate,
        m.vocab,
        m.query_heads,
        m.kv_heads,
        m.head_dim,
        m.delta_heads,
        m.delta_dim,
        m.conv_width,
    ] != [2048, 6144, 248320, 8, 2, 256, 16, 128, 4]
        || m.layers
            != (0..24)
                .map(|i| {
                    if i % 4 == 3 {
                        Layer::Full
                    } else {
                        Layer::Delta
                    }
                })
                .collect::<Vec<_>>()
    {
        return Err(invalid(
            "DSL dimensions/layers disagree with qwen35_text_v1",
        ));
    }
    if [&spec.prefill, &spec.decode]
        .iter()
        .any(|m| m.tp != 1 || m.cp != 1 || m.sp)
    {
        return Err(invalid(
            "checkpoint-bound lowering currently requires TP=CP=1 and SP=false; distributed v1 mock plans are not executable lowering",
        ));
    }
    if spec.max_context() > 262144 {
        return Err(invalid("context exceeds checkpoint position limit"));
    }
    let expected = crate::checkpoint::schema();
    if checkpoint.weights.len() != expected.len()
        || expected.iter().any(|(k, (dtype, shape))| {
            checkpoint
                .weights
                .get(k)
                .is_none_or(|w| &w.dtype != dtype || &w.shape != shape)
        })
    {
        return Err(invalid("checkpoint weight schema does not match contract"));
    }
    let inputs = vec![
        Tensor {
            name: "token_ids".into(),
            shape: vec![Dim::Tokens],
            dtype: Scalar::I64,
        },
        Tensor {
            name: "positions".into(),
            shape: vec![Dim::Tokens],
            dtype: Scalar::I64,
        },
        // Request offsets are CSR boundaries of length Requests+1, represented separately.
        Tensor {
            name: "request_offsets".into(),
            shape: vec![Dim::RequestBoundaries],
            dtype: Scalar::I64,
        },
        Tensor {
            name: "request_slots".into(),
            shape: vec![Dim::Requests],
            dtype: Scalar::I64,
        },
        Tensor {
            name: "page_table".into(),
            shape: vec![
                Dim::Requests,
                Dim::Fixed(spec.max_context().div_ceil(spec.page_tokens)),
            ],
            dtype: Scalar::I64,
        },
    ];
    let mut b = Builder {
        nodes: vec![],
        states: vec![],
        serial: 0,
    };
    let embedding = format!("{PREFIX}embed_tokens.weight");
    let mut x = b.one(
        Op::Embedding {
            weight: embedding.clone(),
        },
        &["token_ids"],
        &[2048],
        Scalar::Bf16,
    );
    for (i, kind) in m.layers.iter().enumerate() {
        let w = |name: &str| format!("{PREFIX}layers.{i}.{name}");
        let norm = b.rms(&x, w("input_layernorm.weight"), &[2048]);
        let projected = match kind {
            Layer::Full => {
                let qg = b.linear(&norm, w("self_attn.q_proj.weight"), 4096);
                let qg = b.one(Op::Reshape, &[&qg], &[8, 512], Scalar::Bf16);
                let split = b.node(
                    Op::Split {
                        axis: 2,
                        lengths: vec![256, 256],
                    },
                    &[&qg],
                    vec![(dims(&[8, 256]), Scalar::Bf16); 2],
                );
                let q = b.rms(&split[0], w("self_attn.q_norm.weight"), &[8, 256]);
                let k = b.linear(&norm, w("self_attn.k_proj.weight"), 512);
                let k = b.one(Op::Reshape, &[&k], &[2, 256], Scalar::Bf16);
                let k = b.rms(&k, w("self_attn.k_norm.weight"), &[2, 256]);
                let v = b.linear(&norm, w("self_attn.v_proj.weight"), 512);
                let v = b.one(Op::Reshape, &[&v], &[2, 256], Scalar::Bf16);
                let rope = || Op::TextRope {
                    rotary_dim: 64,
                    theta: 10000000,
                    interleaved_sections: [11, 11, 10],
                };
                let q = b.one(rope(), &[&q, "positions"], &[8, 256], Scalar::Bf16);
                let k = b.one(rope(), &[&k, "positions"], &[2, 256], Scalar::Bf16);
                let state_shape = vec![
                    Dim::Pages,
                    Dim::Fixed(spec.page_tokens),
                    Dim::Fixed(2),
                    Dim::Fixed(256),
                ];
                let keys = b.state(
                    format!("layer.{i}.keys"),
                    state_shape.clone(),
                    Scalar::Bf16,
                    StateKind::PagedKeys,
                );
                let values = b.state(
                    format!("layer.{i}.values"),
                    state_shape,
                    Scalar::Bf16,
                    StateKind::PagedValues,
                );
                let attn = b.one(
                    Op::PagedAttention {
                        keys,
                        values,
                        scale: "1/sqrt(256)".into(),
                        causal: true,
                    },
                    &[
                        &q,
                        &k,
                        &v,
                        "positions",
                        "request_offsets",
                        "request_slots",
                        "page_table",
                    ],
                    &[8, 256],
                    Scalar::Bf16,
                );
                let gate = b.one(Op::Sigmoid, &[&split[1]], &[8, 256], Scalar::Bf16);
                let gated = b.one(Op::Multiply, &[&attn, &gate], &[8, 256], Scalar::Bf16);
                let flat = b.one(Op::Reshape, &[&gated], &[2048], Scalar::Bf16);
                b.linear(&flat, w("self_attn.o_proj.weight"), 2048)
            }
            Layer::Delta => {
                let qkv = b.linear(&norm, w("linear_attn.in_proj_qkv.weight"), 6144);
                let z = b.linear(&norm, w("linear_attn.in_proj_z.weight"), 2048);
                let z = b.one(Op::Reshape, &[&z], &[16, 128], Scalar::Bf16);
                let a = b.linear(&norm, w("linear_attn.in_proj_a.weight"), 16);
                let beta = b.linear(&norm, w("linear_attn.in_proj_b.weight"), 16);
                let beta = b.one(Op::Sigmoid, &[&beta], &[16], Scalar::Bf16);
                let decay = b.one(
                    Op::DeltaDecay {
                        a_log: w("linear_attn.A_log"),
                        dt_bias: w("linear_attn.dt_bias"),
                        clamp: false,
                    },
                    &[&a],
                    &[16],
                    Scalar::Fp32,
                );
                let history = b.state(
                    format!("layer.{i}.conv"),
                    vec![Dim::Requests, Dim::Fixed(6144), Dim::Fixed(3)],
                    Scalar::Bf16,
                    StateKind::ConvPreviousInputs,
                );
                let conv = b.one(
                    Op::CausalConvSilu {
                        weight: w("linear_attn.conv1d.weight"),
                        history,
                        kernel_width: 4,
                    },
                    &[&qkv, "request_offsets", "request_slots"],
                    &[6144],
                    Scalar::Bf16,
                );
                let qkv = b.node(
                    Op::Split {
                        axis: 1,
                        lengths: vec![2048, 2048, 2048],
                    },
                    &[&conv],
                    vec![(dims(&[2048]), Scalar::Bf16); 3],
                );
                let mut split = vec![];
                for t in &qkv {
                    split.push(b.one(Op::Reshape, &[t], &[16, 128], Scalar::Bf16));
                }
                let q = b.one(
                    Op::L2Norm {
                        epsilon: "1e-6".into(),
                    },
                    &[&split[0]],
                    &[16, 128],
                    Scalar::Bf16,
                );
                let k = b.one(
                    Op::L2Norm {
                        epsilon: "1e-6".into(),
                    },
                    &[&split[1]],
                    &[16, 128],
                    Scalar::Bf16,
                );
                let state = b.state(
                    format!("layer.{i}.delta"),
                    vec![
                        Dim::Requests,
                        Dim::Fixed(16),
                        Dim::Fixed(128),
                        Dim::Fixed(128),
                    ],
                    Scalar::Fp32,
                    StateKind::DeltaKeyValue,
                );
                let delta = b.one(
                    Op::DeltaRecurrence {
                        state,
                        q_scale: "1/sqrt(128)".into(),
                        state_layout: "request,head,key,value".into(),
                    },
                    &[
                        &q,
                        &k,
                        &split[2],
                        &decay,
                        &beta,
                        "request_offsets",
                        "request_slots",
                    ],
                    &[16, 128],
                    Scalar::Bf16,
                );
                let gated = b.one(
                    Op::GatedRms {
                        weight: w("linear_attn.norm.weight"),
                        epsilon: "1e-6".into(),
                        normalize_before_gate: true,
                    },
                    &[&delta, &z],
                    &[16, 128],
                    Scalar::Bf16,
                );
                let flat = b.one(Op::Reshape, &[&gated], &[2048], Scalar::Bf16);
                b.linear(&flat, w("linear_attn.out_proj.weight"), 2048)
            }
        };
        x = b.one(Op::Add, &[&x, &projected], &[2048], Scalar::Bf16);
        let norm = b.rms(&x, w("post_attention_layernorm.weight"), &[2048]);
        let gate = b.linear(&norm, w("mlp.gate_proj.weight"), 6144);
        let up = b.linear(&norm, w("mlp.up_proj.weight"), 6144);
        let gate = b.one(Op::Silu, &[&gate], &[6144], Scalar::Bf16);
        let gated = b.one(Op::Multiply, &[&gate, &up], &[6144], Scalar::Bf16);
        let down = b.linear(&gated, w("mlp.down_proj.weight"), 2048);
        x = b.one(Op::Add, &[&x, &down], &[2048], Scalar::Bf16);
    }
    x = b.rms(&x, format!("{PREFIX}norm.weight"), &[2048]);
    let frontier = b
        .node(
            Op::SelectFrontier,
            &[&x, "request_offsets"],
            vec![(vec![Dim::Requests, Dim::Fixed(2048)], Scalar::Bf16)],
        )
        .remove(0);
    let output = b
        .node(
            Op::Linear { weight: embedding },
            &[&frontier],
            vec![(vec![Dim::Requests, Dim::Fixed(248320)], Scalar::Bf16)],
        )
        .remove(0);
    let text_weight_bytes = checkpoint.weights.values().map(|w| w.bytes).sum();
    let mut plan=Plan {
        format:"ironpangu.typed-plan.v1".into(),executable:false,contract:"qwen35_text_v1".into(),
        reference_source_sha256:"788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263".into(),
        key:String::new(),spec:spec.clone(),checkpoint,
        inputs,states:b.states,nodes:b.nodes,output,text_weight_bytes,
        recurrent_and_conv_bytes_per_request:18*(16*128*128*4+6144*3*2),
        kv_bytes_per_page:6*2*spec.page_tokens as u64*2*256*2,
        link_requirements:vec![
            "C++ ACLNN/Ascend C kernel selection and ABI qualification; no native kernels linked".into(),
            "Explicit key,value -> value,key state transpose if a CANN GDR kernel requires it".into(),
            "Unbounded-negative decay support: never clamp g to meet an operator domain".into(),
            "Packed-request masking, state initialization, cache append and page-table validation".into(),
            "Target/driver/CANN fingerprint, weight-payload digest, arenas, workspaces and graph buckets".into(),
            "Chunk/step numerical equivalence, native PD transfer, graph replay qualification".into(),
        ]
    };
    validate_graph(&plan)?;
    let mut identity = serde_json::to_vec(&plan).map_err(|e| invalid(e.to_string()))?;
    identity.extend_from_slice(include_bytes!("bound.rs"));
    identity.extend_from_slice(include_bytes!("checkpoint.rs"));
    plan.key = sha(&identity);
    Ok(plan)
}
/// Check SSA use-before-definition, duplicate definitions and complete weight binding.
pub fn validate_graph(plan: &Plan) -> Result<()> {
    let mut tensors = BTreeMap::new();
    for t in &plan.inputs {
        if tensors.insert(t.name.clone(), t).is_some() {
            return Err(invalid("duplicate input"));
        }
    }
    let mut used = std::collections::BTreeSet::new();
    for n in &plan.nodes {
        if n.inputs.iter().any(|s| !tensors.contains_key(s)) {
            return Err(invalid("IR use before definition"));
        }
        let weights = match &n.operation {
            Op::Embedding { weight }
            | Op::Linear { weight }
            | Op::ZeroCenteredRms { weight, .. }
            | Op::CausalConvSilu { weight, .. }
            | Op::GatedRms { weight, .. } => vec![weight],
            Op::DeltaDecay { a_log, dt_bias, .. } => vec![a_log, dt_bias],
            _ => vec![],
        };
        for w in weights {
            if !plan.checkpoint.weights.contains_key(w) {
                return Err(invalid("IR missing weight"));
            }
            used.insert(w);
        }
        validate_node(plan, n, &tensors)?;
        for t in &n.outputs {
            if tensors.insert(t.name.clone(), t).is_some() {
                return Err(invalid("duplicate SSA output"));
            }
        }
    }
    if used.len() != plan.checkpoint.weights.len() || !tensors.contains_key(&plan.output) {
        return Err(invalid("IR incomplete weight binding/output"));
    }
    // Mixed-precision checkpoint exception must survive lowering.
    if plan
        .checkpoint
        .weights
        .values()
        .filter(|w| w.dtype == Dtype::F32)
        .count()
        != 36
    {
        return Err(invalid("unexpected FP32 weights"));
    }
    Ok(())
}

fn validate_node(plan: &Plan, n: &Node, tensors: &BTreeMap<String, &Tensor>) -> Result<()> {
    let args: Vec<_> = n.inputs.iter().map(|k| tensors[k]).collect();
    let first = args
        .first()
        .ok_or_else(|| invalid("IR operation lacks input"))?;
    let mut expected = vec![(first.shape.clone(), first.dtype.clone())];
    let weight = |key: &str| {
        plan.checkpoint
            .weights
            .get(key)
            .ok_or_else(|| invalid("IR weight missing"))
    };
    let arity = match &n.operation {
        Op::Embedding { weight: key } => {
            if first.dtype != Scalar::I64 {
                return Err(invalid("embedding requires integer IDs"));
            }
            expected[0].0.push(Dim::Fixed(weight(key)?.shape[1]));
            expected[0].1 = Scalar::Bf16;
            1
        }
        Op::Linear { weight: key } => {
            let w = weight(key)?;
            if w.shape.len() != 2
                || first.shape.last() != Some(&Dim::Fixed(w.shape[1]))
                || first.dtype != Scalar::Bf16
            {
                return Err(invalid("linear shape/dtype mismatch"));
            }
            *expected[0]
                .0
                .last_mut()
                .ok_or_else(|| invalid("scalar linear"))? = Dim::Fixed(w.shape[0]);
            1
        }
        Op::ZeroCenteredRms {
            weight: key,
            epsilon,
        }
        | Op::GatedRms {
            weight: key,
            epsilon,
            ..
        } => {
            let w = weight(key)?;
            if w.shape.len() != 1
                || first.shape.last() != Some(&Dim::Fixed(w.shape[0]))
                || first.dtype != Scalar::Bf16
                || epsilon != "1e-6"
            {
                return Err(invalid("norm contract mismatch"));
            }
            if matches!(n.operation, Op::GatedRms { .. }) {
                2
            } else {
                1
            }
        }
        Op::Reshape => {
            fn elements(shape: &[Dim]) -> Option<usize> {
                shape.iter().try_fold(1usize, |a, d| match d {
                    Dim::Fixed(n) => a.checked_mul(*n),
                    _ => None,
                })
            }
            let out = n
                .outputs
                .first()
                .ok_or_else(|| invalid("reshape output missing"))?;
            if first.shape.first() != out.shape.first()
                || elements(&first.shape[1..]).is_none()
                || elements(&first.shape[1..]) != elements(&out.shape[1..])
            {
                return Err(invalid("reshape changes elements"));
            }
            expected[0].0 = out.shape.clone();
            1
        }
        Op::Split { axis, lengths } => {
            if first.shape.get(*axis) != Some(&Dim::Fixed(lengths.iter().sum()))
                || lengths.contains(&0)
            {
                return Err(invalid("invalid split dimension"));
            }
            expected = lengths
                .iter()
                .map(|len| {
                    let mut shape = first.shape.clone();
                    shape[*axis] = Dim::Fixed(*len);
                    (shape, first.dtype.clone())
                })
                .collect();
            1
        }
        Op::TextRope {
            rotary_dim,
            theta,
            interleaved_sections,
        } => {
            if first.shape.last() != Some(&Dim::Fixed(256))
                || *rotary_dim != 64
                || *theta != 10000000
                || *interleaved_sections != [11, 11, 10]
            {
                return Err(invalid("RoPE contract mismatch"));
            }
            2
        }
        Op::PagedAttention {
            keys,
            values,
            scale,
            causal,
        } => {
            if first.shape != dims(&[8, 256]) || scale != "1/sqrt(256)" || !causal {
                return Err(invalid("attention contract mismatch"));
            }
            for (key, kind) in [
                (keys, StateKind::PagedKeys),
                (values, StateKind::PagedValues),
            ] {
                let state = plan
                    .states
                    .iter()
                    .find(|s| &s.tensor.name == key)
                    .ok_or_else(|| invalid("missing KV state"))?;
                if state.kind != kind
                    || state.tensor.dtype != Scalar::Bf16
                    || state.tensor.shape
                        != vec![
                            Dim::Pages,
                            Dim::Fixed(plan.spec.page_tokens),
                            Dim::Fixed(2),
                            Dim::Fixed(256),
                        ]
                {
                    return Err(invalid("invalid KV state"));
                }
            }
            7
        }
        Op::CausalConvSilu {
            weight: key,
            history,
            kernel_width,
        } => {
            let state = plan
                .states
                .iter()
                .find(|s| &s.tensor.name == history)
                .ok_or_else(|| invalid("missing convolution state"))?;
            if *kernel_width != 4
                || weight(key)?.shape != [6144, 1, 4]
                || first.shape != dims(&[6144])
                || state.tensor.shape != vec![Dim::Requests, Dim::Fixed(6144), Dim::Fixed(3)]
                || state.tensor.dtype != Scalar::Bf16
                || state.kind != StateKind::ConvPreviousInputs
            {
                return Err(invalid("convolution contract mismatch"));
            }
            3
        }
        Op::DeltaDecay {
            a_log,
            dt_bias,
            clamp,
        } => {
            if *clamp
                || weight(a_log)?.dtype != Dtype::F32
                || weight(dt_bias)?.shape != [16]
                || first.shape != dims(&[16])
            {
                return Err(invalid("decay contract mismatch"));
            }
            expected[0].1 = Scalar::Fp32;
            1
        }
        Op::DeltaRecurrence {
            state,
            q_scale,
            state_layout,
        } => {
            let s = plan
                .states
                .iter()
                .find(|s| &s.tensor.name == state)
                .ok_or_else(|| invalid("missing recurrent state"))?;
            if s.kind != StateKind::DeltaKeyValue
                || s.tensor.shape
                    != vec![
                        Dim::Requests,
                        Dim::Fixed(16),
                        Dim::Fixed(128),
                        Dim::Fixed(128),
                    ]
                || s.tensor.dtype != Scalar::Fp32
                || q_scale != "1/sqrt(128)"
                || state_layout != "request,head,key,value"
            {
                return Err(invalid("recurrent state contract mismatch"));
            }
            expected[0] = (dims(&[16, 128]), Scalar::Bf16);
            7
        }
        Op::L2Norm { epsilon } => {
            if epsilon != "1e-6" || first.dtype != Scalar::Bf16 {
                return Err(invalid("L2 contract mismatch"));
            }
            1
        }
        Op::Add | Op::Multiply => 2,
        Op::Sigmoid | Op::Silu => 1,
        Op::SelectFrontier => {
            expected[0].0[0] = Dim::Requests;
            2
        }
    };
    if args.len() != arity {
        return Err(invalid("IR operand count mismatch"));
    }
    if matches!(n.operation, Op::Add | Op::Multiply | Op::GatedRms { .. })
        && (args[0].shape != args[1].shape || args[0].dtype != args[1].dtype)
    {
        return Err(invalid("elementwise operand mismatch"));
    }
    if let Op::GatedRms {
        normalize_before_gate,
        ..
    } = &n.operation
        && !normalize_before_gate
    {
        return Err(invalid("gate must follow normalization"));
    }
    if matches!(n.operation, Op::PagedAttention { .. })
        && (args[1].shape != dims(&[2, 256]) || args[2].shape != dims(&[2, 256]))
    {
        return Err(invalid("KV head mismatch"));
    }
    if matches!(n.operation, Op::DeltaRecurrence { .. }) {
        for (i, shape, dtype) in [
            (0, dims(&[16, 128]), Scalar::Bf16),
            (1, dims(&[16, 128]), Scalar::Bf16),
            (2, dims(&[16, 128]), Scalar::Bf16),
            (3, dims(&[16]), Scalar::Fp32),
            (4, dims(&[16]), Scalar::Bf16),
        ] {
            if args[i].shape != shape || args[i].dtype != dtype {
                return Err(invalid("delta operand mismatch"));
            }
        }
    }
    let actual: Vec<_> = n
        .outputs
        .iter()
        .map(|t| (t.shape.clone(), t.dtype.clone()))
        .collect();
    if actual != expected {
        return Err(invalid(format!(
            "IR output type mismatch: {:?}",
            n.operation
        )));
    }
    Ok(())
}

#[cfg(test)]
mod bound_tests {
    use super::*;
    fn checkpoint() -> Checkpoint {
        let weights = crate::checkpoint::schema()
            .into_iter()
            .map(|(k, (dtype, shape))| {
                let bytes = shape.iter().product::<usize>() as u64 * dtype.bytes();
                (
                    k,
                    crate::checkpoint::Weight {
                        shard: "fixture".into(),
                        dtype,
                        shape,
                        offset: 0,
                        bytes,
                    },
                )
            })
            .collect();
        Checkpoint {
            config_sha256: "fixture".into(),
            index_sha256: "fixture".into(),
            shards: BTreeMap::new(),
            weights,
            excluded: BTreeMap::new(),
            metadata_sha256: "fixture".into(),
        }
    }
    fn spec() -> Spec {
        pangu_dsl::parse_checkpoint(include_str!("../../../examples/qwen35-2b-checkpoint.pangu"))
            .unwrap()
    }
    #[test]
    fn binds_all_weights_and_rejects_incompatible_requests() {
        let p = compile(&spec(), checkpoint()).unwrap();
        assert_eq!(p.states.len(), 48);
        assert_eq!(p.checkpoint.weights.len(), 320);
        assert_eq!(p.key, compile(&spec(), checkpoint()).unwrap().key);
        let mut s = spec();
        s.prefill.tp = 2;
        s.decode.tp = 2;
        assert!(compile(&s, checkpoint()).is_err());
        let mut s = spec();
        s.model.layers[0] = Layer::Full;
        assert!(compile(&s, checkpoint()).is_err());
        let mut c = checkpoint();
        c.weights.remove(&format!("{PREFIX}norm.weight"));
        assert!(compile(&spec(), c).is_err());
        let mut c = checkpoint();
        c.weights
            .get_mut(&format!("{PREFIX}layers.0.linear_attn.A_log"))
            .unwrap()
            .dtype = Dtype::BF16;
        assert!(compile(&spec(), c).is_err());
    }
    #[test]
    fn rejects_gate_packing_type_errors_and_state_orientation() {
        let p = compile(&spec(), checkpoint()).unwrap();
        let mut bad = p.clone();
        let split = bad
            .nodes
            .iter_mut()
            .find(|n| matches!(n.operation, Op::Split { axis: 2, .. }))
            .unwrap();
        // Global Q/gate halves would silently interleave the wrong rows.
        split.operation = Op::Split {
            axis: 1,
            lengths: vec![4, 4],
        };
        assert!(validate_graph(&bad).is_err());
        let mut bad = p.clone();
        bad.nodes[1].outputs[0].dtype = Scalar::Fp32;
        assert!(validate_graph(&bad).is_err());
        let mut bad = p.clone();
        if let Op::DeltaRecurrence { state_layout, .. } = &mut bad
            .nodes
            .iter_mut()
            .find(|n| matches!(n.operation, Op::DeltaRecurrence { .. }))
            .unwrap()
            .operation
        {
            *state_layout = "request,head,value,key".into();
        }
        assert!(validate_graph(&bad).is_err());
        let mut bad = p;
        bad.nodes[0].inputs[0] = "undefined".into();
        assert!(validate_graph(&bad).is_err());
    }
    #[test]
    fn native_lowering_covers_weights_and_rejects_edited_plans() {
        let p = compile(&spec(), checkpoint()).unwrap();
        let native = crate::lower::lower(&p, 128).unwrap();
        assert_eq!(native.blocks.len(), 24);
        let mut names = std::collections::BTreeSet::from([
            native.embedding.clone(),
            native.final_norm.name.clone(),
        ]);
        for block in &native.blocks {
            assert_eq!(
                block.bindings.len(),
                if block.kind == Layer::Delta { 14 } else { 11 }
            );
            names.extend(block.bindings.iter().map(|b| b.name.clone()));
        }
        assert_eq!(names, p.checkpoint.weights.keys().cloned().collect());
        assert!(crate::lower::lower(&p, 127).is_err());
        assert_ne!(native.key, crate::lower::lower(&p, 512).unwrap().key);
        let mut edited = p;
        edited.nodes.swap(0, 1);
        assert!(crate::lower::lower(&edited, 128).is_err());
    }
}
