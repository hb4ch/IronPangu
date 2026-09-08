# Checkpoint-bound DSL and compilation

**Current update:** full-model native inference and Rust HTTP serving are now qualified within a single-request, context-128 scope. Read [the native milestone](npu-native-milestone.md) first. The earlier-stage description below is historical; native PD and the general `AscendBackend` integration remain outstanding.

Validated on 7 September 2026 against `/data/p00603624/models/qwen35`, inside `inferfabric-dev` on `root@7.156.99.58`. Rust was built locally under WSL and cross-compiled for ARM64. This stage reads metadata and emits a mathematical plan; it neither loads tensor payloads nor calls the NPU.

## Language and pipeline

[The complete program](../examples/qwen35-2b-checkpoint.inferfabric) starts with:

```text
contract = qwen35_text_v1
model = Qwen3.5-2B-text
hidden = 2048
intermediate = 6144
vocab = 248320
# Remaining required dimensions, ordered layers, meshes and serving buckets follow.
```

`contract` selects a versioned model definition. Dimension and layer declarations are assertions, not overrides of the checkpoint. The compiler rejects disagreement. The name is a display label, not an architecture selector. Every existing DSL field remains required, unknown/duplicate fields are rejected, and the ordinary mock parser rejects this contract line. This avoids accepting a checkpoint program as an older mock program.

The first bound contract supports the supplied Qwen3.5-2B text configuration, BF16 activations and FP32 recurrence, TP=CP=1, and SP=false. Prefill and decode use the same mathematical graph with different packed token segments. Distributed shape/communication sketches in the old compiler remain mock-only; checkpoint compilation rejects them until real sharding and collectives are lowered. Batch/context buckets and token budget are scheduling constraints carried forward to native specialization, not device-capacity guarantees.

The implemented pipeline is:

1. Parse the strict DSL and require the named math contract.
2. Check architecture, dimensions, layer schedule, gates, activation, tying, normalization and RoPE against `config.json`.
3. Read the index and each bounded safetensors header (maximum 16 MiB). Validate tensor sizes, checked byte ranges, contiguous storage, shard/index agreement and total payload size. Reject path traversal and shard symlinks outside the checkpoint directory.
4. Bind exactly 320 expected text tensors by name, dtype, shape and absolute shard offset. Exclude only the recognized vision and MTP namespaces; reject other unexpected tensors.
5. Emit 640 typed operations with explicit inputs, outputs, symbolic dimensions and state references. Check SSA dependencies, weight coverage, operation shapes/dtypes, Q/gate packing and state contracts.
6. Write deterministic `inferfabric.typed-plan.v1` JSON, including metadata fingerprint, compiler-source-dependent key, state sizes and outstanding link requirements. `executable` is always false.

The existing executable-artifact loader does not accept this JSON. Native linking must produce a separate target-qualified bundle before the runtime can admit real requests. There is no automatic mock fallback and no Python engine integration. Serving remains exclusively through the pinned vLLM 0.25.1 Rust frontend.

## Exact model semantics recorded

| Block | Contract |
|---|---|
| Embedding and output | One `[248320,2048]` BF16 tensor, shared by embedding and LM projection. Project only each request's last consumed token into logits. |
| Decoder and Q/K norm | FP32 RMS computation with epsilon `1e-6`; multiply by `1 + weight`, then cast BF16. |
| Full attention | Q projection is `[4096,2048]`; reshape to `[tokens,8,512]` and split the **last** axis into Q/gate of 256 each. K/V have 2 heads. Q/K norm precedes partial RoPE; causal GQA scale is `1/sqrt(256)`. Multiply attention output by sigmoid(gate), then output projection. |
| RoPE | First 64 of 256 channels, theta 10000000, half-rotation; mRoPE sections `[11,11,10]`. Text positions are equal across all three position axes. |
| Delta | Separate QKV, z, a, b projections. Depthwise causal width-4 convolution and SiLU on packed Q/K/V; then split the three 2048-channel sections. |
| L2 normalization | The inspected Transformers fallback performs square, reduction output, epsilon addition, reciprocal square root and multiplication in the input BF16 dtype before the recurrence casts to FP32. A fused FP32-normalization alternative requires its own numerical qualification. |
| Decay and beta | `beta=sigmoid(b)`; `g=-exp(float(A_log))*softplus(float(a)+float(dt_bias))`. No clamp. |
| Recurrence | FP32 state `[request,head,key,value]`. Decay state, predict `k^T*S`, apply beta-scaled residual, add the outer product, and read with `q/sqrt(128)`. Cast output BF16. |
| Gated norm | FP32 RMS, cast normalized values BF16, multiply **direct FP32 weight**, multiply SiLU(float(z)), cast BF16. Norm before gate; no `1+weight`. |
| FFN and residuals | Input norm → attention/delta → output projection → residual → post-attention norm → `down(SiLU(gate(x))*up(x))` → residual. |

Reference source was read, not executed: the container's Transformers `models/qwen3_5/modeling_qwen3_5.py`, SHA-256 `788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263`. That hash is recorded in the plan. These are lowering contracts; floating-point numerical agreement is not established by metadata tests.

## State and checkpoint evidence

The 320 text tensors occupy **3,763,655,360 bytes**. There are 36 FP32 tensors: 18 `A_log` and 18 gated-norm weights. The compiler excludes 297 vision tensors and 15 MTP tensors. The original shard is 4,548,221,488 bytes including its 76,648-byte header and 8-byte prefix.

The graph declares 48 state tensors: six K/V pairs, 18 recurrent matrices and 18 convolution histories. Per request, recurrent and convolution state occupy **19,537,920 bytes**. A logical 128-token page across all six attention layers occupies **1,572,864 bytes**. These totals exclude alignment, allocator overhead, scratch, activations and native workspaces.

`Tokens` denotes packed consumed tokens, `Requests` active request slots, `RequestBoundaries` the CSR offsets of length `Requests+1`, and `Pages` the physical KV page pool. Offsets partition tokens into nonempty contiguous request segments. Positions are absolute within each request. Slot IDs and page tables select persistent state. Native lowering must validate bounds, mask inactive bucket entries, and preserve state across irregular prefill chunks and decode steps.

Canonical convolution history stores the preceding three **raw projected** values per channel, oldest first. Reference/vendor APIs that retain four values need an explicit adapter. Canonical recurrence uses key-major state; the inspected CANN chunk GDR documents value-major state, so the native adapter must transpose both incoming and outgoing state. Equal key/value dimensions do not make that transpose optional.

The plan key binds metadata and compiler source, **not weight payload bytes**. Payload integrity must be established at upload/link time; changing bytes without changing headers does not alter this metadata key. No model data was copied locally; the checked-in fixtures contain only config and tensor headers.

## Native compilation decisions

Keep Rust parsing, checking, lowering, scheduling and serving in the local cross-build. Use a narrow C++ ABI for ACL/ACLNN, graph construction, kernel descriptors and Ascend C implementations; compile those inside the remote Docker container.

The source trees supply candidates, not verified callable kernels:

- `D:/cann/ops-transformer/attention/chunk_gated_delta_rule`: documents A2/A3 support, value-major state, and restricted decay domain. Select only when the actual math contract is satisfied; never clamp model decay to force compatibility.
- `D:/omni/omni-ops/inference/ascendc/src/ops-transformer/attention/ai_infra_fused_causal_conv1d`: registers `ascend910b` and `ascend910_93`; its C API has chunk/state-index operands. This is an A3 candidate where the inspected stock causal-convolution support was insufficient. Qualify state layout and graph-safe metadata updates.
- The adjacent `ai_infra_recurrent_gated_delta_rule` registers A2/A3, but both its header and op definition specify **BF16 state**. It cannot satisfy this FP32-state contract unchanged. Use a verified FP32-state vendor implementation or C++/Ascend C fallback.
- Matmul, zero-centered RMS, RoPE, paged attention, activation/gates and FP32 recurrence require explicit kernel mapping, workspace planning and numerical checks. A same-named fused op is not evidence of semantic equivalence.

Next native stages are kernel coverage/linking, lifetime-based arenas and per-bucket specialization, eager numerical qualification, then PD state transfer and ACL Graph replay. TP/SP/CP require their own lowering and numerical validation. NPU execution remains deferred while devices are occupied.

## Reproduce

Locally in WSL, with the prepared container sysroot:

```bash
export PATH=/home/p00603624/rust/cargo/bin:/usr/local/bin:/usr/bin:/bin
bash scripts/cross-compiler.sh
```

Copy `.deploy/inferfabric-compiler` and the checkpoint DSL to `/data/p00603624/inferfabric`, then execute through Docker:

```bash
docker exec -w /data/p00603624/inferfabric inferfabric-dev \
  ./inferfabric-compiler compile-checkpoint examples/qwen35-2b-checkpoint.inferfabric \
  /data/p00603624/models/qwen35 qwen35.typed-plan.json
```

Validation: 25 workspace tests pass, Clippy with warnings denied passes, ARM64 cross-build passes, actual checkpoint compilation succeeds in Docker, and two successive outputs compare byte-for-byte equal. Tests cover real config/header fixtures, malformed ranges and overflow, incompatible schemas/contracts, incorrect Q/gate splitting, output type errors and state-orientation errors. No numerical inference or NPU work was performed.
