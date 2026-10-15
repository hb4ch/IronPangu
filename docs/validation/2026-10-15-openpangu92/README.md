# OpenPangu V2 92B: checkpoint binding and native W8A8 qualification

Date: 2026-10-15. This milestone starts the 92B inference implementation. It does **not** provide a complete decoder, generated execution plan, or serving endpoint for OpenPangu yet.

## Real checkpoint contract

The checkpoint at `/mnt/sdb_data/models/92B` (mounted read-only as `/models/92B`) uses `model_type=openpangu_v2`, 46 main layers, three declared MTP layers, dynamic W8A8 compressed-tensors linear weights, mHC, MoME, learned sinks, and shared sparse indexers. It differs from the earlier local reference config, including tokenizer IDs and quantization.

[Checkpoint summary](checkpoint-summary.json) records metadata identities and measured sizes:

- 73,790 stored tensors in 50 shards; 104,347,285,362 stored tensor bytes.
- 69,092 main-model tensors bound by exact name, shape and dtype; 93,415,683,954 bytes.
- 10,931,601,408 bytes in declared MTP layers explicitly excluded from the initial inference contract.
- The index declares 200,272,648,050 bytes, which does not match the stored quantized data. Default validation rejects it. `--allow-stale-index-size` permits this discrepancy only, preserves both values, and leaves all tensor-range, ownership and shape checks enabled. No checkpoint file was modified.
- The tokenizer JSON is approximately 20 MiB; hashing accepts at most 64 MiB for this file. Other metadata keeps the 16 MiB bound.

The binding contract records mHC FP32 casts, MoME channel/width layout conversion, W8A8 per-output scales and shared-index dependencies. It rejects unsupported quantization schemes, missing weights/scales, extra main-model tensors and incompatible shapes. It does not verify all weight payloads or choose device placement. Stored bytes are not peak HBM.

## Hardware checks

The other workloads released the devices before testing. No processes were killed. Testing used `inferfabric-92b` on `root@7.156.99.58`, with all compilation/execution inside Docker and Rust cross-compiled locally. Workspace: `/data/p00603624/inferfabric/openpangu92`.

Image: `registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a3-arm-develop-202610090945-debug:0.0.1`.
Image ID: `sha256:bf260616632829c9367f88f6a63b5d385f9cddb220333d69235d1cde1b1fb0cb`.
Compiler: GCC 10.3.1; CANN installation: `/usr/local/Ascend/cann-9.1.0`; driver reported by npu-smi: 25.5.1.

The initial container with explicit device nodes failed ACL initialization (`500000`, driver console log query returned 4). The qualification container uses the same privileged driver-access mode as the existing working development containers. This is a development setup, not a least-privilege production deployment recipe.

The native probe uses `aclnnDynamicQuant` and `aclnnQuantMatmulV5` with per-token activation scales and per-output weight scales. Real checkpoint projections tested:

| Projection | Weight shape `[output,input]` | Purpose |
|---|---|---|
| `model.layers.0.self_attn.q_b_proj` | `[9216,1024]` | Attention projection |
| `model.layers.0.mlp.down_proj` | `[2560,9216]` | Dense FFN projection |
| `model.layers.2.mlp.experts.0.gate_proj` | `[1024,2560]` | Routed expert projection |

Each projection passed at batch sizes 1 and 4 on devices 0 and 1: **12 cases**. Each case checks three inputs (two distinct signed patterns and zeros), eager execution, and three rounds of graph replay. Graph replay matches eager BF16 outputs bit-for-bit. Inputs, outputs, scales and workspaces retain stable addresses; replay invokes no workspace query, operator preparation, allocation or recapture.

Dynamic quantization matched the CPU round-to-nearest reference exactly for these inputs. Matmul is checked against an independent CPU integer dot product using the observed device quantized inputs/scales, followed by weight scaling and BF16 rounding. Maximum absolute error was 0.015625 for attention batch 4; all other cases reported zero. The checker allows `0.0001 + 0.015625 * abs(expected)` for BF16 numerical differences and at most one integer quantization step (observed zero). This is kernel qualification on synthetic activations, not a full-model accuracy or throughput result.

CANN rejected `aclSetAclOpExecutorRepeatable` for DynamicQuant with status 561000 (`launcher repeat check fail, find tensor relation fail`). The probe uses one-shot executors for eager warmup and capture, then replays the captured graph directly. This lifecycle must be preserved when integrating the operation into the model executor. The existing Qwen runtime was not changed.

[Reports](reports/) include per-case results, projection payload/fixture hashes and native artifact hashes. Weight payloads are kept on the remote machine and are not committed.

## Reproduce

Local WSL, after the [cross-build setup](../../build.md):

```bash
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
bash scripts/cross-npu-probe.sh
bash scripts/cross-openpangu-probe.sh
```

Upload `.deploy/inferfabric-npu`, `.deploy/export_openpangu_linear`, and the native sources to the container workspace. Inside Docker:

```bash
source /usr/local/Ascend/cann/set_env.sh
./inferfabric-npu bind-openpangu-checkpoint /models/92B bound.json --allow-stale-index-size
./export_openpangu_linear /models/92B model.layers.0.self_attn.q_b_proj qb.bin --allow-stale-index-size
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Release
cmake --build native-build --target openpangu_linear_probe -j4
./native-build/openpangu_linear_probe qb.bin 0 1
./native-build/openpangu_linear_probe qb.bin 0 4
```

Repeat with device 1 and the other two projection names. The fixture exporter reads only the selected validated weight/scale payloads, checks finite positive scales, and writes `IFW8A801` fixtures plus SHA-256 provenance. The C++ probe performs inference math without Torch/Python. The binding JSON is an inspection artifact, not a directly executable bundle.

Local validation: 66 workspace tests passed, formatting and Clippy passed, and the exporter cross-compiled locally for ARM64. A reduced four-layer/two-expert metadata fixture tests dense/MoE/mHC/MoME bindings and rejects missing scales, incorrect dtype/shape, unsupported quantization, missing shared-index producers and extra tensors. No payload is included in that fixture.

## Next implementation gates

1. Expose the qualified W8A8 preparation/capture lifecycle through the native runtime ABI, with scale validation and bounded load staging.
2. Implement and qualify mHC pre/Sinkhorn/post and final stream merge, MoME causal histories, sigmoid routing with correction bias and routed/shared expert combination.
3. Implement MLA projections/cache, RoPE, sinks, SWA and sparse index selection/reuse; compare complete decoder layers against frozen reference outputs.
4. Plan rank placement for the measured main-model weights plus casts, staging, cache, graph and workspace memory; add expert/tensor-parallel execution and collectives.
5. Run all 46 layers, verify logits/generation, then integrate the vLLM 0.25.1 Rust frontend, generation defaults, tokenizer and batching/cancellation semantics. Full-model correctness precedes performance claims.

MTP remains explicitly disabled until the base model passes. No 92B endpoint or production-readiness claim is made by this milestone.
