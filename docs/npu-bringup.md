# NPU bring-up: 7 September 2026

**Current update:** full-model native inference and Rust HTTP serving are now qualified within a single-request, context-128 scope. Read [the native milestone](npu-native-milestone.md) first. The earlier-stage description below is historical; native PD and the general `AscendBackend` integration remain outstanding.

The first native checkpoint operations now execute on Ascend devices 0 and 1. This is operator and graph qualification, not full Qwen inference. The vLLM 0.25.1 Rust frontend and mock scheduler remain unchanged; `AscendBackend` still fails closed until a complete linked model plan exists.

## Environment

`ironpangu-npu` uses image `19cc2ebef117`, the same vLLM 0.25.1 ARM64 image as `ironpangu-dev`. It runs with Docker `runc`, explicit `/dev/davinci0` and `/dev/davinci1`, the three management devices, a 2 GiB shared-memory segment, and the existing `/data/p00603624` writable bind. Driver files and npu-smi are read-only mounts. It is not privileged and does not start a Python engine. The earlier management-only development container is preserved.

All C++ compilation and accelerator execution occurred inside `ironpangu-npu`. The Rust probe was release-cross-compiled locally in WSL using the extracted container sysroot. The native ABI is now version 2 and links the installed `libascendcl` and `libopapi`.

## Implemented bridge

- Checked allocation, host/device copies, and thread-affine session ownership.
- Prepared BF16 `X[M,K] @ W[N,K]^T`, preserving the checkpoint's physical weight layout through explicit tensor strides; FP32 accumulation requested through ACLNN's default math mode.
- Prepared weighted RMS with FP32 input cast, FP32 RMS at epsilon `1e-6`, then BF16 output cast. The Rust checkpoint adapter constructs gamma as `1.0f + float(original_weight)` once at upload, preserving zero-centered semantics.
- Repeatable ACLNN executors, persistent tensor descriptors, fixed temporary buffers and shared sequential workspace. No workspace query or allocation occurs during eager execution or replay.
- Warmup, ACL Graph capture, synchronized replay, and cleanup ordered as graph destruction → executor/descriptor/workspace destruction → external buffers → stream/context.

RMS preparation exposed an executor-lifetime issue: delaying `aclSetAclOpExecutorRepeatable` until after all three workspace queries yielded error 561000 for the first stage. Retaining each executor immediately after its own successful query resolved it. The final implementation was retested on both devices.

The generic operation execute/capture APIs support both prepared operations. Existing linear-specific entry points remain compatibility aliases. The math probe dynamically loads the explicitly supplied shared-library path and checks ABI version; ordinary mock CLI operations do not load CANN.

## Numerical and graph evidence

The Rust probe validates the checkpoint through the typed compiler, reads seven selected real tensor payloads, hashes their bytes, uploads them, and compares results against an independent CPU calculation. No Python/PyTorch runtime is used.

| Case | Checkpoint tensors | Shapes |
|---|---|---|
| Delta QKV projection | Layer 0 `in_proj_qkv` | K=2048, N=6144 |
| Full attention Q+gate projection | Layer 3 `q_proj` | K=2048, N=4096 |
| MLP gate projection | Layer 0 `gate_proj` | K=2048, N=6144 |
| MLP down projection | Layer 0 `down_proj` | K=6144, N=2048 |
| Zero-centered RMS | Layer 0 input norm, layer 3 Q norm, final norm | K=2048 or 256 |

Each case runs M=1 and M=4, with three input variants on each device: **28 cases, 84 eager/reference comparisons and 84 graph/eager comparisons**. Outputs are overwritten with a sentinel before replay, and inputs change while captured addresses remain fixed. RMS variants include zero input, an epsilon-sensitive small input, and ordinary magnitudes.

CPU matrix reference uses FP64 accumulation of BF16 operands and rounds the result to BF16; it is not a bit-exact oracle for the NPU's FP32 accumulation order. Acceptance is absolute error at most `0.005 + 0.01 * abs(reference)`, with NaN/Inf rejected. On device 0 the maximum observed projection error was **0.001953125**; RMS results were bitwise equal to the CPU reference. Every graph result was bitwise equal to its corresponding eager result on both devices. These small input fixtures do not qualify arbitrary distributions, every model shape, or full-model accuracy.

[Device 0 report](validation-npu-device0-2026-09-07.json) and [device 1 report](validation-npu-device1-2026-09-07.json) contain individual errors, selected weight hashes, native-library and Rust-binary hashes, timestamps, and the bound plan key. The complete checkpoint payload has not been hashed or uploaded.

The graph-copy probe also passed on both devices, including rejection of an out-of-bounds write, an unknown graph ID and allocation from the wrong thread, followed by a successful replay proving that those rejections did not poison the session. The local Rust workspace has 26 passing tests, including BF16 rounding and CPU matrix-layout checks; Clippy passes with warnings denied. Native builds use `-Wall -Wextra -Werror`.

## Reproduce

On local WSL:

```bash
export PATH=/home/p00603624/rust/cargo/bin:/usr/local/bin:/usr/bin:/bin
bash scripts/cross-npu-probe.sh
```

Upload `.deploy/pangu-npu` and native sources into `/data/p00603624/ironpangu`. Build and run through Docker:

```bash
docker exec ironpangu-npu bash -lc '
  source /usr/local/Ascend/cann/set_env.sh
  cd /data/p00603624/ironpangu
  cmake -S native -B native-build
  cmake --build native-build -j4
  ./native-build/acl_graph_probe 0
  ./pangu-npu npu-math-probe examples/qwen35-2b-checkpoint.pangu \
    /data/p00603624/models/qwen35 native-build/libpangu_acl.so \
    0 npu-math-device0.json
'
```

Pass device 1 and a separate report path for the second device. Do not overwrite the shared library or probe executable while a probe process is running.

## Next native work

1. Implement FP32-state delta recurrence and causal convolution with exact continuation state. The installed CANN headers for both recurrent and chunk GDR specify BF16 state; header availability alone does not meet the FP32 contract. Qualify a FP32 implementation or provide an Ascend C fallback. Do not silently downcast state or clamp decay.
2. Implement the remaining transformations, partial RoPE, paged attention and gate/FFN composition; link actual operations from the typed plan rather than selecting hardcoded model operations in a serving path.
3. Add per-request native state arenas and verified transfer, complete eager-model numerical tests, and then qualify complete decode graphs and PD generation through the existing Rust frontend.

The current probe intentionally tests a bounded subset selected by weight name. It is not the eventual plan interpreter, model loader, scheduler backend, or a readiness proof for serving.
