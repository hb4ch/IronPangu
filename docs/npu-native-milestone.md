# Native Qwen milestone — 7 September 2026

The supplied Qwen3.5-2B text checkpoint now runs all 24 layers on Ascend and serves real completions/chat through the vendored **vLLM 0.25.1 Rust frontend only**. Rust is cross-compiled locally in WSL; C++ builds and all accelerator execution take place in `ironpangu-npu` on `root@7.156.99.58`, under `/data/p00603624/ironpangu`. No Python inference engine is used.

## Qualified scope

The compiler validates the versioned checkpoint DSL, emits a canonical typed mathematical plan, then `pangu-compiler::lower` maps that plan into ordered native block bindings. Lowering rejects edited/noncanonical graphs and undeclared context buckets. All 320 text tensors are bound; actual uploaded payloads are SHA-256 hashed. Norm weights are transformed once to FP32 `1 + weight` where required. The mathematical plan remains non-executable by itself: runtime loading, preparation and successful startup qualification are required.

`pangu-native` owns one complete model on one dedicated device thread. The C++ ABI prepares embedding, 18 delta blocks, six full-attention blocks, final normalization and tied output projection. ACLNN descriptors/executors/workspaces and graph addresses persist across requests. Startup compares eager and graph logits and all 48 state regions, then clears state before reporting ready. Each token replay updates token/position/append/mask metadata without device allocation or recapture.

The experimental HTTP mode has **one active request, greedy sampling, context 128, TP=CP=1 and no SP**. Prefill consumes tokens sequentially through the same graph; this is a correctness baseline, not optimized chunked prefill. State is reset between requests. Overlap is rejected, stream drop cancels execution, unsupported sampling is rejected, and native failures stop readiness. The frontend may clamp requested output length to the remaining context before backend admission.

The existing mock scheduler and its PD/continuous-batching tests remain separate. `AscendBackend` and registered hardware transport are still unimplemented. This single-request engine does not establish native PD, continuous batching, multi-rank execution, a persistent compiled native cache, or performance parity.

## Numerical evidence

Reports are in [validation-stateful-2026-09-07](validation-stateful-2026-09-07/). They record binary/weight identities and individual cases; hashes differ across successive native-library revisions.

- FP32 delta recurrence: both devices, rectangular 128x64 and production 128x128 state, multiple continued tokens and decay down to -80 without clamping. Maximum CPU state error 7.45e-9; eager/graph outputs and state match bitwise.
- Width-four causal convolution: both devices, 6144 channels, batch 1/2, continued history. CPU outputs/history and eager/graph results match bitwise.
- Complete first delta decoder layer: real weights, three continued tokens, independent Rust CPU block reference. Maximum relative L2 output error 0.00108513; eager/graph outputs and both state regions match bitwise.
- Full attention: two requests, noncontiguous physical pages, six positions crossing a page boundary, real Q/K norm weights, dynamic positions/tables/masks. CPU and eager/graph outputs match bitwise on both devices.
- Complete model, device 0: all 320 weights and three continued input tokens `[1,2,3]`. Independent scalar Rust CPU logits have relative L2 errors 0.014174, 0.011338 and 0.014601. Top-1 tokens agree at all positions: 5328, 220, 16. Native eager/graph logits and every state region match bitwise. This is a small correctness sample, not broad language-model evaluation.
- Actual HTTP checkpoint: completion begins ` Paris` for `The capital of France is`; chat answers `4` for `What is 2 + 2? Answer briefly.` and stops on EOS. SSE/collected agreement, different-request state reset, unsupported sampling, context overflow, overlapping-request rejection and disconnect recovery pass.

The installed gated-delta operator rejects FP32 state (status 161002) while accepting BF16 state. The implementation therefore uses a C++ ACLNN composite preserving FP32 state and unclamped decay. Local CANN and omni-ops source support alone is not treated as proof of installed binary support.

## Reproduce

Local WSL, using the extracted image sysroot:

```bash
export PATH=/home/p00603624/rust/cargo/bin:/usr/local/bin:/usr/bin:/bin
bash scripts/cross-npu-probe.sh
bash scripts/cross-frontend.sh
```

Transfer sources and binaries to the remote workspace. Build the native library and launch **inside Docker**:

```bash
docker exec ironpangu-npu bash -lc 'source /usr/local/Ascend/cann/set_env.sh && cd /data/p00603624/ironpangu && cmake -S native -B native-build && cmake --build native-build -j4'
docker exec ironpangu-npu bash -lc 'source /usr/local/Ascend/cann/set_env.sh && cd /data/p00603624/ironpangu && ./pangu-native-server --native examples/qwen35-2b-checkpoint.pangu /data/p00603624/models/qwen35 native-build/libpangu_acl.so 0 18081'
```

The server binds `127.0.0.1:18081` **inside the container**, model name `ironpangu-qwen35`. The existing mock server on 18080 in `ironpangu-dev` is separate. The native debug build spends time hashing/uploading weights and preparing operators before readiness.

```bash
docker exec ironpangu-npu bash -lc 'cd /data/p00603624/ironpangu && ./pangu-native-server --native-smoke-test http://127.0.0.1:18081'
docker exec ironpangu-npu bash -lc 'source /usr/local/Ascend/cann/set_env.sh && cd /data/p00603624/ironpangu && ./pangu-npu npu-model-probe examples/qwen35-2b-checkpoint.pangu /data/p00603624/models/qwen35 native-build/libpangu_acl.so 0 examples/qualification-token-ids.json npu-model-device0.json'
```

Local validation: 28 workspace tests and three frontend tests pass; both workspaces pass Clippy with warnings denied, and ARM64 cross-builds pass. SIGINT shutdown was checked: the server exited and npu-smi reported no remaining accelerator processes before restart.

Stop the native server before running the model probe on its device. Send SIGINT to its verified container PID for orderly frontend and ACL cleanup.

## Next implementation stage

1. Broaden model/reference qualification to real tokenizer prompts, longer contexts and more sequences; record latency and memory separately from correctness.
2. Expose typed device state regions/frontier ownership from the native engine and implement actual P-to-D transfer with visibility fences, preserving the existing reserve/commit/ACK contract.
3. Connect native state to scheduler slots/page ownership; qualify cancellation, reuse, continuous admission and every additional batch/context graph bucket.
4. Implement and numerically qualify TP/SP/CP collectives and sharding after the single-rank state path is stable.
