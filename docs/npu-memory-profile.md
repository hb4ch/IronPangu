# Startup HBM profiling and context length

Native serving accepts `--max-model-len N` and `--gpu-memory-utilization F`. Context is the combined prompt and output token capacity. An explicit length specializes and recompiles the canonical DSL plan, including cache and attention shapes; it need not be an existing DSL bucket. Without an override, the largest DSL context bucket is used (8192 in the example). The checkpoint limit is 262144. Unsupported or over-budget lengths fail; they are never silently reduced.

Run inside the NPU Docker container after locally cross-compiling the Rust frontend:

```sh
./pangu-native-server --native examples/qwen35-2b-checkpoint.pangu \
  /data/p00603624/models/qwen35 native-build/libpangu_acl.so 0 18081 \
  --max-model-len 2048 --gpu-memory-utilization 0.9 \
  --memory-profile native-memory-profile-device0.json
```

Add `--profile-only` to perform initialization and qualification, write the report, release resources and exit without opening HTTP. The default report is `native-memory-profile-deviceN.json`. `/v1/models` advertises the selected context length.

The startup run measures weights, KV cache, recurrent/convolution state, other persistent buffers, operator temporaries, CANN workspaces and observed runtime/allocator overhead. It prepares and captures the model and both sampler graphs, then compares eager/captured logits at the final context position and exercises non-greedy sampling before resetting request state. This final-position run uses synthetic cache contents; it qualifies memory and graph execution, not long-document answer quality. Prefill remains sequential per sequence; [continuous batching](npu-continuous-batching.md) controls resident slots and mixed token rounds.

The default budget is 90% of total device HBM, capped at current free memory minus a 256 MiB reserve. Bridge allocations are checked before allocation; snapshots and replay checks also detect budget use by CANN. A weights-only insufficiency is rejected before upload. Initialization must pass before the server becomes ready. This is a budget check for the requested context, not a search for the largest possible context or a multi-request KV pool allocator.

Measurements use free-memory differences from the initialized ACL context and allocation/phase watermarks. They do not trace every short-lived SDK allocation. Context creation is already present in the baseline, and concurrent processes can affect measurements. Use an otherwise idle device for reproducible profiles.

Validation on 2026-09-07: context 2050 (outside the original buckets) passed with bitwise equal final-position logits. Observed HBM growth was 5,898,883,072 bytes: 3,763,862,208 resident weight bytes and 2,135,020,864 non-weight bytes. KV cache was 26,738,688 bytes and workspace was 1,342,449,408 bytes. A 0.001 utilization budget failed before weight upload; a direct native allocation probe rejected an allocation exceeding its 2 MiB budget. See the accompanying JSON reports in `validation-memory-2026-09-07`.

HTTP qualification at context 2048 passed a real 512-token prompt with two generated tokens, context overflow rejection, chat/completion correctness, SSE agreement, state reset, admission rejection and disconnect recovery. Seeded sampling replay, four distinct seed outputs, greedy seed independence and minimum-token handling also passed. Rust workspace tests (31), frontend tests (5), and Clippy passed. Rust binaries were cross-compiled locally; ACL compilation and NPU tests ran inside `ironpangu-npu`.

Profiling now includes all `--max-num-seqs` lanes and per-rank TP allocations. Eight slots at context 8192 passed startup graph/state qualification, with observed HBM growth of 9,231,245,312 bytes. See `validation-batching-2026-09-07/batching-8-8192.json`.
