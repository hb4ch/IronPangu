# Iron Pangu

An experimental Rust LLM inference compiler/runtime for Huawei Ascend. Model architecture and parallelism are expressed in a DSL. Necessary C++/Ascend C implementations will live behind hardware interfaces; no Python is required by this workspace.

**Current milestone: native continuously batched Qwen3.5-2B text inference through the vLLM 0.25.1 Rust frontend.** All 24 layers execute on Ascend, with eager/graph and independent CPU checks. Serving supports greedy and seeded non-greedy NPU sampling with configurable context and startup HBM profiling ([controls](docs/npu-memory-profile.md)); native continuous batching supports mixed prefill/decode ([controls](docs/npu-continuous-batching.md)); DSL-configured TP=2 is qualified ([details](docs/npu-tensor-parallel.md)); native PD remains unimplemented. See [native validation and deployment](docs/npu-native-milestone.md).

## Run

Requires Rust 1.96+ and Cargo. From the repository root:

```sh
cargo run -p pangu-cli -- demo examples/qwen35-2b.pangu examples/requests.json
cargo run -p pangu-cli -- compile examples/qwen35-2b.pangu /tmp/qwen.pangu-bin
cargo run -p pangu-cli -- inspect /tmp/qwen.pangu-bin
```

`demo` compiles/loads cached binary plans, starts separate mock prefill/decode rank threads, warms up, captures and validates every decode bucket, then admits token-ID requests. Its JSON report contains generated IDs, per-request errors, startup/step/communication traces and final resource counts. The supplied fixture deliberately cancels request 4; the other three finish. Logical ticks control fixture arrival and cancellation; this is not a real-time serving benchmark.

The commands above select **mock**. Native checkpoint loading, CANN execution and HTTP serving use the separate explicit native path documented below. Multi-process launching is not implemented. TP/SP/CP plans are structurally traced; mock ranks repeat deterministic computation and do not perform distributed arithmetic or CP state redistribution.

## Workspace

| Crate | Responsibility |
|---|---|
| `pangu-model` | Model/mesh metadata, validation and common errors |
| `pangu-dsl` | Strict line-oriented textual DSL parser |
| `pangu-ir` | Serializable rank instructions, state layouts and target identity |
| `pangu-compiler` | TP/SP/CP structural lowering, binary artifact integrity and atomic cache |
| `pangu-runtime` | Startup readiness, graph lifecycle, backend contracts, rank threads |
| `pangu-transfer` | PD protocol, leases, logical page/slot ownership, hardware transport contract |
| `pangu-scheduler` | Chunked prefill, continuous decode batching, cancellation and reclamation |
| `pangu-cli` | Compile, inspect, mock demos and native qualification probes |
| `pangu-native` | Thread-affine batched checkpoint runner with NPU sampling, HBM profiling and DSL TP |

The real DSL subset uses required `key = value` declarations, space-separated ordered layers/buckets, and `TP CP SP` mesh fields. `#` starts a comment. Unknown/duplicate fields, loops, inheritance and expressions are rejected. See the [example](examples/qwen35-2b.pangu). For a distributed structural demo use `prefill = 2 2 true` and `decode = 2 1 false`; the checked-in [distributed example](examples/qwen35-2b-distributed.pangu) has these settings.

Artifacts are versioned binary containers with a checksummed JSON instruction payload, not native model ELF files. Their keys cover normalized metadata, target/toolchain/kernel identity, flags, buckets and compiler/IR/model sources. Native binary compilation belongs to the backend. `.pangu-cache/` stores local artifacts; graph handles are always recreated at worker startup.

## Verification

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests exercise malformed DSL, artifact corruption/cache invalidation, generated communication layouts, startup failures, graph readiness, chunk invariance, continuous admission, noncontiguous pages, cancellation during handoff, stale epochs, capacity exhaustion and resource reclamation.

Read the [NPU-agent handoff](docs/npu-handoff.md), [design](docs/design.md), and [primary-source research](docs/research.md).

For the planned HTTP/chat integration, see the [vLLM Rust frontend guide](docs/vllm-frontend.md). Both explicit mock and native modes are connected.

## Docker frontend bring-up (7 September 2026)

An optional server now lives in `frontend/`, using vendored vLLM **0.25.1 Rust frontend** code and a persistent, explicitly synthetic scheduler. Local ARM64 cross-compilation and Docker HTTP/chat/SSE smoke tests pass. It does not start a Python engine. The native directory contains the qualified C++ ACLNN model and ACL Graph bridge; see the newer native milestone report.

See [remote build and run instructions](docs/remote-development.md). The original CLI demo and all Ascend `NotImplemented` boundaries remain explicit; real checkpoint inference is still pending.

Checkpoint-aware compilation is now available through `pangu compile-checkpoint`: see [the DSL/compiler guide](docs/dsl-compilation.md) and [the checkpoint program](examples/qwen35-2b-checkpoint.pangu). It validates the supplied Qwen3.5-2B tensor schema and emits a typed, non-executable mathematical plan. Native linking and numerical execution remain separate work.

See [NPU sampling](docs/npu-sampling.md) for the model-card defaults, supported controls and seeded validation.

See [activation memory management](docs/npu-activation-memory.md) for stable workspace pools, Rust live-interval planning, C++ activation reuse and dedicated/pooled NPU qualification.
