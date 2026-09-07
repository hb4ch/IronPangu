# NPU-node agent handoff

**Current update:** full-model native inference and Rust HTTP serving are now qualified within a single-request, context-128 scope. Read [the native milestone](npu-native-milestone.md) first. The earlier-stage description below is historical; native PD and the general `AscendBackend` integration remain outstanding.

## What works today

The Rust skeleton runs token-ID requests through a real parser, compiler, binary cache, startup state machine, bounded worker queues, logical page/slot allocator, PD ownership protocol and iteration scheduler. Its backend is explicitly synthetic. Do not interpret passing tests or generated tokens as Qwen correctness, actual paged attention, real graph capture, or real TP/CP execution.

`pangu-runtime::Backend` and `pangu-transfer::RegisteredTransport` are the hardware extension points. Their default methods return `Error::NotImplemented`. `AscendBackend` and `AscendTransport` deliberately implement none of the hardware methods. No C++ ABI has been guessed. The CLI only selects mock mode; preserve this explicit separation when adding an Ascend launcher.

## Contracts to implement

| Rust contract | NPU responsibility |
|---|---|
| `Backend::compile_kernels` | Resolve compatible ACLNN kernels, compile/cache missing Ascend C specializations; return a target-matching native bundle |
| `prepare`, `upload_weights` | Device/context ownership, rank communicators, stable arenas, workspace/descriptor preparation, safetensors weight upload |
| `allocate`, `release`, `state_region` | Opaque allocation handles and checked offset/byte regions; no foreign pointer reinterpretation |
| `warmup`, `capture`, `validate_graph`, `reset_scratch` | Warmup outside capture, ACL Graph per batch/context bucket, actual replay validation and disposable-state reset |
| `execute` | Walk the supplied rank plan in dependency order for eager prefill; replay the prepared graph for decode; return matching request/epoch/frontier results |
| `communicate`, `synchronize` | HCCL lowering and stream completion; enqueue matching ring sends/receives without blocking every sender before receives are posted |
| `close` | Fence work, destroy graph instances, then free referenced descriptors/workspaces/buffers and communicators; support cleanup after partial initialization |
| `RegisteredTransport::{register,transfer,wait_visible,deregister}` | HIXL registered regions, transfer handles, receiver device visibility and safe deregistration |

`BufferHandle`, `GraphHandle`, `Registration` and `TransferHandle` are backend-local opaque integer IDs, not raw pointers. Use internal maps/RAII to own actual resources. Backend execution/completion calls are synchronous at the Rust boundary: when they return successfully, result state is safe to consume. Device submission can be asynchronous internally, but the completion contract must hold. C++ exceptions must not cross a future C ABI.

The worker owns its backend on one dedicated thread. All ranks receive the same logical step; `WorkerGroup` waits for every rank before returning the leader result. The current mock factory is explicit; add an Ascend factory only with real target validation. The mock repeats full logical state on each rank, so it does **not** implement physical sharding or numerical collectives.

## State and transfer integration

`StepInput` carries request/epoch, tokens, consumed frontier, a state slot/generation and physical logical-page IDs. The runtime rejects aliased slots/pages and insufficient page capacity before execution. Map those IDs to stable NPU arenas and device-side metadata. Keep page-table and length data mutable without changing captured addresses. No shared-prefix pages are supported yet.

`HybridState` currently contains mock markers for KV, recurrent matrices and convolution history. These are test payloads, not tensors. Replace/adapt their storage with typed device-region descriptors for the hardware runner; do not copy or upload marker vectors as model state. Preserve the request/epoch/frontier and ownership protocol. `StateLayout` supplies per-layer shapes and dtype, with TP-local heads. Full-attention shape is per page; recurrent/conv shapes are per request. A real adapter must qualify exact layouts, Qwen normalization/gates/position parameters, alignment and operator domains against the pinned checkpoint and selected CANN release.

`InMemoryTransport` models reserve → transfer → complete → commit → ACK. Its snapshot copy is deliberately local. Add the registered transport data path using the same control-state transitions. A completion tick in the mock is not an acceptable device fence. Device state manifests need region/rank identities and actual completion events. CP-to-decode page redistribution and selection of final recurrent/conv owners remain hardware work; generated CP instructions only describe the intended communications.

At P/D handoff, `consumed=L` means prompt tokens `[0,L)` have been processed. `next_token=y0` is generated from final prompt logits and has **not** been consumed. D processes it at position L. The first token is published after commit/ACK. P pages survive through ACK. Aborted leases are retired and reusable only after a completion fence. Duplicate manifests/ACKs are idempotent while the lease is live; after finish, old epochs return `Stale`. Retained mock manifests are freed when a request finishes.

## Startup and artifact invariants

Startup validates the artifact and backend identity, resolves native kernels, prepares resources, warms up, captures and replay-validates **all** batch×context decode buckets, resets scratch, and only then admits requests. A cache hit reuses plans, never graph handles. Exceptions/failures prevent readiness; never choose eager or mock automatically. Once ready, the serving path cannot compile or capture. Out-of-capacity requests fail explicitly.

The current artifact format is 8-byte versioned magic, 8-byte little-endian payload length, 32-byte SHA-256 digest, and a JSON-encoded typed rank program. This is an instruction binary container, not native device code. The compiler additionally reconstructs and compares the canonical plan when reading it. Native bundle caching is your adapter's responsibility. Include actual compiler/CANN/kernel/SoC/flags/weight-layout revisions in target identity; do not use the mock strings for Ascend. Vendor-toolchain Python dependencies must be qualified against the no-Python startup requirement before selecting that build path.

The first structural compiler uses replicated embedding/vocabulary head/sampling, TP-sharded attention and MLP, SP token-layout transitions, exact-attention CP ring descriptors and ordered recurrence-boundary handoff. Kernel names are operation contracts, not callable CANN symbols. Extend descriptors with selected kernel ABI parameters during hardware lowering rather than hardcoding guessed symbols in the generic compiler. The small DSL lacks the complete Qwen mathematical contract; add verified checkpoint-specific parameters before claiming numerical execution.

## Hardware acceptance work

1. Select SKU, driver, firmware, CANN release and checkpoint paths. Qualify operator coverage; local causal-convolution support differed between A2/A3 and 950.
2. Add backend/unit probes for allocation lifetime, BF16 math, hybrid state, communication and graph metadata mutation.
3. Connect actual device-region PD transfer and ensure receiving streams see writes before admission.
4. Validate Qwen eager correctness, chunk invariance and transferred continuation. Then require graph/eager agreement across lengths, page changes and slot reuse.
5. Validate TP/SP/CP arithmetic and collectives independently; mock trace tests establish only structural intent.
6. Keep CPU tests passing. Add hardware tests behind an explicit feature/test target requiring the NPU environment. Never gate CPU builds on CANN installation.

Deferred beyond this skeleton: HTTP, tokenizer/chat templates, actual safetensors parsing, distributed process supervision, real network control messages, optimized sampling and performance benchmarking. These need implementation in addition to filling kernel bodies for a genuine model-serving demo.

## 7 September 2026 code-first bring-up

Read [remote development](remote-development.md) for the current Docker image, local cross-build, and verified commands. `frontend/` now adapts the pinned vLLM 0.25.1 Rust frontend to `pangu-scheduler::live::LiveScheduler` through an in-process generation trait. Its only constructor is explicitly mock. `native/` now contains an installed-header-checked C++ ACL ABI and graph-copy qualification probe; it is not yet an implementation of `AscendBackend` or `RegisteredTransport`. No weights or NPU execution were used in these checks.

## Checkpoint-bound compilation (7 September 2026)

The supplied checkpoint is `/data/p00603624/models/qwen35`. [DSL and compilation](dsl-compilation.md) documents the implemented metadata binder and typed mathematical IR. `compile-checkpoint` validates 320 text tensors and emits 640 typed operations, with `executable=false`. Only TP=CP=1 is bound; the earlier distributed mock programs remain structural. Do not feed this JSON to the mock artifact loader or claim native linkage. The native adapter must preserve the new key-major FP32 recurrent layout and three-value convolution history (or explicitly convert to its physical layout); older structural state descriptors are not interchangeable.

## Native operator milestone

See [NPU bring-up](npu-bringup.md) and its two machine-readable reports. BF16 checkpoint matrix projections and FP32-computed zero-centered RMSNorm now pass eager/reference and graph/eager tests on devices 0 and 1. The C++ ABI exposes prepared operations with persistent executors/workspaces. These operations are not yet linked into the typed full-model graph or `AscendBackend`; recurrence, convolution, paged attention, real hybrid state transfer and complete serving remain outstanding.
