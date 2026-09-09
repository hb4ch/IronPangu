# InferFabric

InferFabric was previously named Iron Pangu. Pangu is a planned model target, not the infrastructure name. Packages and commands now use `inferfabric`, environment variables use `INFERFABRIC_*`, DSL files use `.inferfabric`, and compiled plans use `.inferfabric-bin`. Rebuild Rust binaries and `libinferfabric_acl.so` together because C ABI symbols also changed. Existing deployment directories and containers are not renamed automatically; substitute your actual checkout and container paths in the examples.

Historical records under `docs/validation*` retain their original names and outputs. They do not validate newly built binaries. The original binary format marker is retained for format continuity.


An inference compiler and runtime for Huawei Ascend, built in Rust with a C ABI to C++/Ascend C operators and CANN libraries. Model structure and parallel execution are declared in a DSL. Serving uses the **vLLM 0.25.1 Rust frontend**, backed by InferFabric's scheduler and native execution engine.

**Working today:** Qwen3.5-2B text generation across all 24 layers, ACL Graph replay, continuous batching, NPU sampling, activation memory reuse, and DSL-configured tensor parallelism on two Ascend devices. No PyTorch or Python engine participates in model loading, execution, or serving.

## Why build this?

For pure inference, eager PyTorch is nice to have, but it is not necessary. It is a useful environment for developing models and experimenting with their mathematics. Once a model is fixed for deployment, inference needs its forward computations, persistent state, sampling, and request scheduling. Those functions can be implemented directly with C-based operators and a native runtime.

InferFabric takes that approach: Rust owns compilation, checkpoint loading, scheduling, and resource lifetimes; C-compatible operator interfaces provide the device computations. The current implementation uses C++ to call ACL/ACLNN and HCCL, with Ascend C as the path for custom kernels. Removing PyTorch means taking responsibility for operator semantics, activation storage, asynchronous execution, and numerical validation. Those are explicit parts of this project.

The objective is to make a fixed model a specialized execution program. Removing a framework alone does not establish a speedup; the opportunity comes from what the compiler can learn about the complete workload and what it can remove from the execution path.

## Why a DSL?

A model DSL gives the compiler more than a sequence of operator calls. It exposes the ordered layer structure, tensor dimensions, persistent state, parallel layout, and serving shape constraints together. That creates optimization opportunities across operator, layer, and device boundaries:

- **Memory planning:** distinguish weights and request state from temporary activations, then reuse storage after its last consumer. Residual paths and recurrent-state snapshots participate in those lifetimes.
- **Fusion and layout selection:** combine compatible normalization, rotation, gating, and elementwise operations; choose layouts that avoid intermediate transposes and copies.
- **Communication planning:** derive weight shards and collectives from a parallel layout, and eventually move or combine communication with surrounding computation.
- **Separate prefill and decode strategies:** use their different shapes and state-access patterns to select kernels, memory layouts, and graph specializations.

Memory reuse and a constrained TP implementation work today. General fusion, communication optimization, and distinct optimized prefill kernels remain development work.

For example, these declarations come from the [complete TP2 program](examples/qwen35-2b-tp2.inferfabric):

```text
contract = qwen35_text_v1
hidden = 2048
intermediate = 6144
query_heads = 8
kv_heads = 2
head_dim = 256
# Mesh fields: TP CP SP
prefill = 2 1 false
decode = 2 1 false
page_tokens = 128
batch_buckets = 1 2 4 8
context_buckets = 128 512 2048 8192
```

This is an excerpt, not a standalone program. The complete DSL also declares the other dimensions and ordered delta/full-attention layers. Dimensions are assertions against the checkpoint, not instructions to reshape incompatible weights. The versioned contract fixes mathematical details such as normalization, gating, and recurrent-state orientation. Unknown fields and incompatible checkpoints are rejected.

## Compilation toward bare-metal performance

Compilation lets us fix model hyperparameters before execution: hidden width, head dimensions, layer types, convolution width, data types, and parallel degree. A serving specialization also fixes batch capacity and context-dependent storage. These constants can produce better native binaries through constant folding, branch elimination, fixed strides and offsets, specialized loop bounds, kernel tiling, and fusion. Large weight tensors remain separately loaded data; specialization does not require embedding the checkpoint into executable code.

The target is bare-metal performance through direct native kernels, planned memory, and minimal host work between graph replays. Request-dependent values—tokens, positions, active lanes, sampling controls, and state contents—remain mutable inputs.

The current pipeline is:

```text
DSL + checkpoint metadata
    -> validated model contract and typed mathematical plan
    -> batch/context/TP specialization and weight binding
    -> prepared ACLNN operators + memory bindings + HCCL operations
    -> warmup, ACL Graph capture, numerical qualification
    -> scheduled graph replay
```

Today, the C++ bridge is compiled ahead of time, and the native runtime prepares vendor operators and captures graphs at startup. Serialized `.inferfabric-bin` artifacts contain versioned, checksummed plans; they are not standalone native model executables. Whole-program native code generation and JIT kernel specialization are roadmap items. ACL Graph capture itself is not a substitute for either.

## Current capabilities

| Area | Implemented behavior |
|---|---|
| Model | Qwen3.5-2B text path: 18 gated-delta layers, 6 full-attention layers, BF16 activations and FP32 recurrent state |
| Serving | vLLM 0.25.1 Rust frontend with chat/completions, streaming, tokenization, and cancellation |
| Scheduling | Continuous admission, mixed prefill/decode rounds, per-slot reset, `--max-num-seqs` and `--max-num-batched-tokens` |
| Sampling | NPU greedy and seeded non-greedy decoding, temperature, top-k/top-p, repetition/presence/frequency penalties, and minimum-token handling |
| Memory | Configurable context length, startup HBM profiling and budget checks, Rust live-interval planning, stable workspace and activation pools |
| Graphs | Model and sampler capture at startup; mutable device inputs during serving; inactive recurrent lanes preserved without recapture |
| Parallelism | DSL-selected TP=1 or TP=2; sharded matmul output rows, HCCL AllGather, and per-rank graph execution |

The native engine currently consumes at most one token per selected sequence per replay. A scheduler iteration can issue several prefill rounds; this is not yet a packed multi-token prefill kernel. Sequence slots reserve fixed KV regions. TP reconstructs canonical activations after each projection; optimized row/column-parallel execution and sharded attention state are still planned. Native prefill/decode disaggregation, CP, SP, and multi-node execution are not implemented. The supported model path is text-only.

### Measured validation

On the qualified Ascend A3/CANN 9.1.0 setup, TP2 with context 2048 and four sequence slots showed:

- Observed startup HBM growth decreased from **5.79 GB to 3.53 GB per rank** with memory reuse, approximately 39%; weight and persistent-state sizes were unchanged.
- Dedicated and pooled allocation produced identical startup/full-context logits and all 48 persistent state-region fingerprints on both ranks.
- Sixty mixed requests matched dedicated-allocation outputs, with no observed process HBM growth across five measurement rounds at MiB resolution.
- Graph/eager checks, independent operator CPU references, seeded sampling, streaming, cancellation, and slot reuse passed. TP1 also passed startup qualification at context 8192 with eight slots.

These measurements compare InferFabric allocator configurations, not InferFabric against PyTorch or another serving engine. See [activation memory validation](docs/npu-activation-memory.md) for reports, methodology, and limits.

## Build and run

Rust requires 1.96 or newer; the qualified local toolchain is 1.98.1. Rust ARM64 binaries are cross-compiled locally under WSL using the target container's sysroot. Native C++ compilation and NPU execution run inside Docker. See [environment and cross-compilation setup](docs/remote-development.md); its initial bring-up sections are historical.

Build the Rust frontend locally after preparing that toolchain and sysroot:

```sh
bash scripts/cross-frontend.sh
```

Deploy `.deploy/inferfabric-server` as `inferfabric-native-server`. In the NPU container, from the repository directory:

```sh
source /usr/local/Ascend/cann/set_env.sh
cmake -S native -B native-build
cmake --build native-build -j8

export HCCL_OP_EXPANSION_MODE=HOST
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
./inferfabric-native-server --native examples/qwen35-2b-tp2.inferfabric \
  /data/p00603624/models/qwen35 native-build/libinferfabric_acl.so 0 18081 \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 16 \
  --gpu-memory-utilization 0.9 --memory-profile memory-profile.json
```

The checkpoint path is deployment-specific. The device argument selects the first device; TP2 uses devices 0 and 1 here. Use [the TP1 program](examples/qwen35-2b-checkpoint.inferfabric) for one device. HCCL HOST expansion is required by the qualified TP2 graph configuration. Startup must pass memory and graph qualification before HTTP becomes ready. Add `--profile-only` to qualify and exit.

The server binds container loopback. From that container, or through a configured tunnel:

```sh
curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inferfabric-qwen35","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":16}'
```

For CPU-only development, `cargo run -p inferfabric-cli -- demo examples/qwen35-2b.inferfabric examples/requests.json` exercises the deterministic mock compiler/scheduler. Mock communication traces are not hardware qualification.

## Roadmap to production

Updated 2026-09-09. This roadmap covers the language, compiler, native execution, serving behavior, and operational work needed for a production system. Existing demos and successful hardware probes are development evidence, not a production release designation. Milestones are complete only when their exit gates pass on the declared support matrix; dates and performance claims will follow measured results.

### Starting point and release scope

| Track | Current evidence | Remaining production boundary |
|---|---|---|
| Language | Legacy checkpoint DSL and canonical Qwen mathematical IR | Proposed v1 parser, reusable modules, elaboration, state/effect checking and migration |
| Plan / execute | Verified static f32 CPU binary plans with optimizations, arena reuse and source-independent execution | General Ascend physical plans, native bundles and a loader that executes emitted decisions |
| NPU qualification | Planner test adapter passed 19,128 output/state comparisons on two devices; existing Qwen CPU/eager/graph checks passed separately | End-to-end **v1 source → native bundle → Qwen execution** through the new runtime |
| Inspection | Full Qwen typed DAG, layer navigation, tensor/weight/state bindings; CPU physical-plan viewer | Native kernels, fusion provenance, rank/stream placement, workspace, arena lifetimes and capture regions |
| Serving | vLLM 0.25.1 Rust frontend and current native Qwen backend | General-plan backend integration, API contract coverage, overload control, recovery and sustained-load qualification |

The [two-device qualification record](docs/validation/2026-09-08-npu-planning/README.md) describes exactly what was tested. Its test adapter does not turn a CPU plan into a production Ascend bundle. The v1 text parser and general native backend remain unimplemented.

The first production release will have a deliberately explicit support matrix: Qwen3.5-2B **text**, selected Ascend/CANN/driver/container versions, declared dtypes and shape limits, qualified single-node TP degrees, and the vLLM Rust frontend only. Passing TP2 on one setup does not qualify other topologies or targets. Additional models, adapters, precision modes and distributed configurations become supported individually after their own gates pass. No Python or PyTorch execution engine will be introduced into loading, enqueue, sampling or serving.

### 1. Freeze language semantics and implement DSL v1

Deliver the versioned grammar, lexer, lossless syntax tree, typed AST, source locations, formatter and `parse`/`check`/`fmt` commands. Implement package imports, named arguments, bounded compile-time composition, reusable modules and separate model/plan declarations. Define operator types, symbolic dimensions, layout constraints, state initialization/update semantics and effects. Add a migration command for the legacy format with explicit diagnostics for unsupported conversions.

**Exit gate:** conformance and negative fixtures cover every construct; parse/format round trips preserve meaning; malformed and adversarial inputs fail with bounded resource use and source diagnostics. A composed layer can be added through package files without adding a model enum or changing compiler source. Publish compatibility and deprecation rules before promising a stable language version.

### 2. Elaborate real models into a general semantic IR

Express Qwen as inspectable standard-library modules with checkpoint mappings, rotary and normalization semantics, recurrent/convolution state, attention caches and tied weights. Lower modules into a typed SSA graph with explicit effects, request boundaries and shape constraints. Treat the current optimized Qwen block path as a possible verified implementation of that graph. Validate missing/extra weights, shapes, dtypes, tied storage, checkpoint metadata and tokenizer/model configuration compatibility.

**Exit gate:** the complete Qwen model elaborates from v1 source; golden operator, layer, state-continuation and full-model comparisons match the established contract. An independently authored custom layer composes with standard modules. Ill-typed state updates, incompatible checkpoint bindings and unresolved dimensions are rejected before device allocation.

### 3. Build the Ascend physical planner

Implement deterministic target-aware kernel selection, constant specialization, dead-code elimination, layout propagation, legal fusion, rank partitioning and collective insertion. Emit the actual DAG with tensor, effect, communication and memory-reuse dependencies. Plan weights, persistent state, activations, workspace, communication buffers and graph-owned storage separately. Resolve every size to a checked constant or a verified bound; reject plans outside the memory budget. Start with a correct serial stream schedule, then add overlap only with measured benefit and sound lifetime analysis.

**Exit gate:** every emitted call has a supported kernel/ABI and explicit inputs, outputs, placement and workspace contract. Verification detects use-before-definition, illegal aliasing, missing state ordering, overflow, inconsistent collectives and overlapping live storage. Dedicated and pooled execution agree on outputs and state across shape boundaries. Kernel choices and rejection reasons are inspectable.

### 4. Emit native bundles and execute without replanning

Define a versioned native bundle containing physical IR, device/host objects or explicit linked-library requirements, specialization guards, weight bindings, relocations, entry points, ABI versions and target fingerprints. Keep large weights external and bind their identities explicitly. Implement a loader that validates the bundle, links dependencies, allocates/binds storage, prepares vendor handles and captures eligible graphs. Execution must not reconstruct a model from its name, rerun optimization or silently substitute a different implementation. Startup preparation required by CANN remains distinct from compilation.

**Exit gate:** compile a model, remove its DSL/compiler from the deployment environment, and execute its bundle in a fresh container with only the declared runtime dependencies. Reject corrupted, truncated, incompatible and unsupported bundles before readiness. Test version upgrades, cache invalidation and rollback. Checksums establish integrity; artifact provenance and trust policy are separate requirements.

### 5. Export every IR and inspect the actual native plan

Provide stable dumps for syntax/AST, elaborated graph, typed/effect IR, optimized graph, physical DAG and bundle manifest. Preserve source-to-operator and fusion provenance. Extend the full-model viewer to show per-rank kernels, tensor layouts, dependencies, arena offsets/lifetimes, workspace bounds, persistent resources and capture buckets. Add plan diffs and optional measured timing/memory overlays with the run identity attached.

**Exit gate:** a planned Qwen binary can be inspected without source or weights; every runtime call and allocation maps back to its emitted descriptor. Users can explain why a kernel was selected, where a tensor lives and which dependency permits reuse. Estimates, measured values and unavailable data remain visibly distinct. Inspection must not launch kernels or trigger JIT.

### 6. Implement and qualify native extension adapters

Define a versioned C ABI for kernel preparation, workspace requirements, launch, errors and destruction, with explicit stream and ownership rules. Build independent adapters for **Ascend C, Triton-Ascend, TileLang-Ascend and PTO**. Pin toolchains, package sources and dependencies, and export native launchers that do not depend on Python/Torch serving objects. Build-time scripting may be used in isolated tooling; it must not leak into the serving runtime. Report unsupported export routes as unavailable.

**Exit gate for each adapter:** a custom RowScale kernel, a stateful operation and a composed custom layer compile without changing the compiler, match independent references including tails and error cases, and survive capture/replay with stable storage. Record source/toolchain/object hashes and supported target/shape domains. Qualify adapters separately; an unqualified adapter cannot be selected by a production plan.

### 7. Generate optimized native code and fusion

Specialize model constants, strides, layouts, loop bounds and tiling into generated Ascend C or other qualified kernel implementations. Begin with measured normalization/rotation/gating chains and intermediate-copy elimination. Preserve an unfused qualified baseline for comparison. Add a cost model using reproducible measurements, including launch overhead, HBM traffic, workspace and communication costs; record tuning provenance with each choice.

**Exit gate:** intermediate tensors, logits and state remain within declared dtype-specific tolerances. Publish before/after latency, memory and traffic evidence on representative prefill and decode shapes. Reject an optimization that violates memory limits or regresses the declared workload budget. Removing framework overhead alone is not evidence of a speedup.

### 8. Deliver efficient prefill and scalable request memory

Implement packed multi-token prefill, chunked prompts, irregular sequence lengths and correct recurrent chunk transitions. Replace fixed per-slot KV reservation with a shared page allocator and explicit ownership/reference lifetimes. Handle mixed prefill/decode batches, admission, cancellation, slot/page reuse and memory pressure. Plan graph buckets and fallback policies for supported shapes. Add prefix reuse only after the model-specific cache contract includes all recurrent and convolution state needed to resume correctly.

**Exit gate:** varied chunking and batch composition produce equivalent outputs/state; canceled or recycled requests cannot read another request's data. Stress page exhaustion, boundary contexts, empty/inactive lanes and repeated reuse. Enforce token/sequence limits and fairness under overload. Graph buffers remain valid until all asynchronous users complete, and measured HBM stays within the configured budget.

### 9. Improve parallel and disaggregated execution

Replace projection-by-projection full AllGather with qualified row/column-parallel layouts and reductions. Add sharded attention/state where mathematically valid, then qualify additional single-node TP degrees. For prefill/decode disaggregation, transfer KV, convolution history and FP32 recurrent state with versioned ownership and completion metadata. Extend to CP/SP and multi-node topologies only after the single-node contracts are stable.

**Exit gate per topology:** compare unsharded, sharded and split execution across prefill/decode boundaries, cancellation and slot reuse. Inject rank failure, collective timeout, partial transfer and stale ownership messages; no rank may continue with partial or mismatched state. Bound retry/cleanup time, prevent duplicate state advancement and publish communication/latency measurements. Unsupported topology combinations fail before readiness.

### 10. Integrate the general runtime with the vLLM Rust frontend

Preserve the **vLLM 0.25.1 Rust frontend** as the sole frontend integration, with a documented compatibility matrix and deliberate upgrade process. Replace the fixed native-model backend boundary with verified bundle execution. Specify supported chat/completions fields, tokenization/chat templates, streaming termination, stop sequences, usage accounting, errors and cancellation semantics. Validate model-provided generation defaults and explicit request overrides. Define RNG ownership and reproducibility guarantees for seeded sampling under batching and request reuse.

**Exit gate:** API conformance and end-to-end tests cover streaming/non-streaming parity, disconnects, invalid requests, stop/EOS handling, penalties, concurrent sampling and context limits. Admission queues are bounded; deadlines and backpressure are enforced. Failed or canceled requests release resources without advancing another request's state. Readiness is published only after model, memory, kernel and graph qualification succeeds.

### 11. Add operational reliability and recovery

Implement structured errors across Rust/C++, poisoned-session handling, device/rank health checks, watchdogs and configurable deadlines. Separate liveness from readiness; support graceful drain, shutdown and restart. Quarantine failed graph/runtime instances and rebuild them before reuse. Define what happens to in-flight requests during process/device failure rather than promising transparent continuation without durable state.

**Exit gate:** fault injection covers allocation failure, bad artifacts, operator errors, failed captures, device reset, rank loss and abrupt client disconnect. The service fails closed for affected requests, reports actionable errors and returns capacity after cleanup or replacement. A documented operator runbook demonstrates recovery, rollback and safe upgrade without admitting traffic to an unqualified instance.

### 12. Add observability, deployment security and reproducible packaging

Expose bounded-cardinality metrics for queue delay, time to first token, inter-token latency, throughput, scheduler occupancy, cache hits, graph variants, HBM categories, workspace, compilation and failures. Correlate requests with model/bundle/target identities while keeping prompts, generated content and credentials out of default logs. Provide tracing and profiling modes with documented overhead.

Publish pinned runtime/build container recipes, local Rust cross-compilation instructions, dependency/SBOM and license inventories, release checksums/provenance and artifact compatibility policy. Treat custom kernels as native code: use explicit trust/allowlist policy, isolated builds and resource limits rather than claiming in-process sandboxing. Bound parser, checkpoint and bundle inputs; test unsafe FFI/lifetime boundaries. Document authentication/TLS at the supported ingress, request quotas, secret handling, least-privilege deployment and model-data access controls.

**Exit gate:** reproduce a release from declared inputs, deploy it into a clean supported environment, scrape metrics and diagnose injected failures. Test oversized/malformed requests and artifacts, incompatible dependencies and unauthorized artifact selection. Recovery and upgrade procedures include artifact/model cache compatibility and rollback. Security and operational responsibilities are explicit between the runtime, container and ingress.

### 13. Implement bounded JIT specialization and autotuning

Build JIT on the same verified planning and bundle contracts as ahead-of-time compilation. Compile eligible missing variants outside enqueue; use bounded queues, workers, timeouts and memory/disk quotas. Key caches by semantic graph, weight specialization where applicable, shapes, dtype, layout, topology, target, compiler/CANN/driver compatibility and optimization settings. Deduplicate concurrent compilation and recover from interrupted or corrupt cache entries.

Warm, numerically qualify and capture a candidate before atomic publication. Use a previously qualified compatible variant while compilation proceeds; otherwise wait within a declared deadline or reject explicitly. Retire code, graph and storage only after in-flight users finish. Allow operators to disable JIT and serve pinned AOT bundles.

**Exit gate:** cold misses, concurrent requests, tuning failure, cache eviction, process restart and rollback preserve correctness and service bounds. Measure cold-start cost and steady-state benefit. No request may execute an unqualified candidate or trigger hidden compilation inside graph replay. JIT is part of the roadmap; it is not required to enable every production deployment.

### 14. Expand models and numerical coverage

Add architectures through standard-library modules and checkpoint adapters, with independent reference fixtures before optimization. Evaluate quantization only with explicit scale/layout, accumulation and calibration contracts. Treat vision/multimodal inputs, MoE, additional dtypes and model families as separate support tracks, each with its own operator/state and API requirements.

**Exit gate per model/precision:** validate tokenizer/templates, checkpoint coverage, intermediate tensors, logits, state continuation and representative task quality. Document numerical tolerances, rounding and NaN/Inf behavior. Publish memory/latency/throughput/quality results and unsupported features. Passing Qwen text tests does not qualify another architecture or a quantized variant.

### 15. Establish release engineering and production acceptance

Automate CPU tests, parser/bundle fuzzing, sanitizer checks where supported, frontend conformance, reproducible builds and a scheduled hardware qualification matrix. Store exact source, artifact, checkpoint, compiler, driver and target identities with results. Define workload-specific latency/throughput/cold-start and memory budgets before benchmarking; compare against a declared baseline with identical models, decoding settings, hardware and load generation. Report distributions and uncertainty, not only peak throughput.

The first production release must satisfy all of these gates for **every advertised supported configuration**:

- **End-to-end implementation:** v1 Qwen source compiles into a verified native bundle, runs without its compiler/source, and serves through the Rust frontend. Language support, a test adapter or the legacy native route alone cannot satisfy this gate.
- **Correctness:** operator/layer/full-model references, state lifecycle, sampling/API contracts and eager/capture equivalence pass across declared shape limits. Disabled or unsupported features fail explicitly.
- **Capacity and service behavior:** a documented workload matrix covers short/long prompts, generation lengths, concurrency, mixed prefill/decode, overload and cancellation. Results meet predeclared service and memory budgets, with no unbounded queues or uncontrolled resource growth.
- **Endurance and isolation:** a minimum 72-hour soak includes churn, context boundaries, repeated page/slot reuse and disconnects. Live-resource accounting returns to baseline after drain; HBM stays within a declared allowance for intentional caches and allocator reservation. No stale-state exposure or unexplained growth is accepted. A soak is evidence, not proof of an availability SLO.
- **Failure handling:** device/rank/process faults, bad artifacts and interrupted upgrades meet documented detection, cleanup and recovery bounds. Readiness, drain and rollback are demonstrated under load.
- **Operational release:** versioned support matrix, reproducible runtime image, provenance, observability, security/deployment guidance, upgrade notes and an incident runbook are shipped and reviewed. Known issues have explicit scope and mitigations.

Promote through development, hardware-qualified candidate, staging soak and canary deployment before general production use. Keep a known-good artifact/image available for rollback. Broader feature support and JIT require the same gates before becoming advertised production capabilities.

### Delivery order

The critical path is **v1 syntax/semantics → Qwen elaboration → native physical planner → bundle loader/executor → Rust frontend integration → production acceptance**. IR inspection and numerical fixtures accompany every compiler/runtime milestone. Reliability, observability, security and packaging start during native integration rather than being deferred to the final release. Packed prefill and optimized TP feed the performance qualification matrix; adapters, JIT and broader models can mature independently but cannot bypass release gates.

Each milestone ships code, documented limitations, reproducible tests and qualification records, followed by a commit and push. Keep the existing qualified Qwen route available during migration; retire it only after the general runtime satisfies the same numerical and serving gates. Detailed contracts remain in the [DSL design](docs/dsl-language-design.md), [implementation plan](docs/dsl-implementation-plan.md), [physical planning design](docs/physical-planning.md) and [IR reference](docs/ir-reference.md).

## Repository and verification

| Component | Responsibility |
|---|---|
| `crates/inferfabric-dsl`, `inferfabric-model`, `inferfabric-ir` | Language, model contracts, typed execution/state metadata |
| `crates/inferfabric-compiler` | Checkpoint validation, lowering, specialization, artifact integrity and cache |
| `crates/inferfabric-native`, `native/` | Native model runner, activation planning, ACL/ACLNN graphs, HCCL, sampling |
| `crates/inferfabric-scheduler`, `inferfabric-runtime`, `inferfabric-transfer` | Scheduling and runtime/transfer contracts, including mock distributed qualification |
| `frontend/`, `vendor/rust/` | InferFabric backend integration and pinned vLLM Rust frontend |
| `crates/inferfabric-cli`, `scripts/` | Compiler/probe commands, local cross-compilation, deployment and validation |

Run local checks with `cargo fmt --all --check`, `cargo test --workspace`, and `cargo clippy --workspace --all-targets -- -D warnings`. The frontend has its own manifest: `cargo test --manifest-path frontend/Cargo.toml --offline`. Inside the NPU container, the server's `--native-smoke-test`, `--sampling-smoke-test`, and `--batching-smoke-test` modes accept the server base URL. Hardware tests supplement rather than replace numerical operator and state checks.

Further reading: [sampling](docs/npu-sampling.md), [context and HBM profiling](docs/npu-memory-profile.md), [continuous batching](docs/npu-continuous-batching.md), [tensor parallelism](docs/npu-tensor-parallel.md), [activation memory](docs/npu-activation-memory.md), and the [target architecture](docs/design.md). The README describes current implementation scope; older design and bring-up notes also contain historical milestones and future targets.

Proposed DSL v1: [language design](docs/dsl-language-design.md) and [implementation plan](docs/dsl-implementation-plan.md). These specify future syntax and extension interfaces; the current parser still accepts the legacy format.


## Physical plan / execute workflow

A locally verified CPU path now compiles a logical tensor DAG into a frozen binary plan and executes it without its source. It includes shape checks, constant folding, dead-node removal, explicit kernel selection, memory reuse dependencies, persistent state, complete IR dumps, and interactive HTML explanation. This path currently accepts logical JSON with static f32 primitives; it does not replace Ascend serving or implement the proposed DSL v1 parser.

```sh
cargo run -p inferfabric-cli -- plan examples/physical-plan/residual-state.logical.json .deploy/demo.ifplan --dump-ir .deploy/demo-ir
cargo run -p inferfabric-cli -- explain .deploy/demo.ifplan --html .deploy/demo-plan.html
cargo run -p inferfabric-cli -- execute .deploy/demo.ifplan examples/physical-plan/inputs.json .deploy/demo-results.json
```

See [physical planning and execution](docs/physical-planning.md) and the [complete IR reference](docs/ir-reference.md), with recorded examples of every produced stage.

## Full-model visual inspection

[Open the planned Qwen3.5-2B text model](docs/validation/2026-09-08-qwen-plan/model.html) in a browser: 24 layers, 640 typed operators, 320 weights and 48 state tensors. Navigate layer DAGs, follow tensor producers/consumers across layers, inspect shapes and checkpoint bindings, and search the whole model. The self-contained HTML includes downloadable typed IR and logical memory formulas.

```sh
cargo run -p inferfabric-cli -- explain-model model.typed-plan.json --html model.html
```

Input is the output of `compile-checkpoint`. This is inspection of the checkpoint-bound mathematical plan; final native allocations and kernel schedules are not yet exported. [Reference and validation](docs/validation/2026-09-08-qwen-plan/README.md).
