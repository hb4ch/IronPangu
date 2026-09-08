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

## Roadmap

1. **Native code generation and fusion.** Lower typed operations into specialized Ascend C kernels and per-rank native modules. Start with measured normalization/rotation/gating chains and eliminate intermediate writes. Encode model constants, layouts, and target identity in generated artifacts; require numerical equivalence and measured launch/HBM-traffic reductions.
2. **Packed prefill and shared KV capacity.** Process multiple prompt tokens per sequence in a kernel invocation, handle irregular recurrent chunks correctly, and allocate KV pages across active requests. Preserve cancellation, state continuity, and graph address stability while reducing padded work and reserved memory.
3. **JIT specialization and autotuning.** Compile missing kernel variants for supported batch/context buckets, layouts, and TP configurations. Cache binaries by model contract, shape, dtype, target, compiler/CANN versions, and optimization settings. Warm up and validate each variant, then capture its graph before publishing it to serving. Bound compilation concurrency and cache memory; retire graph/storage entries only after in-flight work completes. An already qualified compatible variant can serve while a new variant compiles. Unsupported shapes must wait or fail explicitly. Measure cold-start cost as well as steady-state performance.
4. **More efficient tensor parallelism.** Introduce complementary column/row-parallel projections and reductions, reduce full-activation AllGather operations, and qualify sharded attention/state layouts. Measure communication volume and latency before expanding beyond the current two-rank configuration.
5. **Native prefill/decode disaggregation.** Transfer KV pages, convolution history, and FP32 recurrent state together with versioned ownership metadata. Prove split execution matches uninterrupted generation under cancellation, retries, and slot reuse before adding CP/SP or multi-node layouts.
6. **Broader model and performance qualification.** Add model contracts with operator/state fixtures, then evaluate quantized kernels and additional architectures. Publish reproducible latency, throughput, memory, and accuracy comparisons across context lengths and concurrency, including compilation/capture costs.

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
