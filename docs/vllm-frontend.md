# vLLM Rust frontend integration

InferFabric now includes an optional server using the **vLLM v0.25.1 Rust frontend**, pinned to `752a3a504485790a2e8491cacbb35c137339ad34`. The upstream Rust source and license are in `vendor/`. `frontend/` is a separate Cargo workspace so the original CPU-only core build stays small and independent of CANN.

## Implemented path

```text
vLLM Rust HTTP / chat templates / tokenizer
  -> vllm-llm::GenerationBackend
  -> persistent InferFabric scheduler thread
  -> mock P workers -> full mock hybrid-state handoff -> mock D workers
  -> bounded token stream -> vLLM incremental decoding / SSE
```

`serve_with_backend` injects generation directly. No engine-core handshake, Python engine, managed launcher, or per-request engine startup occurs. The engine-core protocol crate remains a Rust type dependency. Core-only administration endpoints reject calls when no engine-core client exists.

The server supports `--mock` (advertising `inferfabric-mock`) and `--native DSL CHECKPOINT LIB DEVICE [PORT]` (advertising `inferfabric-qwen35`). Native mode runs real weights through a dedicated ACL thread, with continuous admission, mixed prefill/decode and configurable context ([memory profiling controls](npu-memory-profile.md)). See [native deployment and validation](npu-native-milestone.md). See [continuous batching](npu-continuous-batching.md) and [DSL tensor parallelism](npu-tensor-parallel.md). The scheduler path described below is the separate synthetic mock mode.

The scheduler initializes P/D workers once, performs token-budgeted prefill and continuous decode, publishes the first token after commit/ACK, and tracks independent request IDs. Dropping the output stream requests cancellation; bounded output queues abort slow consumers instead of blocking shared scheduling. Unexpected stream closure becomes an error. Shutdown retires active requests and closes workers. Ordinary requests use deterministic mock generation with `temperature=0`; unsupported sampling options return HTTP 400.

## Build and validation

See [remote development](remote-development.md) for local ARM64 cross-compilation and Docker-only remote execution. The image-specific sysroot is required for the full frontend.

Local checks:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo clippy --manifest-path frontend/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path frontend/Cargo.toml
```

Container acceptance client:

```sh
./inferfabric-server --smoke-test http://127.0.0.1:18080
```

It checks readiness, chat, SSE/collected agreement, concurrent requests and unsupported options through the actual vLLM Rust routes.

## Remaining integration work

Connect an explicitly selected hardware scheduler to the same generation interface after implementing and qualifying checkpoint loading, Qwen math, typed state regions and NPU transport. Replace mock completion fences with actual device visibility guarantees. Add real tokenizer/template golden fixtures when checkpoint metadata arrives. Performance, NPU graph execution, numerical correctness and real PD remain unvalidated.

Keep frontend and hardware readiness coupled: no requests before all supported graph buckets and required P/D workers are ready, and no automatic mock/eager fallback on hardware failure.

Native sampling now supports temperature/top-k/top-p/min-p, seeds and repetition/presence/frequency penalties. Omitted values inherit checkpoint/model-card defaults before lowering; explicit request values are preserved. See [NPU sampling](npu-sampling.md).
