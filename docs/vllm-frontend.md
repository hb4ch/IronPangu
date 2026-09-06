# Connecting the vLLM Rust frontend

## Current status

**There is no vLLM frontend integration in Iron Pangu yet.** The current CLI supports `compile`, `inspect` and `demo`. It has no HTTP listener, vLLM handshake endpoint or live token stream. The executable skeleton deliberately deferred these pieces.

Today you can run the orchestration demo:

```sh
cargo run -p pangu-cli -- demo examples/qwen35-2b.pangu examples/requests.json
```

This produces mock token IDs and a final JSON report, not a chat service. The steps below are an implementation guide, not commands for an already available integration.

## Upstream boundary

The official vLLM Rust CLI's managed `serve` example launches a Python engine beside the Rust frontend. Its separate `frontend` command connects to an existing vLLM engine endpoint. Neither command can currently connect to Iron Pangu. [Upstream CLI guide](https://github.com/vllm-project/vllm/blob/main/rust/src/cmd/examples/README.md)

In the inspected source, `vllm-llm::Llm` owns a concrete `EngineCoreClient` and exposes generation streams, abort and shutdown. It is not a generic backend trait that our runtime can implement unchanged. [LLM facade](https://github.com/vllm-project/vllm/blob/main/rust/src/llm/src/lib.rs)

The HTTP server constructs that engine client and then the LLM, text and chat layers. Its router-extension hook adds routes; it does not replace engine construction. [Server construction](https://github.com/vllm-project/vllm/blob/main/rust/src/server/src/lib.rs)

Sources inspected on 6 September 2026. These are moving `main` links; pin an upstream commit and record it before modifying/reusing code.

## Recommended integration

Adapt the Rust frontend at its tokenized generation boundary:

```text
OpenAI-compatible HTTP request
  → reused/adapted Rust chat templates and tokenizer
  → Iron Pangu generation adapter
  → persistent Rust scheduler
  → P workers → hybrid-state handoff → D workers
  → token events → incremental detokenization → SSE response
```

Keep the HTTP/chat/tokenization layer separate from the DSL compiler and Ascend implementation. Start by connecting it to the mock backend; replace that backend only after the NPU implementation passes its tests. No frontend feature should start a Python engine.

A vLLM wire-protocol compatibility server is another possible approach, but it requires implementing the pinned engine's handshake, request/output encodings, lifecycle and administrative semantics. Avoid taking on that compatibility surface for the first integration. A small maintained fork adapting the generation facade is the proposed first approach.

## Work required in Iron Pangu

1. **Make the scheduler persistent.** Refactor `run_mock(artifact, Vec<Request>, capacity)` into a long-lived engine with bounded `Submit`, `Cancel` and `Shutdown` commands. Keep the fixture runner as a test client. Initialize workers once, rather than once per HTTP request.
2. **Stream engine events.** Emit token events when they become available, plus terminal finish/error events. Preserve the existing rule that the first output token is published only after PD commit/ACK. Replace fixture `submit_at`/`cancel_at` ticks with real channel commands in the serving adapter.
3. **Expose a narrow frontend contract.** Proposed operations are submit(token IDs, request ID, generation limits), cancel(request ID), readiness and shutdown. Submission returns a request-scoped event receiver. These operations do not yet exist as a public API.
4. **Preserve identity and backpressure.** Map external string IDs to internal request/attempt IDs. Cancel on disconnect or dropped response stream. A full event queue must not block the shared scheduler indefinitely: abort the affected request and fence its state before reclamation. Repeated cancellation is harmless.
5. **Publish readiness only after startup.** DSL/native compilation, warmup, all-bucket graph capture/validation and required P/D readiness precede request admission. Never compile/capture inside an HTTP handler. Return an explicit unavailable response if the engine failed startup.

The current `Request` contains token IDs and limits, and `Report` collects all results. Wrapping `run_mock` in one blocking task per HTTP request would create independent engines and lose cross-request batching; that is not the intended adapter.

## Work required in the frontend fork

- Pin the vLLM revision, preserve license notices, and replace concrete engine-client construction with an injected generation service. Adapt downstream stream/output types where they depend on vLLM engine-core types; this is not necessarily a single-file change.
- Retain suitable Rust HTTP, chat-template, tokenization, incremental decoding and SSE code. Supply model metadata from the pinned Qwen checkpoint and the Iron Pangu capability description, without requesting it from a Python engine.
- Remove the managed-Python launcher from the Iron Pangu executable. Audit the selected dependency graph and renderer/parser paths: the upstream workspace lists Python-related crates, which does not by itself prove every frontend build requires them. [Workspace manifest](https://github.com/vllm-project/vllm/blob/main/rust/Cargo.toml)
- Initially support text-only, one completion per request, a bounded output length and the backend's actual decoding mode. The mock is deterministic synthetic output; a real greedy decoder remains NPU work. Reject unsupported sampling, logprobs, tools and multimodal requests explicitly rather than accepting parameters that have no effect.
- Map token limits, EOS, finish reasons, errors and usage consistently. Test incremental Unicode decoding and chat-template token IDs. Do not report synthetic mock responses as Qwen-generated text.

## Acceptance checklist

- Multiple HTTP requests share one ready engine and enter/leave batches independently.
- SSE emits incremental output before completion; non-streaming responses collect the same events.
- Client disconnect, cancellation during PD transfer, capacity exhaustion and shutdown reclaim all state.
- Startup failures prevent admission; no request triggers kernel compilation or graph capture.
- Tokenizer/chat-template fixtures match the pinned checkpoint; unsupported parameters fail clearly.
- No Python process starts and no Python interpreter is loaded by the selected frontend dependency path.
- Run these tests with mock mode first, then separately establish real Ascend inference correctness.

See [NPU integration responsibilities](npu-handoff.md) for the hardware boundary. This guide adds no frontend dependency or executable HTTP server to the repository.
