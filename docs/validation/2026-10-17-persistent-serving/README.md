# Persistent executable-program serving — 2026-10-17

The generic physical token executor now has a persistent C ABI and an adapter for
the vLLM 0.25.1 Rust frontend. One TP8/EP8 worker group loads the 92B checkpoint
once, prepares its invocations and ACLGraph, binds the frozen arena, and serves
successive requests. Request reset clears device continuation state on each
owning stream; it preserves weights, prepared invocations and captured addresses.

This is the S1 serial-serving baseline. It supports one active greedy request
and four queued requests. Prompt processing still executes one token at a time.
It does **not** implement multi-sequence decode, packed/chunked prefill, shared
page pools, prefix reuse or the learned long-context indexer. Those remain S2–S6.
The current compiled envelope is 512 tokens including generated output.

## Numerical and memory evidence

`native-results.json` and `tokens.jsonl` contain nine alternating arithmetic,
geography and Chinese requests in one engine lifetime: all **99 output tokens**
match the existing C8 fixture exactly. This comparison checks complete token
sequences, not per-layer tensors or logits. The same shared library is used by
the offline C++ probe and the Rust server; frozen plans identify the library
rather than the enclosing executable.

Short-prompt engine TTFT is 229.9–283.1 ms; per-request average TPOT is
17.82–18.42 ms. Prompts contain 12–15 tokens and outputs 8–17 tokens. TTFT includes
device-state reset and serial prompt processing, but excludes model loading and
HTTP. This is a regression measurement, not a long-context performance claim.

`rank-audits.json` summarizes the eight native audits with their original hashes.
Every rank uses a frozen arena and a fixed **130,916,352-byte workspace**, with
zero workspace growth. Each graph has 214 replays. Retained per-step allocation
and internal-fence records cover the final request (history is bounded by clearing
it at reset). Transfer audits cover the whole engine lifetime and record **zero
activation host transfers**. Parameter preparation and requested output/control
transfers are recorded separately. HBM snapshots are device-wide final usage and
growth relative to the startup baseline; they are not a sampled lifetime peak.

`http-results.json` records matching collected/SSE responses, three concurrently
submitted requests processed with isolated state, disconnect recovery, rejection
of unsupported stochastic sampling, and successful graceful SIGTERM shutdown.
Loopback SSE TTFT is 231.3–282.3 ms. Chunk intervals are preserved as observed;
they are not labeled token latency because a text chunk may combine tokens.

## Reproduction

Build Rust locally using `scripts/cross-frontend.sh`. In the NPU Docker container,
build CMake targets `inferfabric_program` and `program_engine_probe`.
Prepare workspace and arena plans against this exact shared library using the
existing discovery → fixed-workspace preparation → frozen binding procedure.
Plans prepared for the old diagnostic executable are not compatible.

Run `scripts/qualify-program-engine.py` with `--binary`, `--plan`, `--checkpoint`,
`--cases` and a fresh `--output` directory. `--prepare` is offline-only; production
open refuses bundles without complete frozen workspace and arena sidecars.

Run `scripts/qualify-program-http.py` with `--server`, `--plan`, `--checkpoint`,
`--library`, `--cases`, `--output` and `--port`. Python is only an offline HTTP
client; model execution and serving use C++ and the vLLM Rust frontend.

Start the service with:

```sh
INFERFABRIC_DECODE_GRAPH=1 inferfabric-server \
  --program BUNDLE CHECKPOINT /absolute/path/libinferfabric_program.so 18081
```

Source the CANN and custom-operator environments first. The listener is loopback
only. Unsupported sampling and requests outside the compiled context are rejected.
Cancellation is observed between token invocations; no in-flight graph is freed.

Initial HTTP qualification exposed two failures that were fixed before release:
the protocol's unsigned `top_k` default rejected `-1`, and asynchronous shutdown
signal handlers conflicted with native-runtime handling. Defaults are now checked
before device startup. Unix blocks termination signals before creating threads
and consumes them with `sigwait`, then drains the frontend and closes the worker
group from normal thread context.
