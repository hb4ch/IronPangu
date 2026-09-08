# Native continuous batching

The vLLM 0.25.1 Rust frontend accepts `--max-num-seqs N` (default 4, range 1..64) and `--max-num-batched-tokens N` (default 1024, positive). The first bounds resident sequences; the second bounds logical input tokens consumed in a scheduler iteration. Prompt chunks and decode tokens share that budget. Budgets smaller than the sequence limit are supported with round-robin selection. The waiting queue holds up to four times the sequence limit; excess admission returns a busy error. There is no Python engine.

```sh
./inferfabric-native-server --native examples/qwen35-2b-checkpoint.inferfabric \
  /data/p00603624/models/qwen35 native-build/libinferfabric_acl.so 0 18081 \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 16 \
  --gpu-memory-utilization 0.9
```

One fixed-shape model graph is captured for the configured sequence capacity. Every matmul processes that batch dimension and weights are shared across lanes. Each replay consumes at most one token from each selected sequence. A scheduler iteration may execute several prefill rounds, with at most one decode token per sequence in that iteration. Requests are admitted between replays and freed slots are reset before reuse. This is sequential per-sequence prefill, not a packed multi-token prefill kernel. Padding computes unused lanes, so the token budget counts selected tokens, not padded FLOPs. Large prefill budgets can increase decode latency in this reference implementation.

Each lane owns fixed KV cache ranges, convolution history, FP32 recurrent state and sampling/RNG buffers. Inactive lanes append only to a private dummy cache row. Captured D2D snapshots and conditional selection preserve their recurrent states bitwise. Runtime replay changes input buffers only; no model or sampler graph is recaptured as requests enter or leave. Per-request random tables are initialized outside capture. This implementation does not yet dynamically pool KV pages across sequence slots or implement native PD.

Startup HBM profiling covers all lanes, temporary state snapshots, sampler graphs and workspaces. It compares eager/captured logits and states, exercises the last context position, then explicitly checks that nonzero inactive lane states remain unchanged. Admission starts only after qualification.

Device validation: four slots at context 2048 passed with token budgets 16 and 2. Twelve concurrent requests matched isolated outputs with greedy and seeded non-greedy sampling, repetition/presence/frequency penalties. Mixed prefill/decode, queued requests, cancellation in prefill/decode, slot reuse, chat, SSE and overflow checks passed. The exact-copy graph observed 6,741,135,360 bytes HBM growth, including 3,763,862,208 resident weight bytes. Reports are in `validation-batching-2026-09-07`.

Inside the container, run `./inferfabric-native-server --batching-smoke-test http://127.0.0.1:18081`, plus the existing native and sampling smoke tests. Rust is cross-compiled locally; native C++ builds and NPU execution stay inside Docker.
