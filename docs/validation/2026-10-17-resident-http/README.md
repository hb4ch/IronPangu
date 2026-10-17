# Rust resident scheduling over the full 92B native engine

Date: 2026-10-17. vLLM 0.25.1 Rust frontend, full 46-layer 92B, TP8/EP8,
two compiled resident states, C8/BF16 cache selection unchanged. Rust was
cross-compiled locally; native serving and tests ran inside the NPU Docker container.

The program worker now continuously admits requests into free native slots and
rotates token work across prefill and decode. It resets only the admitted slot,
preserves each peer's history and position, and checks cancellation between
completed native invocations. The configured sequence count and context cannot
exceed compiled capacities. A smaller token-round budget rotates fairly rather
than starving higher-numbered slots.

## Verified behavior

[HTTP results](http-results.json) cover collected/SSE agreement and full expected
fixture text with exact usage counts, six concurrently submitted requests,
disconnect recovery, explicit rejection of unsupported sampling and context
overflow, and graceful SIGTERM shutdown. The existing native exact-token
[resident qualification](../2026-10-17-resident-states/README.md) uses the same
unchanged C++ library and frozen program.

During a 504-token prompt, the test disconnects a short decode stream and admits
a replacement. The replacement returns the expected complete output before the
long request completes; the long peer also matches its serial oracle. Short
request TTFT is **471.8 ms**, total **737.7 ms**, while long-prompt TTFT is
**9.923 s**, total **10.054 s**. The short stream's observed text-chunk intervals
are 37.75–38.11 ms. They include peer execution and are not labeled token latency:
text chunks can combine tokens. Isolated short-fixture SSE TTFT is 231.0–286.1 ms,
with roughly 18 ms text-chunk intervals.

The [scheduler log](server.log) records `max_active=2`, 47 mixed prefill/decode
rounds and 862 consumed input tokens. [All-rank audits](rank-audits.json) cover
all 862 token steps per rank: zero step allocations/internal completion boundaries,
zero activation host transfers, frozen arena **12,431,494,720 bytes/rank**, and
fixed workspace **130,916,352 bytes/rank** with zero growth. HBM snapshots include
device runtime overhead and are not a sampled lifetime peak.

Fourteen local frontend tests pass, including the actual scheduler loop under
injected rank failure, cancellation during invocation, shutdown with active/queued
work, queued disconnect, slot reuse and a one-token round budget.
Frontend-only strict Clippy also passes (`--no-deps -- -D warnings`).

The same HTTP independent-progress gate was run with `--max-num-seqs 1` as a
negative control. With submissions kept within the smaller queue's capacity,
the harness exits 1 with `long prefill blocked short request: no independent
resident progress`. Its [scheduler log](serial-control-server.log) confirms
`max_active=1`, `mixed_ticks=0` and orderly worker closure. An initial control run
submitted six requests into the one-resident/four-queued configuration and hit
overload earlier; the harness now bounds this isolation test's submissions so
queue rejection cannot mask the concurrency gate.

## Reproduction and identity

Use `scripts/qualify-program-http.py` with the locally cross-compiled `--server`,
qualified two-resident `--plan`, checkpoint, library, immutable short fixtures and
a fresh output directory. Add:

```text
--max-num-seqs 2 --max-num-batched-tokens 2
--resident-cases docs/validation/2026-10-17-resident-states/boundary-cases.json
```

SHA-256 of the tested Rust server:
`bad5b8a91a6c77ebe475bbb6707af0f793f1d8857404f6547ffdfbe76fdf633a`.
Native library and execution binary identities are unchanged from the linked
resident-state report. The audit summary records hashes of the original per-rank
preparation files.

## Limits

This is continuous admission and mixed-phase scheduling over **single-token
invocations**, not packed execution. Increasing the round token budget does not
create a multi-token kernel. The active context cap remains 512. Packed/chunked
prefill, multi-sequence kernels, dynamic page lending/SWA recycling, learned DSA,
hybrid prefix reuse, stochastic sampling qualification and the 64K release gates
remain unfinished. Queue capacity is bounded at four times resident count;
the current upstream backend-error mapping reports overload as HTTP 500, which
still needs a retryable overload status before production readiness.
