# Packed prefill through the vLLM Rust frontend

Date: 2026-10-17. Full 92B TP8/EP8, two resident histories, four-token
prefill chunks, single-token decode, fixed C8/BF16 cache selection. Native code
and the frozen bundle are unchanged from the [native qualification](../2026-10-17-packed-prefill/README.md).
Rust was cross-compiled locally and serving ran inside the remote NPU container.

## Scheduler and ABI

The Rust adapter accepts only the declared serial and packed native contracts.
Before readiness it checks native context, vocabulary, EOS, resident capacity,
chunk capacity and required function availability against the hashed manifest.
The packed entry point accepts exactly the compiled width; unsupported shapes
cannot cause an implicit alternate implementation or allocation.

Each round visits resident slots from a rotating cursor. A prefill chunk consumes
its actual number of input tokens from `max_num_batched_tokens`; a decode consumes
one. A short tail or a remaining budget below the compiled width uses the declared
single-token path. Only the last prompt row produces an output token. Cancellation
is checked between completed invocations and suppresses output from a cancelled
chunk. Independent slots preserve peers' state when a finished slot is reused.

The default budget is configured residents times compiled chunk width. An explicit
smaller budget remains valid and does not starve later slots. Invocations still
contain one request; this milestone does not claim fused multi-request kernels.

## Qualification

The HTTP harness verifies full expected fixture text, SSE/collected agreement,
exact prompt/output usage, six concurrent requests, disconnect recovery,
unsupported-sampling/context-overflow rejection, and graceful SIGTERM shutdown.
During a 504-token prompt it cancels a short stream and submits a replacement;
the replacement must finish before the long peer, and both completed outputs
must match their oracles.

- [Packed budget 8](packed-results.json), [all-rank audits](packed-audit.json),
  [scheduler log](packed-server.log).
- [Serial budget 1](serial-results.json), [all-rank audits](serial-audit.json),
  [scheduler log](serial-server.log).

The packed run must replay the prefill graph beyond startup; the one-token-budget
control must record only the warmup replay. All eight ranks must preserve the
frozen arena/workspace and have zero activation host transfers, allocation in
retained steps, internal fences and workspace growth. HTTP text-chunk intervals
are not equated with TPOT because a text chunk may combine output tokens.

In the mixed workload, packed prefill gives the 504-token request **4.663 s TTFT**
versus **9.863 s** with the serial budget. The short replacement finishes in
**582 ms** versus **738 ms**; both finish while the long peer is still active.
The short stream's text-chunk intervals are about 52 ms with packed peer work
versus 38 ms with serial peer work: larger nonpreemptible chunks improve prefill
throughput but increase the peer's decode interval. Future chunk tuning must
account for this latency tradeoff.

The packed run consumes 862 input tokens in 340 invocations per rank and records
175 prefill-graph replays (174 serving chunks plus warmup). The budget-1 control
uses 862 invocations and only the warmup prefill replay. Both use the same frozen
arena, **12,450,806,784 bytes/rank**, and **130,916,352 bytes/rank** shared workspace.

Local tests cover supported/rejected manifests, chunk/tail budgets 1/3/4/5/8,
short-request progress, exact final-chunk sampling, cancellation within a chunk,
resident reset isolation, rank-failure propagation and shutdown. Frontend tests
and strict frontend Clippy pass. These are targeted tests, not a production soak.

## Reproduction and identity

Run `qualify-program-http.py` with `--plan packed4-v2-bound`, the qualified native
library, `--max-num-seqs 2 --max-num-batched-tokens 8 --expect-prefill packed`, and
`--resident-cases resident-boundary-cases.json`. Repeat with token budget 1 and
`--expect-prefill serial`. The harness owns the server lifecycle and checks the
post-shutdown native audits. Remote outputs are `packed-http-v2` and
`packed-http-budget1-v2`.

Frontend SHA-256: `8b7a6b0f0d9b58cd74f0ff5970ba966ab7d545158ab9225511dd018a4cfed391`.
Native/program hashes are recorded in the native qualification linked above.

## Remaining work

Context remains 512. Dynamic shared paging, SWA page reclamation, learned DSA,
complete hybrid prefix-state restore and 64K correctness/performance qualification
are still required. Larger efficient chunks, batched decode, stochastic sampling,
overload status mapping, slow-client and sustained churn/failure tests remain
production gaps. This milestone does not claim production readiness.
