# 64K-context HTTP serving qualification — 2026-10-18

PASS: vLLM Rust 0.25.1, OpenPangu 92B TP8/EP8, two active residents and a
4,096-token batch budget. The same frozen plan qualified at shorter contexts is
used unchanged. Qualifier exit 0.

[HTTP results](results.json), [scheduler log](server.log),
[eight-rank resource/graph audits](ranks.json).

- Three 65,520-token retrieval prompts (early/middle/late needle) match the
  independent reference through first EOS, with concurrent cancellation and
  short requests. The 65,536 context includes generation space.
- Every short request produces text before its long peer finishes prefill.
  Baseline collected/SSE agreement, eight queued concurrent requests, token
  usage, unsupported sampling rejection and context overflow rejection pass.
- 27 admissions, four mixed prefill/decode ticks, 201,181 completed input tokens
  accounted by arithmetic histories; all shared pages reclaimed on shutdown.
- All ranks retain arena 39,572,690,960 bytes and workspace 159,221,248 bytes.
  Workspace growth and activation host transfers are zero. Decode graphs replay;
  prefill stays uncaptured. [Native traces](../2026-10-18-sequence-64k/README.md)
  separately establish exactly sixteen forwards per long prompt.

Cold HTTP TTFT: **97.718/99.877/99.575 seconds**. Concurrent short-request TTFT:
**6.173/6.413/6.183 seconds**. These are observed mixed-workload timings, not
production SLOs. The client socket timeout was explicitly 300 seconds; it is
reported in the artifact and does not hide actual TTFT. Pure-decode native ABI
medians were 21.507 ms (one sequence) and 29.029 ms (two), measured separately;
HTTP text-chunk gaps include scheduling/prefill and are not isolated TPOT.

This closes the tested 64K uncached serving correctness gate. Cold-prefill
performance, APC, wider quality evaluation and sustained-load/slow-client/fault
handling qualification remain open. Greedy-only capability limits and the
experimental context opt-in remain explicit; this is not a production-ready
release claim.

Reproduce with the [16K HTTP setup](../2026-10-18-http-16k/README.md), using
`--resident-cases retrieval-64k-reference.json --request-timeout-seconds 300`
and a fresh output directory such as `pooled64k-http`. The backend and frozen
bundle are unchanged from the shared-pool milestone.
