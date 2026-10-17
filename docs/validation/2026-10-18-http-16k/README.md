# 16K HTTP chunk scheduling — 2026-10-18

PASS on the existing `pooled8k-clean-frozen-plan`, OpenPangu 92B TP8/EP8,
vLLM Rust 0.25.1, two residents, 4,096-token batch budget. Qualifier exit 0.

[HTTP output/timing results](results.json), [scheduler log](server.log),
[eight-rank memory/graph audits](ranks.json).

- Baseline collected/streamed outputs and eight queued requests pass.
- Three 16,368-token retrieval prompts (early/middle/late needle) match reference
  output through EOS during concurrent cancellation and short-request recovery.
- In **all three cases**, the short request produces its first text before the
  long request produces its first text. Timestamps share one monotonic clock.
  The short request does not necessarily finish first: total completion order
  depends on both output lengths and is no longer used as a fairness proxy.
- 27 admissions, seven mixed prefill/decode ticks; 53,725 completed input tokens
  accounted by arithmetic histories; shared pages fully reclaimed on shutdown.
- All ranks keep the same 39,572,690,960-byte arena and 159,221,248-byte workspace,
  with zero workspace growth or activation host transfers. Decode graphs remain
  active; no long-prefill graph capture.

Long-request HTTP TTFT: 24.622/25.069/24.927 seconds. Short-request TTFT:
5.951/5.862/5.850 seconds. The short stream can have a later gap of about 11
seconds while mixed work executes; first-output progress is not a latency SLO.
This remains a performance optimization target. HTTP text chunk gaps are not
isolated token decode times; see the [native measurements](../2026-10-18-sequence-16k/README.md).

Reproduce using the [8K HTTP command](../2026-10-18-http-8k/README.md), changing
`--resident-cases` to `retrieval-16k-reference.json` and `--output` to a fresh
`pooled16k-http` directory. The updated checker requires short first-output
progress for prompts larger than two chunks, and rejects fixtures submitted
after the long request already produced output. Eight local checker tests pass.

This qualifies the tested 16K serving cases. 32K/64K correctness, APC and
sustained-load robustness remain open; the experimental context guard remains.
